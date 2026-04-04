from pathlib import Path
import spatialdata as sd
import pandas as pd
import tifffile
import numpy as np
import dask.dataframe as dd
import xarray as xr
from scipy.linalg import lstsq
from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from skimage.measure import regionprops_table
import anndata as ad
import dask
import zarr
import time
from spatialdata._io._utils import _resolve_zarr_store
from tqdm.notebook import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from spatialdata.transformations import (
    get_transformation,
    set_transformation,
    Identity,
    Sequence,
    Scale,
    Affine,
)

from spatialdata.models import (
    PointsModel,
    TableModel,
    Labels3DModel,
    Image3DModel,
)

_DIHEDRAL_TRANSFORMS = [
    ('identity',
        lambda img:       img,
        lambda x, y, H, W: (x,         y        )),
    ('rot90_ccw',
        lambda img: np.rot90(img, k=1),
        lambda x, y, H, W: (W - 1 - y, x        )),
    ('rot180',
        lambda img: np.rot90(img, k=2),
        lambda x, y, H, W: (W - 1 - x, H - 1 - y)),
    ('rot90_cw',
        lambda img: np.rot90(img, k=3),
        lambda x, y, H, W: (y,         H - 1 - x)),
    ('flipud',
        lambda img: np.flipud(img),
        lambda x, y, H, W: (x,         H - 1 - y)),
    ('fliplr',
        lambda img: np.fliplr(img),
        lambda x, y, H, W: (W - 1 - x, y        )),
    ('transpose',
        lambda img: np.flipud(np.rot90(img, k=1)),
        lambda x, y, H, W: (y,         x        )),
    ('anti_transpose',
        lambda img: np.fliplr(np.rot90(img, k=1)),
        lambda x, y, H, W: (W - 1 - y, H - 1 - x)),
]

def get_bigwarp_params(bigwarp_json_path):
    with open(bigwarp_json_path, 'r') as f:
        bw_json = json.load(f)
    for source, source_data in bw_json['Sources'].items():
        if source_data['isMoving']:
            if 'confocal' in source_data['uri'].lower():
                continue
            moving_image_path = Path(source_data['uri'])
            if moving_image_path.name=='?':
                moving_image_path = moving_image_path.parent
        else:
            reference_image_path = Path(source_data['uri'])
            if reference_image_path.name=='?':
                reference_image_path = reference_image_path.parent
    if 'zstack' in moving_image_path.name.lower() or 'z_stack' in moving_image_path.name.lower():
        moving_image = 'czstack'
        reference_image = 'xenium_section'
    transform_type = bw_json['Transform']['type']
    bigwarp_params = {'moving_image': moving_image,
                    'reference_image': reference_image,
                    'moving_image_path': moving_image_path,
                    'reference_image_path': reference_image_path,
                    'transform_type': transform_type}
    return bigwarp_params

def invert_xenium_y_landmarks(landmarks, landmarked_image_path):
    with tifffile.TiffFile(landmarked_image_path) as tif:
        landmarked_image_shape = tif.pages[0].shape
    full_y_size = landmarked_image_shape[0]
    landmarks['xenium_y'] = full_y_size - landmarks['xenium_y']
    return landmarks

def remove_landmark_buffer(landmarks, czstack_buffer=None, xenium_buffer=None):
    if czstack_buffer is not None:
        landmarks['czstack_y'] = landmarks['czstack_y'] - czstack_buffer.get('y', 0)
        landmarks['czstack_x'] = landmarks['czstack_x'] - czstack_buffer.get('x', 0)
        landmarks['czstack_z'] = landmarks['czstack_z'] - czstack_buffer.get('z', 0)
    if xenium_buffer is not None:
        landmarks['xenium_y'] = landmarks['xenium_y'] - xenium_buffer.get('y', 0)
        landmarks['xenium_x'] = landmarks['xenium_x'] - xenium_buffer.get('x', 0)
        landmarks['xenium_z'] = landmarks['xenium_z'] - xenium_buffer.get('z', 0)
    return landmarks

def get_section_landmarks(landmarks_path=None, dims_order=['x','y','z'], bigwarp_project_path=None, moving_img=None):
    if landmarks_path is not None:
        if bigwarp_project_path is not None:
            bigwarp_params = get_bigwarp_params(bigwarp_project_path)
        elif moving_img is not None:
            bigwarp_params = {'moving_image': moving_img}
        else:
            print("No BigWarp project path or moving image specified - assuming czstack was moving image")
            bigwarp_params = {'moving_image': 'czstack'}
        print(f"Loading landmarks from: {landmarks_path}")
        landmarks = pd.read_csv(landmarks_path, header=None)
        if bigwarp_params['moving_image']=='czstack':
            # Flatten the lists using + operator to concatenate them
            landmarks.columns = ['landmark_name', 'active'] + [f'czstack_{dim}' for dim in dims_order] + [f'xenium_{dim}' for dim in dims_order]
        else:
            landmarks.columns = ['landmark_name', 'active'] + [f'xenium_{dim}' for dim in dims_order] + [f'czstack_{dim}' for dim in dims_order]
    else:
        landmarks = None
        bigwarp_params = None

    return landmarks, bigwarp_params

def _normalize_bw_uri(uri):
    """Normalize a BigWarp source URI to a Path.

    Handles two common formats:
      - Plain paths:  /root/capsule/scratch/.../file.tif
      - URI paths:    file:/root/capsule/data/.../.zarr/?
    """
    uri = uri.strip()
    if uri.startswith('file:'):
        uri = uri[len('file:'):]   # strip 'file:' scheme
    p = Path(uri)
    if p.name == '?':
        p = p.parent               # strip trailing '/?' from zarr URIs
    return p

def _is_czstack(uri_or_name):
    """Return True if the URI/name suggests a cortical z-stack image."""
    s = uri_or_name.lower()
    return any(kw in s for kw in ('zstack', 'z_stack', 'gcamp'))

def extract_bigwarp_params(bigwarp_json_path, axes_order=['x', 'y', 'z']):
    """Parse a BigWarp project JSON and return source metadata and landmark DataFrame.

    Supports two project formats:
      * Format A (e.g. 01.json): landmarks stored in separate CSV files;
        movingPoints / fixedPoints are empty; URIs may use 'file:.../?' notation.
      * Format B (e.g. section_1_..._warp_project.json): landmarks embedded in
        the JSON as movingPoints / fixedPoints arrays.

    Parameters
    ----------
    bigwarp_json_path : str or Path
    axes_order : list of str
        Ordered axis labels matching the coordinate positions stored in each
        BigWarp point, e.g. ['x', 'y', 'z'] means pt[0] → x, pt[1] → y,
        pt[2] → z.  Change to e.g. ['z', 'y', 'x'] when BigWarp stores
        coordinates in a different order.

    Returns
    -------
    bigwarps_params : dict
    lm_df : pd.DataFrame or None
        None when no landmarks are stored in the JSON (Format A).
        Columns follow the pattern czstack_{ax} / xenium_{ax} for each ax
        in axes_order.
    """
    with open(bigwarp_json_path, 'r') as f:
        bw_json = json.load(f)

    moving_image_name = moving_image_path_raw = moving_image_dataset = None
    target_image_name = target_image_path_raw = target_image_dataset = None

    for source_data in bw_json['Sources'].values():
        uri  = source_data['uri']
        name = source_data.get('name', '')

        # Skip auxiliary confocal sources present in some 756772-style projects
        if 'confocal' in uri.lower(): #or 'confocal' in name.lower():
            continue

        dataset_type = 'czstack' if _is_czstack(uri) or _is_czstack(name) else 'xenium_section'

        if source_data['isMoving']:
            moving_image_name     = name
            moving_image_path_raw = uri
            moving_image_dataset  = dataset_type
        else:
            target_image_name     = name
            target_image_path_raw = uri
            target_image_dataset  = dataset_type

    # ── Landmark extraction ───────────────────────────────────────────────
    lm_df          = None
    bw_tf_type     = None
    transform_ndim = None

    if 'Transform' in bw_json:
        bw_lm      = bw_json['Transform'].get('landmarks', {})
        bw_tf_type = bw_json['Transform'].get('type')
        transform_ndim = bw_lm.get('numDimensions')

        moving_pts = bw_lm.get('movingPoints', [])
        fixed_pts  = bw_lm.get('fixedPoints',  [])
        lm_names   = bw_lm.get('names',  [])
        lm_active  = bw_lm.get('active', [])

        # Only build a DataFrame when landmarks are actually present in the JSON
        if len(moving_pts) > 0 and len(fixed_pts) > 0:
            czstack_pts = moving_pts if moving_image_dataset == 'czstack' else fixed_pts
            xenium_pts  = moving_pts if moving_image_dataset == 'xenium_section' else fixed_pts
            lm_df = pd.DataFrame({
                'name':   lm_names,
                'active': lm_active,
                # axes_order drives both the column name and the index into each point
                **{f'czstack_{ax}': [pt[i] for pt in czstack_pts] for i, ax in enumerate(axes_order)},
                **{f'xenium_{ax}':  [pt[i] for pt in xenium_pts]  for i, ax in enumerate(axes_order)},
            })
            lm_df = lm_df.loc[lm_df['active']==True]

    bigwarps_params = {
        'moving_image_dataset':  moving_image_dataset,
        'moving_image_name':     moving_image_name,
        'moving_image_path':     moving_image_path_raw,
        'target_image_dataset':  target_image_dataset,
        'target_image_name':     target_image_name,
        'target_image_path':     target_image_path_raw,
        'transform_type':        bw_tf_type,
        'transform_ndim':        transform_ndim,
    }
    return bigwarps_params, lm_df


def _thumbnail(img, max_size=128):
    """Downsample to at most max_size × max_size by uniform strided slicing."""
    sr = max(1, img.shape[0] // max_size)
    sc = max(1, img.shape[1] // max_size)
    return img[::sr, ::sc]

def _ncc(a, b):
    """Normalised cross-correlation in [-1, 1]; 1 = perfect match."""
    a = a.astype(float); b = b.astype(float)
    a -= a.mean();        b -= b.mean()
    denom = np.sqrt((a**2).sum() * (b**2).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def plot_img_landmark_transforms(sdata_img, 
                                lm_img, 
                                best_transformed_img, 
                                transform_info, 
                                landmarks=None,
                                landmarks_out=None, 
                                section=None, 
                                save_path=None,
                                show=True):
    """
    Parameters
    ----------
    show : bool
        If True, call plt.show() to display inline.  Set to False when running
        in a background thread (matplotlib is not thread-safe) or in batch mode
        to avoid blocking; the figure is saved to ``save_path`` and then closed.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(lm_img, cmap='gray')
    if landmarks is not None:
        axes[0].scatter(landmarks['xenium_x'], landmarks['xenium_y'],
                        c='red', s=15, zorder=5)
    axes[0].set_title('Landmarked image + original landmarks')

    axes[1].imshow(best_transformed_img, cmap='gray')
    if landmarks is not None:
        axes[1].scatter(landmarks['xenium_x'], landmarks['xenium_y'],
                    c='red', s=15, zorder=5)
    axes[1].set_title(f"sdata '{transform_info['matched_level']}' after '{transform_info['transform_name']}' "
                    f"(NCC={transform_info['ncc_score']:.3f})")

    axes[2].imshow(sdata_img, cmap='gray')
    if landmarks_out is not None:
        axes[2].scatter(landmarks_out['sdata_x'], landmarks_out['sdata_y'],
                        c='red', s=15, zorder=5)
    axes[2].set_title(f"sdata '{transform_info['matched_level']}' + transformed landmarks")

    plt.tight_layout()
    if section is not None:
        plt.suptitle(f"Section: {section}")
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

def find_landmarked_img_transforms(landmarked_image_path, 
                                    sdata_path, 
                                    landmarks,
                                    lm_img_ch_n=1,
                                    thumbnail_size=128,
                                    plot_imgs=True, 
                                    save_imgs_path=None):
    """
    Find how the BigWarp landmarked image relates to the spatialdata morphology
    image, then remap landmark coordinates into spatialdata pixel space.

    Steps
    -----
    1. Match the landmarked TIFF to the correct pyramid level of
       ``morphology_focus`` by comparing shapes (rotation-aware).
    2. Score all 8 dihedral (rotation + flip) transforms of the sdata image
       against the landmarked image using normalised cross-correlation on
       downsampled thumbnails.
    3. Apply the inverse of the best transform to the landmark
       (xenium_x, xenium_y) columns, adding (sdata_x, sdata_y) columns.

    Parameters
    ----------
    landmarked_image_path : Path or str
        TIFF used for BigWarp landmarking.
    sdata_path : Path or str
        Section ``.zarr`` spatialdata path.
    landmarks : pd.DataFrame or None
        Must contain ``xenium_x`` and ``xenium_y`` columns (lm image pixel
        coordinates from the BigWarp JSON).  Pass ``None`` when no landmarks
        are available; the function will still return ``transform_info`` but
        ``landmarks_out`` will be ``None``.
    plot_imgs : bool
        Display the side-by-side plot inline.  The figure is always saved to
        ``save_imgs_path`` when that argument is provided, regardless of this
        flag.
    lm_img_ch_n : int
        Channel index to read from the landmarked TIFF for pixel comparison.
    thumbnail_size : int
        Max edge length for NCC comparison thumbnails.

    Returns
    -------
    transform_info : dict
        ``matched_level``, ``matched_level_idx``, ``transform_name``,
        ``ncc_score``, ``all_scores``, ``sdata_img_shape``.
    landmarks_out : pd.DataFrame or None
        Copy of ``landmarks`` with ``sdata_x`` and ``sdata_y`` added,
        or ``None`` when no landmarks were provided.
    """
    # ── 1. Load landmarked TIFF ───────────────────────────────────────────
    with tifffile.TiffFile(landmarked_image_path) as tif:
        lm_stack = tif.asarray()
    lm_img = lm_stack if lm_stack.ndim == 2 else lm_stack[lm_img_ch_n]

    # ── 2. Find matching pyramid level ────────────────────────────────────
    sdata = sd.read_zarr(sdata_path)
    morph = sdata['morphology_focus']

    matched_level = matched_idx = None
    for lvl_idx, (lvl, s_l) in enumerate(morph.items()):
        level_shape = tuple(s_l.image.shape[-2:])          # (H, W) ignoring c
        if set(level_shape) == set(lm_img.shape):           # rotation-agnostic
            matched_level = lvl
            matched_idx   = lvl_idx
            break

    if matched_level is None:
        avail = [tuple(s_l.image.shape[-2:]) for _, s_l in morph.items()]
        raise ValueError(
            f"No pyramid level matched lm_img shape {lm_img.shape}. "
            f"Available shapes: {avail}"
        )

    sdata_img = np.asarray(sd.get_pyramid_levels(morph, n=matched_idx)[lm_img_ch_n])
    H, W = sdata_img.shape

    # ── 3. Score dihedral transforms ──────────────────────────────────────
    lm_thumb = _thumbnail(lm_img.astype(float), thumbnail_size)
    scores   = {}
    for name, img_fn, _ in _DIHEDRAL_TRANSFORMS:
        candidate = img_fn(sdata_img)
        if candidate.shape != lm_img.shape:
            continue                             # wrong output shape → skip
        scores[name] = _ncc(_thumbnail(candidate.astype(float), thumbnail_size),
                             lm_thumb)

    best_name  = max(scores, key=scores.get)
    best_score = scores[best_name]
    best_img_fn, best_lm_fn = next(
        (img_fn, lm_fn)
        for name, img_fn, lm_fn in _DIHEDRAL_TRANSFORMS
        if name == best_name
    )

    print(f"Matched pyramid level : {matched_level}  (sdata shape {H}×{W})")
    print(f"Best transform        : '{best_name}'  (NCC = {best_score:.4f})")
    print("All NCC scores        :", {k: f"{v:.4f}" for k, v in scores.items()})

    # ── 4. Map landmarks into sdata pixel space ───────────────────────────
    if landmarks is None or landmarks.empty:
        print("  No landmarks for this section — skipping coordinate remapping")
        landmarks_out = None
    else:
        landmarks_out = landmarks.copy()
        sdata_coords  = np.array([
            best_lm_fn(float(x), float(y), H, W)
            for x, y in zip(landmarks['xenium_x'], landmarks['xenium_y'])
        ])
        landmarks_out['sdata_x'] = sdata_coords[:, 0]
        landmarks_out['sdata_y'] = sdata_coords[:, 1]

    transform_info = {
        'matched_level':     matched_level,
        'matched_level_idx': matched_idx,
        'matched_level_transforms': get_transformation(sd.get_pyramid_levels(morph, n=matched_idx), get_all=True),
        'pixel_size': sdata['table'].uns['section_metadata']['pixel_size'],
        'transform_name':    best_name,
        'ncc_score':         best_score,
        'all_scores':        scores,
        'sdata_img_shape':   (H, W),
    }

    # Always save the diagnostic image when a path is given; only show when
    # plot_imgs=True (avoids blocking plt.show() inside worker threads).
    if plot_imgs or save_imgs_path is not None:
        section_n = sdata['table'].obs['section'].iloc[0] if 'section' in sdata['table'].obs else None
        plot_img_landmark_transforms(sdata_img=sdata_img, 
                                    lm_img=lm_img,
                                    best_transformed_img=best_img_fn(sdata_img),  
                                    transform_info=transform_info,
                                    landmarks=landmarks,
                                    landmarks_out=landmarks_out, 
                                    section=section_n, 
                                    save_path=save_imgs_path,
                                    show=plot_imgs)

    return transform_info, landmarks_out

def parse_landmarks(landmarks, transform_info):
    landmarks.drop(columns=['sdata_x','sdata_y'], inplace=True)
    full_scale_pixel_size = transform_info['pixel_size']
    landmarks = landmarks.rename(columns={'xenium_x': 'x', 'xenium_y': 'y', 'xenium_z': 'z'})
    landmarks = PointsModel.parse(landmarks)
    landmarks.attrs['transform'] = {}
    landmarks.attrs['transform']['global'] = transform_info['matched_level_transforms']
    landmarks.attrs['transform']['microns'] = Sequence([transform_info['matched_level_transforms'], Scale([full_scale_pixel_size, full_scale_pixel_size], axes=('x', 'y'))])
    return landmarks

def _process_section(s_n, paths, alignment_params):
    """Process landmarks for one section. Returns (s_n, landmarks) or (s_n, None).

    Designed to run concurrently: each section reads its own independent files
    (BigWarp JSON, landmarked TIFF, zarr), so there are no shared-state conflicts.
    plot_imgs=False keeps matplotlib out of worker threads (not thread-safe).
    """
    bigwarp_folder_path = (paths['data_root']
                           / alignment_params['bigwarp_projects_folder']
                           / alignment_params['bigwarp_projects_names_fn'](s_n))
    landmarked_image_path = (paths['data_root']
                              / alignment_params['landmarked_images_folder']
                              / alignment_params['landmarked_images_names_fn'](s_n))
    sdata_path = paths['sdata_path'] / f"section_{s_n}.zarr"

    # ── Early-exit: check all files before any heavy I/O ──────────────────
    if not bigwarp_folder_path.exists():
        print(f"  Section {s_n}: BigWarp project not found at {bigwarp_folder_path}, skipping")
        return s_n, None
    if not landmarked_image_path.exists():
        print(f"  Section {s_n}: landmarked image not found at {landmarked_image_path}, skipping")
        return s_n, None

    bigwarp_params, lm_df = extract_bigwarp_params(bigwarp_folder_path)
    if lm_df is None or lm_df.empty:
        print(f"  Section {s_n}: no landmarks in BigWarp project, looking for landmarks path...")
        if alignment_params.get('landmarks_folder', None) is None:
            return s_n, None
        landmarks_path = (paths['data_root']
                           / alignment_params['landmarks_folder']
                           / alignment_params['landmarks_folder_names_fn'](s_n))
        if not landmarks_path.exists():
            print(f"  Section {s_n}: landmarks not found at {landmarks_path}, skipping")
            return s_n, None
        print(f"  Manually transforming landmarks...")
        landmarked_image_path = paths['data_root'] / alignment_params['landmarked_images_folder'] / alignment_params['landmarked_images_names_fn'](s_n)
        formatted_landmarks = manual_landmarks_transform(s_n=s_n, 
                                                        sdata_path=sdata_path, 
                                                        landmarks_path=landmarks_path,
                                                        landmarked_image_path=landmarked_image_path, 
                                                        alignment_params=alignment_params, 
                                                        bigwarp_params=bigwarp_params)

    else:
        print(f"Formatting section {s_n} landmarks")
        transform_info, landmarks_out = find_landmarked_img_transforms(
            landmarked_image_path=landmarked_image_path,
            sdata_path=sdata_path,
            landmarks=lm_df,
            plot_imgs=False,                         
            save_imgs_path=alignment_params['validation_images_folder'] / f"section_{s_n}.png",
        )
        if landmarks_out is None:
            return s_n, None
        else:
            formatted_landmarks = parse_landmarks(landmarks_out, transform_info)

    return s_n, formatted_landmarks


def manual_landmarks_transform(s_n, sdata_path, landmarks_path, landmarked_image_path, alignment_params, bigwarp_params, dims_order=['x','y','z']):
    starting_lm_df = pd.read_csv(landmarks_path, header=None)
    if bigwarp_params['moving_image_dataset']=='czstack':
        starting_lm_df.columns = ['landmark_name', 'active'] + [f'czstack_{dim}' for dim in dims_order] + [f'xenium_{dim}' for dim in dims_order]
    else:
        starting_lm_df.columns = ['landmark_name', 'active'] + [f'xenium_{dim}' for dim in dims_order] + [f'czstack_{dim}' for dim in dims_order]

    landmarks_out = starting_lm_df.copy()
    if alignment_params.get('czstack_buffer', None) is not None:
        landmarks_out = remove_landmark_buffer(landmarks_out, czstack_buffer=alignment_params['czstack_buffer'])   
    
    if alignment_params.get('fix_cropped_landmarks', None) and landmarked_image_path is not None:
        landmarks_out, landmarks_info = get_landmarked_image_props(
                            landmarked_image_path, 
                            sdata_path, 
                            landmarks_out, 
                            s_n,
                            invert_y=alignment_params.get('invert_lm_y', None))
    elif alignment_params.get('invert_lm_y', None) and landmarked_image_path is not None:
        landmarks_out = invert_xenium_y_landmarks(landmarks_out, landmarked_image_path)
    landmarks_out = landmarks_out.rename(columns={'xenium_x': 'x', 'xenium_y': 'y', 'xenium_z': 'z'})
    full_scale_pixel_size = landmarks_info.get('full_scale_pixel_size')
    parsed_lm = PointsModel.parse(landmarks_out)
    set_transformation(parsed_lm, Identity(), to_coordinate_system='global')
    set_transformation(
        parsed_lm,
        Scale([full_scale_pixel_size, full_scale_pixel_size], axes=('x', 'y')),
        to_coordinate_system='microns'
    )

    return parsed_lm

def get_section_landmarks_threads(xenium_section_ns, paths, alignment_params, n_workers=None):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    if n_workers is None:
        n_workers = min(4, len(xenium_section_ns))
    print(f"Processing {len(xenium_section_ns)} sections with {n_workers} parallel workers …")
    sections_landmarks={}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_process_section, s_n, paths, alignment_params): s_n
            for s_n in xenium_section_ns
        }
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc='Processing sections', unit='section'):
            s_n, lm = fut.result()
            if lm is not None:
                sections_landmarks[s_n] = lm
    return sections_landmarks

# ── Affine comparison helpers ─────────────────────────────────────────────────

def _to_3x3(mat):
    """Extract the 2D xy affine (3×3) from either a 3×3 or 4×4 matrix.

    Handles two axis conventions:
      - 3×3: already a 2D homogeneous affine [[a,b,tx],[c,d,ty],[0,0,1]]
      - 4×4: extract rows/cols for x,y + homogeneous  →  indices [0,1,3]
    """
    mat = np.asarray(mat, dtype=float)
    if mat.shape == (3, 3):
        return mat
    if mat.shape == (4, 4):
        idx = [0, 1, 3]   # x-row, y-row, homogeneous row; same for cols
        return mat[np.ix_(idx, idx)]
    raise ValueError(f"Expected 3×3 or 4×4 matrix, got {mat.shape}")

def _swap_xy_3x3(m):
    """Swap the x and y axes of a 3×3 2D affine matrix.

    Equivalent to pre- and post-multiplying by the permutation [[0,1,0],[1,0,0],[0,0,1]].
    This converts between (row=x, col=y) and (row=y, col=x) conventions.
    """
    P = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 1]], dtype=float)
    return P @ m @ P

def _decompose_2d_affine(m):
    """Decompose a 3×3 2D affine into rotation (deg), x/y scale, shear, translation."""
    a, b, tx = m[0, 0], m[0, 1], m[0, 2]
    c, d, ty = m[1, 0], m[1, 1], m[1, 2]
    sx   = np.sqrt(a**2 + c**2)
    sy   = np.sqrt(b**2 + d**2)
    angle = np.degrees(np.arctan2(c, a))
    return {'angle_deg': angle, 'scale_x': sx, 'scale_y': sy,
            'tx': tx, 'ty': ty}

def _get_matrix(affine_obj):
    """Return a plain numpy array from a spatialdata Affine object or ndarray."""
    if hasattr(affine_obj, 'matrix'):
        return np.asarray(affine_obj.matrix)
    return np.asarray(affine_obj)

def compare_affines(derived_affine, calculated_affines, section_n,
                    match_keys=None, print_matrices=True):
    """
    Compare a pre-existing (derived) 2D affine against all entries in a
    calculated affines dict, reporting residuals and geometric decomposition.

    The function checks both the direct comparison and the xy-swapped version
    of each calculated affine, since BigWarp / spatialdata can use different
    (row, col) ↔ (x, y) conventions.

    Parameters
    ----------
    derived_affine : array-like, shape (3, 3)
        Reference affine in 2D homogeneous form.
    calculated_affines : dict
        Keys → spatialdata Affine or ndarray (3×3 or 4×4).
    match_keys : list of str, optional
        Subset of keys to compare.  Defaults to all keys.
    print_matrices : bool
        Print full matrix for the best-matching key.
    """
    derived = _to_3x3(np.asarray(derived_affine, dtype=float))
    keys    = match_keys or list(calculated_affines.keys())

    print("=" * 70)
    print(f"SECTION: {section_n}")
    print("Affine comparison  (Frobenius norm of residual matrix)")
    print("=" * 70)

    best_key   = None
    best_norm  = np.inf
    best_swapped = False
    results    = {}

    for key in keys:
        mat  = _to_3x3(_get_matrix(calculated_affines[key]))
        norm_direct  = np.linalg.norm(derived - mat)
        norm_swapped = np.linalg.norm(derived - _swap_xy_3x3(mat))
        use_swap     = norm_swapped < norm_direct
        best_norm_k  = min(norm_direct, norm_swapped)
        results[key] = {'norm': best_norm_k, 'xy_swapped': use_swap,
                        'norm_direct': norm_direct, 'norm_swapped': norm_swapped}

        flag = " ← xy-swapped" if use_swap else ""
        print(f"  {key:<42s}  residual = {best_norm_k:8.4f}{flag}")

        if best_norm_k < best_norm:
            best_norm    = best_norm_k
            best_key     = key
            best_swapped = use_swap

    print()
    print(f"Best match : '{best_key}'  (residual norm = {best_norm:.4f}"
          + (", xy-axes swapped)" if best_swapped else ")"))

    # ── Geometric decomposition ───────────────────────────────────────────
    best_mat  = _to_3x3(_get_matrix(calculated_affines[best_key]))
    if best_swapped:
        best_mat = _swap_xy_3x3(best_mat)

    d_dec = _decompose_2d_affine(derived)
    c_dec = _decompose_2d_affine(best_mat)

    print()
    print(f"{'Parameter':<14} {'Derived':>14} {'Calculated':>14} {'Δ':>12}")
    print("-" * 58)
    for param in ['angle_deg', 'scale_x', 'scale_y', 'tx', 'ty']:
        dv, cv = d_dec[param], c_dec[param]
        print(f"  {param:<12} {dv:>14.5f} {cv:>14.5f} {dv - cv:>12.5f}")

    if print_matrices:
        print()
        print("Derived affine (3×3):")
        print(np.array2string(derived, precision=6, suppress_small=True))
        print(f"\nCalculated '{best_key}' (3×3{', xy-swapped' if best_swapped else ''}):")
        print(np.array2string(best_mat, precision=6, suppress_small=True))
        print()
        print("Residual matrix (derived − calculated):")
        print(np.array2string(derived - best_mat, precision=6, suppress_small=True))

    print("=" * 70)
    return results

def get_landmarked_image_props(landmarked_image_path, sdata, landmarks, 
                                section_n, invert_y=False):
    with tifffile.TiffFile(landmarked_image_path) as tif:
        landmarked_image_shape = tif.pages[0].shape
    print(f"Landmarked image shape: {landmarked_image_shape}")
    if isinstance(sdata, str) or isinstance(sdata, Path):
        sdata = sd.read_zarr(sdata)
    bbox_xmin = 0
    bbox_ymin = 0

    cell_labels = sd.get_pyramid_levels(sdata['cell_labels'], n=0)
    section_shape_y, section_shape_x = cell_labels.data.shape

    bbox_dict = sdata['table'].uns.get('sections_bboxes', None)

    if bbox_dict is not None and str(section_n) in bbox_dict:
        # Paired: invert within image space FIRST, then scale to full slide + offset
        if invert_y:
            landmarks['xenium_y'] = landmarked_image_shape[0] - landmarks['xenium_y']
        
        section_bbox = bbox_dict[str(section_n)]
        bbox_xmin = section_bbox['x_min']
        bbox_ymin = section_bbox['y_min']
        full_slide_shape_y = np.max([bbox['y_max'] for bbox in bbox_dict.values()])
        full_slide_shape_x = np.max([bbox['x_max'] for bbox in bbox_dict.values()])
        scale_factor_y = full_slide_shape_y / landmarked_image_shape[0]
        scale_factor_x = full_slide_shape_x / landmarked_image_shape[1]
        print(f"Paired section — full slide shape: {[full_slide_shape_y, full_slide_shape_x]}")
        print(f"Image downsampled by: (yx) [{scale_factor_y:.4f}, {scale_factor_x:.4f}]")
        print(f"Section bbox offset: {section_bbox}")
    else:
        # Standalone: scale to full section first, then invert within section space
        scale_factor_y = section_shape_y / landmarked_image_shape[0]
        scale_factor_x = section_shape_x / landmarked_image_shape[1]
        print(f"Standalone section — section array shape: {[section_shape_y, section_shape_x]}")
        print(f"Image downsampled by: (yx) [{scale_factor_y:.4f}, {scale_factor_x:.4f}]")

    landmarks['xenium_x'] = landmarks['xenium_x'] * scale_factor_x
    landmarks['xenium_y'] = landmarks['xenium_y'] * scale_factor_y
    landmarks['xenium_x'] = landmarks['xenium_x'] - bbox_xmin
    landmarks['xenium_y'] = landmarks['xenium_y'] - bbox_ymin

    if bbox_dict is None or str(section_n) not in bbox_dict:
        # Standalone: invert after scaling, using full section height
        if invert_y:
            landmarks['xenium_y'] = section_shape_y - landmarks['xenium_y']

    landmarks_tf_info = {'scale_factor_x': scale_factor_x, 
                         'scale_factor_y': scale_factor_y, 
                         'bbox': bbox_dict[str(section_n)] if bbox_dict is not None and str(section_n) in bbox_dict else None,
                         'fullres_pixel_size': sdata['table'].uns['section_metadata']['pixel_size']}

    return landmarks, landmarks_tf_info

def format_landmarks(sdata_path, 
                    landmarks_path, 
                    section_n, 
                    bigwarp_project_paths=None,
                    moving_img=None,
                    czstack_buffer=None,
                    invert_lm_y=False, 
                    fix_cropped_landmarks=False,
                    landmarked_image_path=None,
                    dims_order=['x','y','z']):
    sdata = sd.read_zarr(sdata_path)
    landmarks, bigwarp_params = get_section_landmarks(
                                    landmarks_path=landmarks_path, 
                                    bigwarp_project_path=bigwarp_project_paths,
                                    moving_img=moving_img,
                                    dims_order=dims_order)

    if czstack_buffer is not None:
        landmarks = remove_landmark_buffer(landmarks, 
                                            czstack_buffer=czstack_buffer)

    # if fix_cropped_landmarks and landmarked_image_path is not None:
        # invert_y is handled inside get_landmarked_image_props at the right point
    landmarks, landmarks_tf_info = get_landmarked_image_props(
                        landmarked_image_path, 
                        sdata, 
                        landmarks, 
                        section_n,
                        invert_y=invert_lm_y)
    # elif invert_lm_y and landmarked_image_path is not None:
    #     # sections where fix_cropped_landmarks=False
    #     landmarks = invert_xenium_y_landmarks(landmarks, landmarked_image_path)

    # Adjust landmarks resolutions
    full_scale_pixel_size = landmarks_tf_info['fullres_pixel_size']
    landmarks = landmarks.rename(columns={'xenium_x': 'x', 'xenium_y': 'y', 'xenium_z': 'z'})
    landmarks = PointsModel.parse(landmarks)
    set_transformation(landmarks, Identity(), to_coordinate_system='global')
    set_transformation(
        landmarks,
        Scale([full_scale_pixel_size, full_scale_pixel_size], axes=('x', 'y')),
        to_coordinate_system='microns'
    )
    return landmarks

