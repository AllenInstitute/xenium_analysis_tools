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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Matplotlib's pyplot interface is NOT thread-safe.  All plot functions that
# may be called from worker threads must acquire this lock before touching any
# plt.* state.
_MATPLOTLIB_LOCK = threading.Lock()

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
# O(1) name → (img_fn, lm_fn) lookup — avoids a second linear scan over the list.
_DIHEDRAL_BY_NAME = {name: (img_fn, lm_fn) for name, img_fn, lm_fn in _DIHEDRAL_TRANSFORMS}

def invert_xenium_y_landmarks(landmarks, landmarked_image_path):
    with tifffile.TiffFile(landmarked_image_path) as tif:
        arr = tif.asarray()
    full_y_size = int(arr.shape[-2])   # robust to (C,H,W) and (H,W) layouts
    landmarks = landmarks.copy()
    landmarks['xenium_y'] = full_y_size - landmarks['xenium_y']
    return landmarks

def remove_landmark_buffer(landmarks, czstack_buffer=None, xenium_buffer=None):
    landmarks = landmarks.copy()
    if czstack_buffer is not None:
        landmarks['czstack_y'] = landmarks['czstack_y'] - czstack_buffer.get('y', 0)
        landmarks['czstack_x'] = landmarks['czstack_x'] - czstack_buffer.get('x', 0)
        landmarks['czstack_z'] = landmarks['czstack_z'] - czstack_buffer.get('z', 0)
    if xenium_buffer is not None:
        landmarks['xenium_y'] = landmarks['xenium_y'] - xenium_buffer.get('y', 0)
        landmarks['xenium_x'] = landmarks['xenium_x'] - xenium_buffer.get('x', 0)
        landmarks['xenium_z'] = landmarks['xenium_z'] - xenium_buffer.get('z', 0)
    return landmarks

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

def extract_bigwarp_params(bigwarp_json_path, axes_order=None):
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
    if axes_order is None:
        axes_order = ['x', 'y', 'z']
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
            lm_df = lm_df.loc[lm_df['active']].copy()
            # Xenium sections are 2D; BigWarp can record sub-pixel z-offsets in its 3D
            # coordinate system (e.g. -0.18 px) that are pure numerical noise.  Zero them
            # out so they don't corrupt tilt_affines() z-plane estimation downstream.
            if 'xenium_z' in lm_df.columns:
                lm_df['xenium_z'] = 0.0

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
    """Normalized cross-correlation in [-1, 1]; 1 = perfect match."""
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
    with _MATPLOTLIB_LOCK:
      _plot_img_landmark_transforms_inner(
          sdata_img, lm_img, best_transformed_img, transform_info,
          landmarks, landmarks_out, section, save_path, show)

def _plot_img_landmark_transforms_inner(sdata_img, lm_img, best_transformed_img,
                                        transform_info, landmarks, landmarks_out,
                                        section, save_path, show):
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


def plot_manual_landmark_transforms(landmarks_before,
                                    landmarks_after,
                                    landmarked_image_path,
                                    sdata,
                                    ch=0,
                                    landmarks_tf_info=None,
                                    section=None,
                                    save_path=None,
                                    show=True):
    """Visualize the manual landmark coordinate transform.

    Two-panel figure:
      Left  — landmarked TIFF with original (xenium_x, xenium_y) landmarks in
              landmarked-image pixel space.
      Right — sdata ``morphology_focus`` (full resolution) with transformed
              (xenium_x, xenium_y) landmarks mapped into sdata pixel space,
              after all scaling / inversion / bbox-offset corrections.

    Parameters
    ----------
    landmarks_before : pd.DataFrame
        CSV-loaded landmark table with ``xenium_x``, ``xenium_y`` columns in
        landmarked-image pixel space (before any coordinate correction).
    landmarks_after : pd.DataFrame
        Same table after coordinate corrections (scaling, inversion, offset)
        but *before* the column rename to ``x``/``y``.  Still has
        ``xenium_x``/``xenium_y`` columns, now in sdata pixel space.
    landmarked_image_path : Path or str
        TIFF that was used for BigWarp landmarking.
    sdata : SpatialData
        Section SpatialData object (already loaded).
    landmarks_tf_info : dict or None
        Dict returned by ``get_landmarked_image_props``.  Used to annotate
        scale factors and bbox offset on the right panel.
    section : int or str, optional
        Section number used as figure suptitle.
    save_path : Path or str, optional
    show : bool
        Display inline.  Set ``False`` when called from a worker thread.
    """
    # ── Load landmarked TIFF ──────────────────────────────────────────────
    # Handle both channel-first (C, H, W) and channel-last (H, W, C) TIFFs.
    # A small first dimension (≤8) reliably indicates (C, H, W); a large one
    # means the first axis is height, i.e. (H, W) or (H, W, C).
    with tifffile.TiffFile(landmarked_image_path) as tif:
        lm_stack = tif.asarray()
    # Disambiguate (C, H, W) vs (H, W, C): a small leading dimension (≤8)
    # means channel-first; a large one means height is axis 0.
    if lm_stack.ndim == 2:
        lm_img = lm_stack
    elif lm_stack.ndim == 3 and lm_stack.shape[0] <= 8:   # (C, H, W)
        lm_img = lm_stack[min(ch, lm_stack.shape[0] - 1)]
    elif lm_stack.ndim == 3:                               # (H, W, C)
        lm_img = lm_stack[:, :, min(ch, lm_stack.shape[2] - 1)]
    else:
        lm_img = lm_stack.reshape(-1, lm_stack.shape[-2], lm_stack.shape[-1])[ch]

    # ── Load sdata morphology at a downsampled level for speed ────────────
    # Full-res (n=0) can be 4k–8k px and is slow to materialise; a lower
    # level is sufficient for a diagnostic overlay. Landmarks are in full-res
    # pixel space, so we compute scale factors to map them to display space.
    morph    = sdata['morphology_focus']
    n_scales = len(list(morph.keys()))
    disp_lvl = min(2, n_scales - 1)
    disp_da  = sd.get_pyramid_levels(morph, n=disp_lvl)
    n_ch     = disp_da.shape[0]
    sdata_img = np.asarray(disp_da[min(1, n_ch - 1)])

    # Scale factors: full-res → display level (dask shape is cheap to read)
    full_res_shape = sd.get_pyramid_levels(morph, n=0).shape[-2:]  # (H, W), lazy
    scale_y = sdata_img.shape[0] / full_res_shape[0]
    scale_x = sdata_img.shape[1] / full_res_shape[1]

    with _MATPLOTLIB_LOCK:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ── Left: landmarked image + original landmarks ───────────────────
        axes[0].imshow(lm_img, cmap='gray')
        if 'xenium_x' in landmarks_before.columns:
            axes[0].scatter(landmarks_before['xenium_x'], landmarks_before['xenium_y'],
                            c='red', s=15, zorder=5)
        axes[0].set_title('Landmarked image  +  original landmarks (image pixel space)')

        # ── Right: sdata image + transformed landmarks ────────────────────
        # Scale landmark coordinates from full-res pixel space to display level.
        axes[1].imshow(sdata_img, cmap='gray')
        axes[1].scatter(landmarks_after['xenium_x'] * scale_x,
                        landmarks_after['xenium_y'] * scale_y,
                        c='red', s=15, zorder=5)
        subtitle = 'sdata morphology_focus (scale0)  +  transformed landmarks'
        if landmarks_tf_info is not None:
            sx   = landmarks_tf_info.get('scale_factor_x')
            sy   = landmarks_tf_info.get('scale_factor_y')
            bbox = landmarks_tf_info.get('bbox')
            if sx is not None and sy is not None:
                subtitle += f'\nscale (x, y): ({sx:.3f}, {sy:.3f})'
            if bbox is not None:
                subtitle += f'  |  bbox offset: ({bbox["x_min"]}, {bbox["y_min"]})'
        axes[1].set_title(subtitle)

        plt.tight_layout()
        if section is not None:
            plt.suptitle(f'Section: {section}', y=1.01)
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
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
       against the landmarked image using Normalized cross-correlation on
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
    best_img_fn, best_lm_fn = _DIHEDRAL_BY_NAME[best_name]

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

def parse_landmarks(landmarks, transform_info, x_col=None, y_col=None):
    """Parse a landmarks DataFrame into a SpatialData PointsModel.

    Parameters
    ----------
    landmarks : pd.DataFrame
        Output of ``find_landmarked_img_transforms``.  Must contain
        ``czstack_x/y/z`` and either ``xenium_x/y`` or ``sdata_x/y`` columns.
    transform_info : dict
        Returned by ``find_landmarked_img_transforms``.  Used for pixel_size
        and matched_level_transforms.
    x_col : str or None
        Column to use as the section x-coordinate (default: ``'sdata_x'`` when
        present, otherwise ``'xenium_x'``).  Use ``'sdata_x'`` when the
        BigWarp TIFF is oriented differently from the SpatialData image (the
        common case when ``landmarked_from_sdata=True`` and the matched dihedral
        transform is not identity).  Use ``'xenium_x'`` only when the TIFF and
        SpatialData images share the same axis orientation (identity transform).
    y_col : str or None
        Column to use as the section y-coordinate (default: ``'sdata_y'`` when
        present, otherwise ``'xenium_y'``).
    """
    landmarks = landmarks.copy()

    # ── 1. Determine which x/y columns to use ────────────────────────────
    # Priority: explicit x_col/y_col > sdata_x/y (when present) > xenium_x/y
    if x_col is None:
        x_col = 'sdata_x' if 'sdata_x' in landmarks.columns else 'xenium_x'
    if y_col is None:
        y_col = 'sdata_y' if 'sdata_y' in landmarks.columns else 'xenium_y'

    # Drop all section-space x/y aliases except the chosen ones, then rename
    # directly to canonical 'x'/'y' names used by PointsModel.
    drop_cols = [c for c in ('xenium_x', 'xenium_y', 'sdata_x', 'sdata_y')
                 if c in landmarks.columns and c not in (x_col, y_col)]
    landmarks = landmarks.drop(columns=drop_cols)
    landmarks = landmarks.rename(columns={x_col: 'x', y_col: 'y', 'xenium_z': 'z'})

    full_scale_pixel_size = transform_info['pixel_size']

    landmarks  = PointsModel.parse(landmarks)
    global_tf  = transform_info['matched_level_transforms']['global']
    set_transformation(landmarks, global_tf, to_coordinate_system='global')

    microns_scale = Scale([full_scale_pixel_size, full_scale_pixel_size], axes=('x', 'y'))
    # Compose: matched-level pixels → full-res pixels (global_tf) → microns.
    # When global_tf is Identity (level-0 / full-res), no upscaling is needed.
    # When both transforms are Scale with the same axes they can be merged into
    # one Scale (element-wise product), avoiding an unnecessary Sequence wrapper.
    if isinstance(global_tf, Identity):
        microns_tf = microns_scale
    elif isinstance(global_tf, Scale) and global_tf.axes == microns_scale.axes:
        microns_tf = Scale(
            [g * m for g, m in zip(global_tf.scale, microns_scale.scale)],
            axes=global_tf.axes,
        )
    else:
        microns_tf = Sequence([global_tf, microns_scale])
    set_transformation(landmarks, microns_tf, to_coordinate_system='microns')
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
        val_folder = alignment_params.get('validation_images_folder')
        save_imgs_path = val_folder / f"section_{s_n}_manual.png" if val_folder is not None else None
        landmarks_out, transform_info = manual_landmarks_transform(
            s_n=s_n,
            sdata_path=sdata_path,
            landmarks_path=landmarks_path,
            landmarked_image_path=landmarked_image_path,
            alignment_params=alignment_params,
            bigwarp_params=bigwarp_params,
            save_imgs_path=save_imgs_path,
        )
        formatted_landmarks = parse_landmarks(landmarks_out, transform_info)

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
            # Use sdata_x/y (landmarks remapped to SpatialData pixel space) when the
            # BigWarp TIFF and SpatialData image may not share the same axis orientation.
            # For manual landmarks (no sdata_ columns) the default falls back to xenium_x/y.
            formatted_landmarks = parse_landmarks(landmarks_out, transform_info,
                                                  x_col='sdata_x', y_col='sdata_y')

    return s_n, formatted_landmarks


def manual_landmarks_transform(s_n, sdata_path, landmarks_path, landmarked_image_path,
                               alignment_params, bigwarp_params,
                               dims_order=None,
                               save_imgs_path=None):
    """Transform manual BigWarp CSV landmarks into SpatialData pixel space.

    Reads a raw BigWarp CSV landmark file, optionally applies coordinate
    corrections (bbox offset via ``fix_cropped_landmarks``, y-axis inversion
    via ``invert_lm_y``), and returns the corrected landmark DataFrame together
    with a ``transform_info`` dict compatible with ``parse_landmarks``.

    Parameters
    ----------
    s_n : int or str
        Section number (used as the bbox key and for diagnostic messages).
    sdata_path : str or Path
        Section ``.zarr`` SpatialData path (loaded once internally).
    landmarks_path : str or Path
        BigWarp-format CSV: columns are
        ``[name, active, czstack_x/y/z, xenium_x/y/z]`` or reversed.
    landmarked_image_path : str or Path
        TIFF used during BigWarp landmarking (needed by
        ``get_landmarked_image_props`` for shape and coordinate corrections).
    alignment_params : dict
        Keys used: ``czstack_buffer``, ``fix_cropped_landmarks``,
        ``invert_lm_y``, ``validation_images_folder``.
    bigwarp_params : dict
        Returned by ``extract_bigwarp_params``; the ``moving_image_dataset``
        key determines CSV column assignment order.
    dims_order : list of str or None
        Axis labels matching the CSV coordinate positions
        (default ``['x', 'y', 'z']``).
    save_imgs_path : Path or str or None
        If provided, a diagnostic plot is saved here.

    Returns
    -------
    landmarks_out : pd.DataFrame
        Corrected landmark coordinates in matched-level pixel space.
    landmarks_tf_info : dict
        ``matched_level_transforms`` (``{'global': Transform}``),
        ``pixel_size``, ``scale_factor_x/y``, ``bbox``.
        Compatible with ``parse_landmarks``.
    """
    if dims_order is None:
        dims_order = ['x', 'y', 'z']
    # ── Read sdata once — needed for pixel size and get_landmarked_image_props ──
    sdata = sd.read_zarr(sdata_path)
    full_scale_pixel_size = sdata['table'].uns['section_metadata']['pixel_size']

    starting_lm_df = pd.read_csv(landmarks_path, header=None)
    if bigwarp_params['moving_image_dataset'] == 'czstack':
        starting_lm_df.columns = (['landmark_name', 'active']
                                   + [f'czstack_{d}' for d in dims_order]
                                   + [f'xenium_{d}'  for d in dims_order])
    else:
        starting_lm_df.columns = (['landmark_name', 'active']
                                   + [f'xenium_{d}'  for d in dims_order]
                                   + [f'czstack_{d}' for d in dims_order])

    landmarks_out    = starting_lm_df.copy()
    landmarks_tf_info = None

    if alignment_params.get('czstack_buffer') is not None:
        landmarks_out = remove_landmark_buffer(
            landmarks_out, czstack_buffer=alignment_params['czstack_buffer'])

    if alignment_params.get('fix_cropped_landmarks', False) or alignment_params.get('invert_lm_y', False):
        landmarks_out, landmarks_tf_info = get_landmarked_image_props(
            landmarked_image_path, sdata, landmarks_out, s_n,
            invert_y=alignment_params.get('invert_lm_y', False))

    if save_imgs_path is not None:
        plot_manual_landmark_transforms(
            landmarks_before=starting_lm_df,
            landmarks_after=landmarks_out,
            landmarked_image_path=landmarked_image_path,
            sdata=sdata,
            landmarks_tf_info=landmarks_tf_info,
            section=s_n,
            save_path=save_imgs_path,
            show=False,
        )

    # When no coordinate corrections were applied, landmarks are already in
    # full-res pixel space (= global coordinate system).
    if landmarks_tf_info is None:
        landmarks_tf_info = {
            'matched_level_transforms': {'global': Identity()},
            'pixel_size': full_scale_pixel_size,
        }
    return landmarks_out, landmarks_tf_info

def get_section_landmarks_threads(xenium_section_ns, paths, alignment_params, n_workers=None):
    if n_workers is None:
        n_workers = min(4, len(xenium_section_ns))
    print(f"Processing {len(xenium_section_ns)} sections with {n_workers} parallel workers …")
    sections_landmarks={}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_process_section, s_n, paths, alignment_params): s_n
            for s_n in xenium_section_ns
        }
        failed = []
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc='Processing sections', unit='section'):
            s_n = futures[fut]
            try:
                s_n_result, lm = fut.result()
                if lm is not None:
                    sections_landmarks[s_n_result] = lm
            except Exception as exc:
                failed.append(s_n)
                print(f"  Section {s_n}: FAILED with {type(exc).__name__}: {exc}")
        if failed:
            print(f"\nWarning: {len(failed)} section(s) failed: {sorted(failed)}")
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
    """Map landmark coordinates from the landmarked TIFF's pixel space into
    the SpatialData section's full-resolution pixel space.

    Two cases are handled:

    **Paired** (multi-section slide, ``sections_bboxes`` present in table.uns):
      The TIFF is a downsampled view of the entire slide.  Coordinates are
      first optionally inverted (in TIFF pixel space), then scaled to the
      full slide, then offset by the section's bounding-box origin.

    **Standalone** (single-section TIFF, no bbox metadata):
      The TIFF is matched to the ``morphology_focus`` pyramid level with the
      same dimensions.  The scale factors are read from that level's stored
      ``global`` transform rather than being computed as a ratio of shapes,
      making them robust to non-integer or asymmetric downsampling.  As a
      fallback (no matching level found), the ratio against ``cell_labels``
      full-res shape is used.  Coordinates are scaled to full-resolution, then
      optionally inverted.

    Both cases produce the same result: inverting before vs after an isotropic
    scale is equivalent (``(H - y) * k  ==  H*k - y*k``).  The ordering
    differs only to keep the inversion in the natural "image" space for each
    case.

    Parameters
    ----------
    landmarked_image_path : str or Path
    sdata : SpatialData or str or Path
        Section SpatialData object (or path to its zarr).
    landmarks : pd.DataFrame
        Must contain ``xenium_x`` and ``xenium_y`` columns in TIFF pixel
        space.  **A copy is returned; the input is not modified.**
    section_n : int or str
    invert_y : bool
        Mirror the y-axis (needed when the TIFF was vertically flipped
        relative to the SpatialData image).

    Returns
    -------
    landmarks : pd.DataFrame
        Copy of input with ``xenium_x``/``xenium_y`` remapped to SpatialData
        full-resolution pixel space.
    landmarks_tf_info : dict
        ``scale_factor_x``, ``scale_factor_y``, ``bbox`` (section bbox dict
        or None), ``fullres_pixel_size``.
    """
    # ── TIFF spatial dimensions ───────────────────────────────────────────
    # Use asarray() so channel dimensions (C, H, W) or (H, W, C) are handled
    # correctly; spatial H and W are always the last two axes.
    with tifffile.TiffFile(landmarked_image_path) as tif:
        arr = tif.asarray()
    lm_H, lm_W = int(arr.shape[-2]), int(arr.shape[-1])
    print(f"Landmarked image shape (H×W): {lm_H}×{lm_W}")

    if isinstance(sdata, (str, Path)):
        sdata = sd.read_zarr(sdata)

    bbox_dict   = sdata['table'].uns.get('sections_bboxes', None)
    section_key = str(section_n)
    is_paired   = bbox_dict is not None and section_key in bbox_dict

    landmarks = landmarks.copy()   # never mutate the caller's DataFrame

    if is_paired:
        # Invert in TIFF pixel space *before* scaling to the full slide
        if invert_y:
            landmarks['xenium_y'] = lm_H - landmarks['xenium_y']

        section_bbox   = bbox_dict[section_key]
        bbox_xmin      = section_bbox['x_min']
        bbox_ymin      = section_bbox['y_min']
        full_slide_H   = np.max([b['y_max'] for b in bbox_dict.values()])
        full_slide_W   = np.max([b['x_max'] for b in bbox_dict.values()])
        scale_factor_y = full_slide_H / lm_H
        scale_factor_x = full_slide_W / lm_W
        # For paired TIFFs landmarks are in the downsampled-TIFF pixel space;
        # the effective scale-to-fullres is the same ratio used above.
        global_tf      = Scale([scale_factor_x, scale_factor_y], axes=('x', 'y'))
        print(f"Paired section — full slide shape: {[full_slide_H, full_slide_W]}")
        print(f"Image downsampled by: (yx) [{scale_factor_y:.4f}, {scale_factor_x:.4f}]")
        print(f"Section bbox: {section_bbox}")
    else:
        section_bbox = None
        bbox_xmin    = 0
        bbox_ymin    = 0

        # ── Match TIFF to a morphology_focus pyramid level ────────────────
        morph = sdata['morphology_focus']
        matched_level = matched_level_idx = None
        for lvl_idx, (lvl, s_l) in enumerate(morph.items()):
            level_shape = tuple(s_l.image.shape[-2:])     
            if set(level_shape) == set((lm_H, lm_W)):    
                matched_level     = lvl
                matched_level_idx = lvl_idx
                break

        if matched_level_idx is not None:
            matched_da = sd.get_pyramid_levels(morph, n=matched_level_idx)
            global_tf  = get_transformation(matched_da, to_coordinate_system='global')
            mat        = global_tf.to_affine_matrix(input_axes=('x', 'y'),
                                                    output_axes=('x', 'y'))
            scale_factor_x = float(mat[0, 0])
            scale_factor_y = float(mat[1, 1])
            print(f"Standalone — matched pyramid level '{matched_level}'  "
                  f"(TIFF {lm_H}×{lm_W})")
            print(f"Scale from stored transform (yx): "
                  f"[{scale_factor_y:.4f}, {scale_factor_x:.4f}]")
        else:
            # Fallback: compute ratio from cell_labels full-res shape
            cell_labels    = sd.get_pyramid_levels(sdata['cell_labels'], n=0)
            sec_H = int(cell_labels.data.shape[-2])
            sec_W = int(cell_labels.data.shape[-1])
            scale_factor_y = sec_H / lm_H
            scale_factor_x = sec_W / lm_W
            global_tf      = Scale([scale_factor_x, scale_factor_y], axes=('x', 'y'))
            print(f"Standalone — no matching morphology pyramid level found; "
                  f"falling back to cell_labels ratio")
            print(f"Image downsampled by: (yx) [{scale_factor_y:.4f}, {scale_factor_x:.4f}]")

    # ── Offset + optional invert ──────────────────────────────────────────
    # The bbox is expressed in full-res pixel space; dividing by scale_factor
    # converts it to matched-level space so it can be applied directly.
    # For standalone sections bbox_{x,y}min = 0, so this is a no-op.
    # Equivalent to: scale → subtract bbox → scale back → same as subtracting bbox/scale.
    landmarks['xenium_x'] -= bbox_xmin / scale_factor_x
    landmarks['xenium_y'] -= bbox_ymin / scale_factor_y

    # Standalone invert: mirror in matched-level space.
    # fullres_H / scale_factor_y = lm_H (by construction), so this simplifies
    # to lm_H - y, identical to the paired branch's pre-scale inversion.
    if not is_paired and invert_y:
        landmarks['xenium_y'] = lm_H - landmarks['xenium_y']

    landmarks_tf_info = {
        'scale_factor_x':     scale_factor_x,
        'scale_factor_y':     scale_factor_y,
        'matched_level_transforms': {'global': global_tf},
        'bbox':               section_bbox,
        'pixel_size': sdata['table'].uns['section_metadata']['pixel_size'],
    }
    return landmarks, landmarks_tf_info

