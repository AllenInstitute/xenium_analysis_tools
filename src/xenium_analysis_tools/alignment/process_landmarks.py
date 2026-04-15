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
                **{f'czstack_{ax}': [pt[i] for pt in czstack_pts] for i, ax in enumerate(axes_order)},
                **{f'xenium_{ax}':  [pt[i] for pt in xenium_pts]  for i, ax in enumerate(axes_order)},
            })
            lm_df = lm_df.loc[lm_df['active']].copy()

    # Clean up paths strings
    moving_image_path_raw = (
        moving_image_path_raw
        .replace('file:', '')
        .replace('/?', '')
    )

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


# ── Image loading / reshape helpers ──────────────────────────────────────────

def _load_landmarked_image(landmarked_image_path):
    """Read a TIFF landmark image and squeeze a leading singleton channel dim."""
    landmarked_image = tifffile.imread(landmarked_image_path)
    if landmarked_image.ndim == 3 and landmarked_image.shape[0] == 1:
        landmarked_image = landmarked_image[0]
    return landmarked_image


def _to_2d(img, channel=0):
    """Return a 2-D view of *img*, picking *channel* for channel-first/last arrays."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[0] <= 4:                           # channel-first (C, H, W)
            return arr[min(channel, arr.shape[0] - 1)]
        if arr.shape[-1] <= 4:                          # channel-last  (H, W, C)
            return arr[..., min(channel, arr.shape[-1] - 1)]
    return np.squeeze(arr)


def _crop_to_section_bbox(image, section_bbox, scale_factor_x, scale_factor_y):
    """Crop *image* to the region corresponding to *section_bbox* in global coords."""
    y0 = max(0, int(np.floor(section_bbox['y_min'] / scale_factor_y)))
    y1 = min(image.shape[-2], int(np.ceil(section_bbox['y_max'] / scale_factor_y)))
    x0 = max(0, int(np.floor(section_bbox['x_min'] / scale_factor_x)))
    x1 = min(image.shape[-1], int(np.ceil(section_bbox['x_max'] / scale_factor_x)))
    return image[..., y0:y1, x0:x1]


def _level_hw(level_obj):
    """Return (H, W) for a multiscale level DataArray or plain array."""
    if hasattr(level_obj, 'image') and hasattr(level_obj.image, 'shape'):
        return tuple(level_obj.image.shape[-2:])
    return tuple(level_obj.shape[-2:])


def _get_level_global_xy_scale(morph, level_idx):
    """Return (sx, sy, all_tfs) for pyramid level *level_idx* of *morph*."""
    level_da = sd.get_pyramid_levels(morph, n=level_idx)
    all_tfs   = get_transformation(level_da, get_all=True)
    tf_global = get_transformation(level_da, to_coordinate_system='global')

    sx = sy = 1.0
    if isinstance(tf_global, Scale):
        axes   = list(tf_global.axes)
        scales = list(tf_global.scale)
        sx = float(scales[axes.index('x')]) if 'x' in axes else float(scales[0])
        sy = float(scales[axes.index('y')]) if 'y' in axes else float(scales[min(1, len(scales) - 1)])
    else:
        try:
            mat = tf_global.to_affine_matrix(input_axes=('x', 'y'), output_axes=('x', 'y'))
            sx = float(abs(mat[0, 0]))
            sy = float(abs(mat[1, 1]))
        except Exception:
            pass
    return sx, sy, all_tfs


def _match_level_by_global_scale(morph, target_sx, target_sy):
    """Find the pyramid level whose global XY scale is closest to *(target_sx, target_sy)*."""
    best = None
    for level_idx, level_name in enumerate(morph.keys()):
        sx, sy, all_tfs = _get_level_global_xy_scale(morph, level_idx)
        ex  = abs(sx - target_sx) / max(abs(target_sx), 1e-8)
        ey  = abs(sy - target_sy) / max(abs(target_sy), 1e-8)
        err = ex + ey
        if best is None or err < best['scale_error']:
            best = {
                'matched_level':            level_name,
                'matched_level_idx':        level_idx,
                'matched_level_transforms': all_tfs,
                'matched_scale_x':          sx,
                'matched_scale_y':          sy,
                'scale_error':              err,
            }
    return best


def _fit_image_to_shape(image, target_hw):
    """Trim or edge-pad *image* so its last two axes match *(H, W) = target_hw*."""
    th, tw = map(int, target_hw)
    ih, iw = map(int, image.shape[-2:])

    out = image[..., :min(ih, th), :min(iw, tw)]
    oh, ow = out.shape[-2:]

    if oh < th or ow < tw:
        pad_y = max(0, th - oh)
        pad_x = max(0, tw - ow)
        out = np.pad(out, ((0, pad_y), (0, pad_x)), mode='edge')
    return out


def load_landmarks_from_csv(landmarks_path, bigwarp_params, dims_order=None):
    """Read a headerless BigWarp CSV landmark file and assign canonical column names.

    Parameters
    ----------
    landmarks_path : str or Path
    bigwarp_params : dict
        Output of :func:`extract_bigwarp_params`.  The ``moving_image_dataset``
        key controls which coordinate columns are labelled ``czstack_*`` vs
        ``xenium_*``.
    dims_order : list of str, optional
        Axis order used in the CSV (default: ``['x', 'y', 'z']``).
    """
    if dims_order is None:
        dims_order = ['x', 'y', 'z']
    landmarks = pd.read_csv(landmarks_path, header=None)
    if bigwarp_params['moving_image_dataset'] == 'czstack':
        landmarks.columns = (
            ['landmark_name', 'active']
            + [f'czstack_{d}' for d in dims_order]
            + [f'xenium_{d}'  for d in dims_order]
        )
    else:
        landmarks.columns = (
            ['landmark_name', 'active']
            + [f'xenium_{d}'  for d in dims_order]
            + [f'czstack_{d}' for d in dims_order]
        )
    return landmarks


def find_manual_landmarked_img_transforms(
    landmarked_image_path,
    sdata,
    landmarks,
    section_n,
    invert_y=False,
    verbose=True,
    force_match_by_scale=True,
):
    """Pre-process a landmark image that was annotated on a paired (cropped/scaled) image.

    Handles two cases:

    * **No section bbox** — only optional y-axis inversion is applied; a
      direct shape-based level match is attempted.
    * **Paired section bbox present** — the scale factors between the full
      slide and the landmarked TIFF are computed, the image is cropped to the
      section region, landmarks are shifted accordingly, and the closest
      pyramid level is chosen by global-scale proximity.

    Returns
    -------
    manual_info : dict
        Metadata about the preprocessing (scale factors, matched level, …).
    corrected_landmarks : pd.DataFrame
        Landmarks with corrected ``xenium_x/y`` coordinates.
    landmarked_image : np.ndarray
        Pre-processed image ready to pass to
        :func:`find_landmarked_img_transforms`.
    """
    if isinstance(sdata, (str, Path)):
        sdata = sd.read_zarr(sdata)

    morph = sdata['morphology_focus']
    landmarked_image    = _load_landmarked_image(landmarked_image_path)
    corrected_landmarks = landmarks.copy()

    lm_h, lm_w = map(int, landmarked_image.shape[-2:])
    if verbose:
        print(f"Landmarked image shape (H×W): {lm_h}×{lm_w}")

    if invert_y:
        corrected_landmarks['xenium_y'] = lm_h - corrected_landmarks['xenium_y']
        landmarked_image = np.flip(landmarked_image, axis=-2)

    bbox_dict    = sdata['table'].uns.get('sections_bboxes') or {}
    section_bbox = bbox_dict.get(str(section_n))

    manual_info = {
        'bbox':                     section_bbox,
        'invert_y':                 invert_y,
        'scale_factor_x':           1.0,
        'scale_factor_y':           1.0,
        'matched_level':            None,
        'matched_level_idx':        None,
        'matched_level_transforms': {'global': Identity()},
        'pixel_size':               sdata['table'].uns['section_metadata']['pixel_size'],
    }

    # ── No paired bbox: try direct shape match ────────────────────────────
    if section_bbox is None:
        for level_idx, (level_name, level_data) in enumerate(morph.items()):
            level_shape = _level_hw(level_data)
            if set(level_shape) == set((lm_h, lm_w)):
                manual_info['matched_level']            = level_name
                manual_info['matched_level_idx']        = level_idx
                manual_info['matched_level_transforms'] = get_transformation(
                    sd.get_pyramid_levels(morph, n=level_idx), get_all=True)
                break
        if verbose:
            print("No paired section bbox found; only optional y-inversion was applied.")
        return manual_info, corrected_landmarks, landmarked_image

    # ── Paired section: compute scale factors and crop ────────────────────
    full_slide_h  = max(bbox['y_max'] for bbox in bbox_dict.values())
    full_slide_w  = max(bbox['x_max'] for bbox in bbox_dict.values())
    scale_factor_y = full_slide_h / lm_h
    scale_factor_x = full_slide_w / lm_w

    corrected_landmarks['xenium_x'] -= section_bbox['x_min'] / scale_factor_x
    corrected_landmarks['xenium_y'] -= section_bbox['y_min'] / scale_factor_y
    landmarked_image = _crop_to_section_bbox(
        landmarked_image, section_bbox,
        scale_factor_x=scale_factor_x, scale_factor_y=scale_factor_y,
    )

    # ── Match pyramid level by global scale ───────────────────────────────
    best = None
    if force_match_by_scale:
        best = _match_level_by_global_scale(morph, scale_factor_x, scale_factor_y)
        if best is not None:
            target_shape     = _level_hw(sd.get_pyramid_levels(morph, n=best['matched_level_idx']))
            landmarked_image = _fit_image_to_shape(landmarked_image, target_shape)

            manual_info['matched_level']            = best['matched_level']
            manual_info['matched_level_idx']        = best['matched_level_idx']
            manual_info['matched_level_transforms'] = best['matched_level_transforms']
            manual_info['matched_scale_x']          = best['matched_scale_x']
            manual_info['matched_scale_y']          = best['matched_scale_y']
            manual_info['scale_error']              = best['scale_error']
            manual_info['matched_level_shape']      = target_shape

    manual_info.update({
        'bbox':                   section_bbox,
        'scale_factor_x':         scale_factor_x,
        'scale_factor_y':         scale_factor_y,
        'paired_scale_to_global': {'x': scale_factor_x, 'y': scale_factor_y},
    })

    if verbose:
        print(f"Paired section — full slide shape: {[full_slide_h, full_slide_w]}")
        print(f"Image downsampled by: (yx) [{scale_factor_y:.4f}, {scale_factor_x:.4f}]")
        if best is not None:
            print(
                f"Matched pyramid level by closest global scale: {best['matched_level']} "
                f"(sx={best['matched_scale_x']:.4f}, sy={best['matched_scale_y']:.4f}, "
                f"err={best['scale_error']:.6f})"
            )
        print(f"Section bbox: {section_bbox}")

    return manual_info, corrected_landmarks, landmarked_image


def plot_landmark_transforms_manual_vs_sdata(
    sdata,
    original_landmarked_image,
    original_landmarks,
    corrected_image,
    corrected_landmarks,
    manual_info,
    section_n=None,
    save_path=None,
    show=False,
    ch=1,
):
    """Three-panel QC plot for manual landmark preprocessing.

    Left  — original landmarked image + original landmarks.
    Middle — manually corrected image + corrected landmarks.
    Right  — matched SpatialData pyramid level + corrected landmarks.

    Parameters
    ----------
    show : bool
        If ``True`` call ``plt.show()``; otherwise save and close.
        Set to ``False`` when calling from background threads.
    """
    with _MATPLOTLIB_LOCK:
        _plot_landmark_transforms_manual_vs_sdata_inner(
            sdata, original_landmarked_image, original_landmarks,
            corrected_image, corrected_landmarks, manual_info,
            section_n, save_path, show, ch,
        )


def _plot_landmark_transforms_manual_vs_sdata_inner(
    sdata, original_landmarked_image, original_landmarks,
    corrected_image, corrected_landmarks, manual_info,
    section_n, save_path, show, ch,
):
    if isinstance(sdata, (str, Path)):
        sdata = sd.read_zarr(sdata)

    morph = sdata['morphology_focus']
    lvl_idx = manual_info.get('matched_level_idx')
    disp    = sd.get_pyramid_levels(morph, n=lvl_idx if lvl_idx is not None else 0)
    sdata_img = np.asarray(disp[ch] if getattr(disp, 'ndim', 2) == 3 else disp)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(_to_2d(original_landmarked_image, channel=0), cmap='gray')
    axes[0].scatter(original_landmarks['xenium_x'], original_landmarks['xenium_y'], c='r', s=12)
    axes[0].set_title('Original landmarked image + landmarks')

    axes[1].imshow(_to_2d(corrected_image, channel=0), cmap='gray')
    axes[1].scatter(corrected_landmarks['xenium_x'], corrected_landmarks['xenium_y'], c='r', s=12)
    axes[1].set_title('Manual-corrected image + landmarks')

    axes[2].imshow(_to_2d(sdata_img, channel=0), cmap='gray')
    axes[2].scatter(corrected_landmarks['xenium_x'], corrected_landmarks['xenium_y'], c='r', s=12)
    axes[2].set_title(f"sdata level: {manual_info.get('matched_level', 'unknown')}")

    for ax in axes:
        ax.axis('off')

    if section_n is not None:
        fig.suptitle(f"Section {section_n}")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


def resolve_landmarked_img_transforms(
    landmarked_image_path,
    sdata,
    landmarks,
    section_n,
    alignment_params,
    save_imgs_path=None,
    verbose=True,
    plot_manual=True,
):
    """Try automatic transform finding; fall back to manual preprocessing if needed.

    First attempts :func:`find_landmarked_img_transforms` directly.  If that
    raises (e.g. no pyramid level matches the raw TIFF shape) *and*
    ``alignment_params`` contains ``fix_cropped_landmarks=True`` or
    ``invert_lm_y=True``, the manual preprocessing pipeline
    (:func:`find_manual_landmarked_img_transforms`) is run first, a QC plot is
    saved, and then the automatic matcher is retried on the corrected image.

    Returns
    -------
    transform_info : dict
    landmarks_out  : pd.DataFrame  (with ``sdata_x/y`` columns added)
    landmarked_image : np.ndarray
    """
    # ── 1. Try automatic path first ───────────────────────────────────────
    try:
        transform_info, landmarks_out = find_landmarked_img_transforms(
            landmarked_img=landmarked_image_path,
            sdata=sdata,
            landmarks=landmarks,
            plot_imgs=False,
            save_imgs_path=save_imgs_path,
        )
        return transform_info, landmarks_out, _load_landmarked_image(landmarked_image_path)

    except Exception as exc:
        needs_manual_prep = (
            alignment_params.get('fix_cropped_landmarks', False)
            or alignment_params.get('invert_lm_y', False)
        )
        if not needs_manual_prep:
            raise
        # Python 3 deletes the 'exc' name after the except block; capture it now.
        _auto_exc = exc

    # ── 2. Manual preprocessing ───────────────────────────────────────────
    if verbose:
        print(f"Automatic landmark/image matching failed: {type(_auto_exc).__name__}: {_auto_exc}")
        print("Using manual preprocessing, then retrying automatic matching...")

    original_landmarked_image = _load_landmarked_image(landmarked_image_path)
    original_landmarks        = landmarks.copy()

    manual_info, corrected_landmarks, corrected_image = find_manual_landmarked_img_transforms(
        landmarked_image_path=landmarked_image_path,
        sdata=sdata,
        landmarks=landmarks,
        section_n=section_n,
        invert_y=alignment_params.get('invert_lm_y', False),
        verbose=verbose,
        force_match_by_scale=True,
    )

    if plot_manual and save_imgs_path is not None:
        plot_landmark_transforms_manual_vs_sdata(
            sdata=sdata,
            original_landmarked_image=original_landmarked_image,
            original_landmarks=original_landmarks,
            corrected_image=corrected_image,
            corrected_landmarks=corrected_landmarks,
            manual_info=manual_info,
            section_n=section_n,
            save_path=save_imgs_path,
            show=False,
        )

    # ── 3. Retry auto matching on pre-processed image ─────────────────────
    transform_info, landmarks_out = find_landmarked_img_transforms(
        landmarked_img=corrected_image,
        sdata=sdata,
        landmarks=corrected_landmarks,
        plot_imgs=False,
        save_imgs_path=None,
    )
    transform_info['manual_prep'] = manual_info
    return transform_info, landmarks_out, corrected_image


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


def find_landmarked_img_transforms(landmarked_img, 
                                    sdata, 
                                    landmarks,
                                    lm_img_ch_n=1,
                                    thumbnail_size=128,
                                    plot_imgs=True, 
                                    save_imgs_path=None):
    # ── 1. Load landmarked TIFF ───────────────────────────────────────────
    if isinstance(landmarked_img, Path) or isinstance(landmarked_img, str):
        with tifffile.TiffFile(landmarked_img) as tif:
            lm_stack = tif.asarray()
    elif isinstance(landmarked_img, np.ndarray):
        lm_stack = landmarked_img
    lm_img = lm_stack if lm_stack.ndim == 2 else lm_stack[lm_img_ch_n]

    # ── 2. Find matching pyramid level ────────────────────────────────────
    if isinstance(sdata, Path) or isinstance(sdata, str):
        sdata = sd.read_zarr(sdata)
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

    Strategy
    --------
    1. Extract landmarks from the BigWarp JSON.  If none are stored there,
       fall back to a standalone CSV specified by ``alignment_params['landmarks_folder']``.
    2. Apply any buffer correction (``czstack_buffer``).
    3. Call :func:`resolve_landmarked_img_transforms`, which tries the
       automatic dihedral-transform matcher first and, when that fails due to a
       shape mismatch, runs the manual-preprocessing pipeline
       (:func:`find_manual_landmarked_img_transforms`) before retrying.
    4. Parse the output landmarks into a SpatialData PointsModel.
    """
    bigwarp_folder_path = (paths['data_root']
                           / alignment_params['bigwarp_projects_folder']
                           / alignment_params['bigwarp_projects_names_fn'](s_n))
    landmarked_image_path = (paths['data_root']
                              / alignment_params['landmarked_images_folder']
                              / alignment_params['landmarked_images_names_fn'](s_n))
    sdata_path = paths['sdata_path'] / f"section_{s_n}.zarr"

    val_folder     = alignment_params.get('validation_images_folder')
    save_imgs_path = val_folder / f"section_{s_n}.png" if val_folder is not None else None

    # ── Early-exit: check required files before any heavy I/O ─────────────
    if not bigwarp_folder_path.exists():
        print(f"  Section {s_n}: BigWarp project not found at {bigwarp_folder_path}, skipping")
        return s_n, None
    if not landmarked_image_path.exists():
        print(f"  Section {s_n}: landmarked image not found at {landmarked_image_path}, skipping")
        return s_n, None

    # ── 1. Load landmarks ─────────────────────────────────────────────────
    bigwarp_params, lm_df = extract_bigwarp_params(bigwarp_folder_path)

    if lm_df is None or lm_df.empty:
        print(f"  Section {s_n}: no landmarks in BigWarp project, looking for CSV fallback...")
        if alignment_params.get('landmarks_folder') is None:
            print(f"  Section {s_n}: no landmarks_folder configured, skipping")
            return s_n, None
        landmarks_path = (paths['data_root']
                          / alignment_params['landmarks_folder']
                          / alignment_params['landmarks_folder_names_fn'](s_n))
        if not landmarks_path.exists():
            print(f"  Section {s_n}: landmarks CSV not found at {landmarks_path}, skipping")
            return s_n, None
        lm_df = load_landmarks_from_csv(landmarks_path, bigwarp_params)

    # ── 2. Apply z-stack buffer correction ───────────────────────────────
    if alignment_params.get('czstack_buffer') is not None:
        lm_df = remove_landmark_buffer(lm_df, czstack_buffer=alignment_params['czstack_buffer'])

    # ── 3. Resolve transforms (auto → manual fallback) ────────────────────
    print(f"  Section {s_n}: resolving landmark transforms…")
    try:
        transform_info, landmarks_out, _ = resolve_landmarked_img_transforms(
            landmarked_image_path=landmarked_image_path,
            sdata=sdata_path,
            landmarks=lm_df,
            section_n=s_n,
            alignment_params=alignment_params,
            save_imgs_path=save_imgs_path,
            verbose=False,       # worker threads: keep stdout clean
            plot_manual=True,    # QC image saved to save_imgs_path
        )
    except Exception as exc:
        print(f"  Section {s_n}: transform resolution FAILED — {type(exc).__name__}: {exc}")
        return s_n, None

    if landmarks_out is None:
        return s_n, None

    # ── 4. Parse into SpatialData PointsModel ─────────────────────────────
    # Use sdata_x/y when they were produced (automatic path); xenium_x/y otherwise.
    x_col = 'sdata_x' if 'sdata_x' in landmarks_out.columns else 'xenium_x'
    y_col = 'sdata_y' if 'sdata_y' in landmarks_out.columns else 'xenium_y'
    formatted_landmarks = parse_landmarks(landmarks_out, transform_info,
                                          x_col=x_col, y_col=y_col)
    return s_n, formatted_landmarks


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

# def _to_3x3(mat):
#     """Extract the 2D xy affine (3×3) from either a 3×3 or 4×4 matrix.

#     Handles two axis conventions:
#       - 3×3: already a 2D homogeneous affine [[a,b,tx],[c,d,ty],[0,0,1]]
#       - 4×4: extract rows/cols for x,y + homogeneous  →  indices [0,1,3]
#     """
#     mat = np.asarray(mat, dtype=float)
#     if mat.shape == (3, 3):
#         return mat
#     if mat.shape == (4, 4):
#         idx = [0, 1, 3]   # x-row, y-row, homogeneous row; same for cols
#         return mat[np.ix_(idx, idx)]
#     raise ValueError(f"Expected 3×3 or 4×4 matrix, got {mat.shape}")

# def _swap_xy_3x3(m):
#     """Swap the x and y axes of a 3×3 2D affine matrix.

#     Equivalent to pre- and post-multiplying by the permutation [[0,1,0],[1,0,0],[0,0,1]].
#     This converts between (row=x, col=y) and (row=y, col=x) conventions.
#     """
#     P = np.array([[0, 1, 0],
#                   [1, 0, 0],
#                   [0, 0, 1]], dtype=float)
#     return P @ m @ P

# def _decompose_2d_affine(m):
#     """Decompose a 3×3 2D affine into rotation (deg), x/y scale, shear, translation."""
#     a, b, tx = m[0, 0], m[0, 1], m[0, 2]
#     c, d, ty = m[1, 0], m[1, 1], m[1, 2]
#     sx   = np.sqrt(a**2 + c**2)
#     sy   = np.sqrt(b**2 + d**2)
#     angle = np.degrees(np.arctan2(c, a))
#     return {'angle_deg': angle, 'scale_x': sx, 'scale_y': sy,
#             'tx': tx, 'ty': ty}

# def _get_matrix(affine_obj):
#     """Return a plain numpy array from a spatialdata Affine object or ndarray."""
#     if hasattr(affine_obj, 'matrix'):
#         return np.asarray(affine_obj.matrix)
#     return np.asarray(affine_obj)

# def compare_affines(derived_affine, calculated_affines, section_n,
#                     match_keys=None, print_matrices=True):
#     """
#     Compare a pre-existing (derived) 2D affine against all entries in a
#     calculated affines dict, reporting residuals and geometric decomposition.

#     The function checks both the direct comparison and the xy-swapped version
#     of each calculated affine, since BigWarp / spatialdata can use different
#     (row, col) ↔ (x, y) conventions.

#     Parameters
#     ----------
#     derived_affine : array-like, shape (3, 3)
#         Reference affine in 2D homogeneous form.
#     calculated_affines : dict
#         Keys → spatialdata Affine or ndarray (3×3 or 4×4).
#     match_keys : list of str, optional
#         Subset of keys to compare.  Defaults to all keys.
#     print_matrices : bool
#         Print full matrix for the best-matching key.
#     """
#     derived = _to_3x3(np.asarray(derived_affine, dtype=float))
#     keys    = match_keys or list(calculated_affines.keys())

#     print("=" * 70)
#     print(f"SECTION: {section_n}")
#     print("Affine comparison  (Frobenius norm of residual matrix)")
#     print("=" * 70)

#     best_key   = None
#     best_norm  = np.inf
#     best_swapped = False
#     results    = {}

#     for key in keys:
#         mat  = _to_3x3(_get_matrix(calculated_affines[key]))
#         norm_direct  = np.linalg.norm(derived - mat)
#         norm_swapped = np.linalg.norm(derived - _swap_xy_3x3(mat))
#         use_swap     = norm_swapped < norm_direct
#         best_norm_k  = min(norm_direct, norm_swapped)
#         results[key] = {'norm': best_norm_k, 'xy_swapped': use_swap,
#                         'norm_direct': norm_direct, 'norm_swapped': norm_swapped}

#         flag = " ← xy-swapped" if use_swap else ""
#         print(f"  {key:<42s}  residual = {best_norm_k:8.4f}{flag}")

#         if best_norm_k < best_norm:
#             best_norm    = best_norm_k
#             best_key     = key
#             best_swapped = use_swap

#     print()
#     print(f"Best match : '{best_key}'  (residual norm = {best_norm:.4f}"
#           + (", xy-axes swapped)" if best_swapped else ")"))

#     # ── Geometric decomposition ───────────────────────────────────────────
#     best_mat  = _to_3x3(_get_matrix(calculated_affines[best_key]))
#     if best_swapped:
#         best_mat = _swap_xy_3x3(best_mat)

#     d_dec = _decompose_2d_affine(derived)
#     c_dec = _decompose_2d_affine(best_mat)

#     print()
#     print(f"{'Parameter':<14} {'Derived':>14} {'Calculated':>14} {'Δ':>12}")
#     print("-" * 58)
#     for param in ['angle_deg', 'scale_x', 'scale_y', 'tx', 'ty']:
#         dv, cv = d_dec[param], c_dec[param]
#         print(f"  {param:<12} {dv:>14.5f} {cv:>14.5f} {dv - cv:>12.5f}")

#     if print_matrices:
#         print()
#         print("Derived affine (3×3):")
#         print(np.array2string(derived, precision=6, suppress_small=True))
#         print(f"\nCalculated '{best_key}' (3×3{', xy-swapped' if best_swapped else ''}):")
#         print(np.array2string(best_mat, precision=6, suppress_small=True))
#         print()
#         print("Residual matrix (derived − calculated):")
#         print(np.array2string(derived - best_mat, precision=6, suppress_small=True))

#     print("=" * 70)
#     return results