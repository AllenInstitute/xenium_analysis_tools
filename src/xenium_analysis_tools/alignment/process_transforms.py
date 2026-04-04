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
from tqdm.notebook import tqdm as tqdm_nb

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

from xenium_analysis_tools.utils.sd_utils import (
    add_micron_coord_sys,
)

####### Functions to find transforms ##########
def get_affine_from_landmarks_flat(moving_coords, ref_coords):
    moving_2d = moving_coords[:, :2]
    ref_2d = ref_coords[:, :2]
    n = moving_2d.shape[0]
    A_xy = np.hstack([moving_2d, np.ones((n, 1))])
    result, _, _, _ = np.linalg.lstsq(A_xy, ref_2d, rcond=None)

    # Z offset: mean difference between ref and moving Z coords
    # mat[2,2] = 1.0 keeps matrix invertible (required by Napari)
    z_offset = float(np.mean(ref_coords[:, 2] - moving_coords[:, 2]))

    mat = np.eye(4)
    mat[0, 0] = result[0, 0]; mat[0, 1] = result[1, 0]; mat[0, 3] = result[2, 0]
    mat[1, 0] = result[0, 1]; mat[1, 1] = result[1, 1]; mat[1, 3] = result[2, 1]
    mat[2, 2] = 1.0   # MUST be 1.0 - keeps matrix invertible for Napari rendering
    mat[2, 3] = z_offset

    return Affine(mat, input_axes=('x', 'y', 'z'), output_axes=('x', 'y', 'z'))

def tilt_affines(moving_pts, fixed_pts, flat_affine):
    # moving_pts = Xenium (2D: x, y, 0)
    # fixed_pts = CZStack (3D: x, y, z)
    
    # Fit Z_stack = a*X_xe + b*Y_xe + c
    A = np.column_stack([moving_pts[:, 0], moving_pts[:, 1], np.ones(len(moving_pts))])
    coeffs, _, _, _ = lstsq(A, fixed_pts[:, 2])
    a, b, c = coeffs
    
    # Build the 3D matrix using the 2D XY results
    tilt_mat = np.array(flat_affine.matrix).copy()
    tilt_mat[2, 0] = a   # Effect of Xenium X on Stack Z
    tilt_mat[2, 1] = b   # Effect of Xenium Y on Stack Z
    tilt_mat[2, 2] = 1.0 # Keeps matrix invertible
    tilt_mat[2, 3] = c   # The base Z-slice offset
    
    return Affine(tilt_mat, input_axes=('x', 'y', 'z'), output_axes=('x', 'y', 'z'))

def add_affine_to_element(element, affine_tf, coord_sys_name, 
                          microns_tf=None, microns_tf_position='before',
                          overwrite_existing=False):
    """
    microns_tf_position: 'before' puts microns_tf before affine_tf (section px → µm → czstack px)
                         'after'  puts microns_tf after  affine_tf (section px → czstack px → czstack µm)
    """
    def _build_tf(tf_to_fullres):
        if microns_tf is not None:
            if microns_tf_position == 'after':
                return Sequence([tf_to_fullres, affine_tf, microns_tf])
            else:
                return Sequence([tf_to_fullres, microns_tf, affine_tf])
        return Sequence([tf_to_fullres, affine_tf])

    if _is_multiscale(element):
        for n_l, level in enumerate(element.keys()):
            element_obj = sd.get_pyramid_levels(element, n=n_l)
            tf_to_fullres = get_transformation(element_obj, to_coordinate_system='global')
            if isinstance(tf_to_fullres, Identity):
                tf_to_fullres = Scale([1.0, 1.0, 1.0], axes=('x', 'y', 'z'))
            if coord_sys_name in element_obj.attrs['transform'] and not overwrite_existing:
                continue
            set_transformation(element_obj, _build_tf(tf_to_fullres), 
                               to_coordinate_system=coord_sys_name)
    else:
        element_obj = element
        tf_to_fullres = get_transformation(element_obj, to_coordinate_system='global')
        if isinstance(tf_to_fullres, Identity):
            tf_to_fullres = Scale([1.0, 1.0, 1.0], axes=('x', 'y', 'z'))
        if coord_sys_name in element_obj.attrs['transform'] and not overwrite_existing:
            return
        set_transformation(element_obj, _build_tf(tf_to_fullres),
                           to_coordinate_system=coord_sys_name)

def get_alignment_transforms(landmarks):
    """
    Compute czstack ↔ section affine transforms from a landmarks DataFrame.

    Parameters
    ----------
    landmarks : pd.DataFrame or dask DataFrame
        Must contain columns czstack_x/y/z and x/y/z (or xenium_x/y/z).

    Returns
    -------
    section_affines : dict
        Four affine matrices: flat and full (tilt-aware) in both directions.
    """
    # ── 1. Materialise dask DataFrames ────────────────────────────────────
    try:
        import dask.dataframe as dd
        if isinstance(landmarks, dd.DataFrame):
            landmarks = landmarks.compute()
    except ImportError:
        pass

    if landmarks is None or (hasattr(landmarks, 'empty') and landmarks.empty):
        raise ValueError("landmarks is None or empty — cannot compute affines")

    # ── 2. Normalise column names: xenium_x/y/z → x/y/z ─────────────────
    rename_map = {}
    if 'xenium_x' in landmarks.columns and 'x' not in landmarks.columns:
        rename_map['xenium_x'] = 'x'
    if 'xenium_y' in landmarks.columns and 'y' not in landmarks.columns:
        rename_map['xenium_y'] = 'y'
    if 'xenium_z' in landmarks.columns and 'z' not in landmarks.columns:
        rename_map['xenium_z'] = 'z'
    if rename_map:
        landmarks = landmarks.rename(columns=rename_map)

    # ── 3. Validate required columns are present ──────────────────────────
    required = {'czstack_x', 'czstack_y', 'czstack_z', 'x', 'y', 'z'}
    missing  = required - set(landmarks.columns)
    if missing:
        raise ValueError(f"landmarks is missing required columns: {missing}")

    # ── 4. Extract coordinate arrays ──────────────────────────────────────
    czstack_lm = landmarks[['czstack_x', 'czstack_y', 'czstack_z']].values
    xenium_lm  = landmarks[['x', 'y', 'z']].values

    # ── 5. Compute flat affines once, reuse for tilt affines ──────────────
    czstack_to_section_flat = get_affine_from_landmarks_flat(czstack_lm, xenium_lm)
    section_to_czstack_flat = get_affine_from_landmarks_flat(xenium_lm, czstack_lm)

    section_affines = {
        'czstack_to_section_affine_flat': czstack_to_section_flat,
        'section_to_czstack_affine_flat': section_to_czstack_flat,
        'czstack_to_section_full_affine': tilt_affines(czstack_lm, xenium_lm,
                                                        czstack_to_section_flat),
        'section_to_czstack_full_affine': tilt_affines(xenium_lm, czstack_lm,
                                                        section_to_czstack_flat),
    }
    return section_affines                        