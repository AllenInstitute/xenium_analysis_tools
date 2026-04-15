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
    get_microns_scales,
    _is_multiscale,
    drop_sdata_elements,
    get_spatial_elements,
    get_single_scale,
    separate_channels,
    rename_coordinate_systems_manual
)

def get_zstack_zarr(alignment_folder, paths, zstack_fov_size=None):
    zstack_folder = alignment_folder / 'zstacks'
    zstack_folder.mkdir(parents=True, exist_ok=True)
    zstacks_zarrs = list(zstack_folder.glob('*.zarr'))

    # If none have been created, generate
    if len(zstacks_zarrs) == 0:
        from xenium_analysis_tools.alignment.zstack_alignment import generate_zstack_sdata
        print("No existing z-stack zarr found. Generating...")
        zstacks = generate_zstack_sdata(paths['zstack_path'], paths['zstack_masks'])
        for sdata_name, sdata in zstacks.items():
            print(f"Writing z-stack zarr: {sdata_name}")
            sdata.write(zstack_folder / f'{sdata_name}.zarr')
            if 'zstacks' in locals():
                del zstacks
        # Re-list the zarrs in the folder now
        zstacks_zarrs = list(zstack_folder.glob('*.zarr'))

    # Load z-stack zarr (if multiple, load the one with the correct FOV size)
    if len(zstacks_zarrs) == 1:
        print(f"Loading existing z-stack zarr: {zstacks_zarrs[0]}")
        zstack_path = zstacks_zarrs[0]
    elif len(zstacks_zarrs) > 1:
        print(f"Multiple z-stacks available: {[z.name for z in zstacks_zarrs]}")
        zstack_path = zstack_folder / f'zstack_{zstack_fov_size}.zarr'
        print(f"Loading z-stack with FOV size {zstack_fov_size} from: {zstack_path}")

    return sd.read_zarr(zstack_path)

def format_czstack(sdata,
                    separate_zstack_chans=True,
                    make_single_scale=True,
                    rp_props=['bbox', 'area', 'extent', 'axis_minor_length', 'axis_major_length']):
    if make_single_scale:
        if _is_multiscale(sdata[list(sdata.images.keys())[0]]):
            sdata = get_single_scale(sdata)
    if separate_zstack_chans:
        if 'zstack' in sdata.images.keys():
            sdata = separate_channels(sdata, 'zstack')
    # Rename global/microns to 'czstack'    
    sdata = rename_coordinate_systems_manual(sdata, {"global": f"czstack", "microns": f"czstack_microns"})

    # Add tables for masks
    for lab in sdata.labels.keys():
        labels_df = pd.DataFrame(regionprops_table(
            np.array(sdata[lab].data), 
            properties=['label', 'centroid'] + rp_props
        ))
        labels_df['region'] = lab
        labels_df.rename(columns={'label': 'cell_labels'}, inplace=True)
        
        adata = ad.AnnData(obs=labels_df.reset_index(drop=True))
        lab_table = TableModel.parse(
            adata,
            region=lab,
            region_key='region',
            instance_key='cell_labels'
        )
        sdata[f"{lab.split('_')[0]}_table"] = lab_table

    return sdata

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

def _extract_2d_affine_at_z0(affine_3d):
    """Reduce a 3D Affine(x,y,z → x,y,z) to a 2D Affine(x,y → x,y) evaluated at z=0.

    A 2D image has no z dimension, so z=0 implicitly.  The z column of the 3D matrix
    contributes nothing (mat[:,2] * 0 = 0), so only the x, y, and translation columns
    matter.  The resulting 3×3 matrix is square and invertible, allowing
    spatialdata's sd.transform() inverse-mapping to work on 2D raster elements.

    The z output row (row 2 of the 3D affine) is dropped because raster resampling of a
    2D image never queries a z output — spatialdata always calls
    transformation.to_affine_matrix(input_axes=element.dims, output_axes=element.dims),
    and a 2D element has dims=(c,y,x) with no z.
    """
    m = np.array(affine_3d.matrix)   # shape (4, 4) for (x,y,z)→(x,y,z)
    mat_2d = np.array([
        [m[0, 0], m[0, 1], m[0, 3]],   # x_out = m00*x + m01*y  (z term = 0)
        [m[1, 0], m[1, 1], m[1, 3]],   # y_out = m10*x + m11*y
        [0.0,     0.0,     1.0     ],
    ])
    return Affine(mat_2d, input_axes=('x', 'y'), output_axes=('x', 'y'))


def add_affine_to_element(element, affine_tf, coord_sys_name,
                          microns_tf=None, microns_tf_position='before',
                          overwrite_existing=False):
    """
    microns_tf_position: 'before' puts microns_tf before affine_tf (section px → µm → czstack px)
                         'after'  puts microns_tf after  affine_tf (section px → czstack px → czstack µm)

    For 2D images (axes c, y, x) the section-to-czstack affine is defined over (x, y, z).
    When spatialdata composes a (c,y,x) scale with an (x,y,z)→(x,y,z) affine it expands
    the axis set to (c,x,y,z) and produces a non-square 5×4 matrix that np.linalg.inv()
    cannot invert, causing a LinAlgError in sd.transform().

    Fix: for 2D elements we extract the XY-only 2D affine from the 3D one (evaluated at
    z=0) and use a matching 2D microns scale.  This keeps every sub-matrix square.
    """
    def _is_2d(el):
        axes = tuple(el.dims) if hasattr(el, 'dims') else ()
        return 'z' not in axes and 'y' in axes and 'x' in axes

    def _reduce_microns_to_2d(mt):
        """Strip z from a microns Scale so it stays in (x,y) space."""
        if mt is None or not hasattr(mt, 'axes'):
            return mt
        xy_scale = [mt.scale[list(mt.axes).index(ax)] for ax in ('x', 'y') if ax in mt.axes]
        if len(xy_scale) == 2:
            return Scale(xy_scale, axes=('x', 'y'))
        return mt

    def _try_combine_scales(s1, s2):
        """Merge two Scale transforms into one when they share the same axes."""
        if isinstance(s1, Scale) and isinstance(s2, Scale) and s1.axes == s2.axes:
            return Scale([a * b for a, b in zip(s1.scale, s2.scale)], axes=s1.axes)
        return None

    def _build_tf(tf_to_fullres, element_obj):
        af = affine_tf
        mt = microns_tf
        # 2D image + 3D affine: reduce both to 2D so all sub-matrices stay square
        if _is_2d(element_obj) and hasattr(af, 'input_axes') and af.input_axes == ('x', 'y', 'z'):
            af = _extract_2d_affine_at_z0(af)
            mt = _reduce_microns_to_2d(mt)
        if mt is not None:
            if microns_tf_position == 'after':
                return Sequence([tf_to_fullres, af, mt])
            else:
                # tf_to_fullres and mt are adjacent Scales — merge when possible
                combined = _try_combine_scales(tf_to_fullres, mt)
                if combined is not None:
                    return Sequence([combined, af])
                return Sequence([tf_to_fullres, mt, af])
        return Sequence([tf_to_fullres, af])

    if _is_multiscale(element):
        for n_l, level in enumerate(element.keys()):
            element_obj = sd.get_pyramid_levels(element, n=n_l)
            tf_to_fullres = get_transformation(element_obj, to_coordinate_system='global')
            if isinstance(tf_to_fullres, Identity):
                tf_to_fullres = Scale([1.0, 1.0, 1.0], axes=('x', 'y', 'z'))
            if coord_sys_name in element_obj.attrs['transform'] and not overwrite_existing:
                continue
            set_transformation(element_obj, _build_tf(tf_to_fullres, element_obj),
                               to_coordinate_system=coord_sys_name)
    else:
        element_obj = element
        tf_to_fullres = get_transformation(element_obj, to_coordinate_system='global')
        if isinstance(tf_to_fullres, Identity):
            tf_to_fullres = Scale([1.0, 1.0, 1.0], axes=('x', 'y', 'z'))
        if coord_sys_name in element_obj.attrs['transform'] and not overwrite_existing:
            return
        set_transformation(element_obj, _build_tf(tf_to_fullres, element_obj),
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

    # ── 2. Normalize column names: xenium_x/y/z → x/y/z ─────────────────
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
    czstack_lm = landmarks[['czstack_x', 'czstack_y', 'czstack_z']].values.astype(float)
    xenium_lm  = landmarks[['x', 'y', 'z']].values.astype(float)

    # ── 4b. Normalize section landmarks to full-res (global) pixel space ──
    try:
        lm_global_tf = get_transformation(landmarks, to_coordinate_system='global')
        if lm_global_tf is not None and not isinstance(lm_global_tf, Identity):
            if isinstance(lm_global_tf, Scale):
                axes = list(lm_global_tf.axes)
                sx = float(lm_global_tf.scale[axes.index('x')]) if 'x' in axes else 1.0
                sy = float(lm_global_tf.scale[axes.index('y')]) if 'y' in axes else 1.0
            else:
                mat = lm_global_tf.to_affine_matrix(input_axes=('x', 'y'),
                                                     output_axes=('x', 'y'))
                sx, sy = float(mat[0, 0]), float(mat[1, 1])
            xenium_lm = xenium_lm.copy()
            xenium_lm[:, 0] *= sx
            xenium_lm[:, 1] *= sy
    except Exception:
        pass   # if transform metadata is absent, use raw coordinates

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

def _rescale_z(tf, mps, z_offset=0.0):
    """Return a new transform with the z-scale replaced by *mps* and z shifted by *z_offset*.

    Works for Scale, Sequence, Identity, or Affine inputs.  Does not mutate the
    input transform.  The z-scale replacement and optional centering translation
    are applied in a single pass — no double-nesting of Sequences.
    """
    def _replace_z_in_scale(t):
        axes = list(t.axes)
        vals = list(t.scale)
        vals[axes.index('z')] = mps
        return Scale(vals, axes=t.axes)

    if isinstance(tf, Identity):
        result = Scale([mps, 1.0, 1.0], axes=('z', 'y', 'x'))
    elif isinstance(tf, Scale):
        if 'z' in tf.axes:
            result = _replace_z_in_scale(tf)
        else:
            result = Sequence([Scale([mps], axes=('z',)), tf])
    elif isinstance(tf, Sequence):
        tfs = list(tf.transformations)
        for i, t in enumerate(tfs):
            if isinstance(t, Scale) and 'z' in t.axes:
                tfs[i] = _replace_z_in_scale(t)
                break
        else:
            # No z-bearing Scale found — prepend one
            tfs = [Scale([mps], axes=('z',))] + tfs
        result = Sequence(tfs)
    else:
        # Affine or other: wrap with z-scale prepended
        result = Sequence([Scale([mps], axes=('z',)), tf])

    # Append z centering as a single translation Affine (one extra step, not a second pass)
    if z_offset != 0.0:
        mat = np.eye(4)
        mat[2, 3] = z_offset  # row 2 = z output, col 3 = translation
        z_center_tf = Affine(mat, input_axes=('x', 'y', 'z'), output_axes=('x', 'y', 'z'))
        if isinstance(result, Sequence):
            result = Sequence(list(result.transformations) + [z_center_tf])
        else:
            result = Sequence([result, z_center_tf])

    return result


def adjust_3d_images_z_scaling(sdata, sections_depth, elements_3d=None, center_z=True):
    """
    Rescale the z-axis of 3D section images to match the known section thickness.

    If center_z=True (default), the slab is further shifted so that z=0 corresponds
    to the section midplane (range: -sections_depth/2 .. +sections_depth/2 µm),
    matching the convention used by adjust_transcripts_z_scaling.

    The z-step is sections_depth / (n_z - 1) so the first and last planes sit exactly
    at ±sections_depth/2 — no empty planes appear in Napari after the transcripts end.

    All coordinate systems already registered on each pyramid level are updated;
    there is no need to pass a coordinate-system list.
    """
    if elements_3d is None:
        elements_3d = ['dapi_zstack']

    for el_name in elements_3d:
        if el_name not in sdata:
            continue
        el = sdata[el_name]

        # Iterate over every pyramid level robustly
        if _is_multiscale(el):
            level_imgs = [sd.get_pyramid_levels(el, n=i) for i in range(len(el.keys()))]
        else:
            level_imgs = [el]

        for img in level_imgs:
            # Robust z-dimension check — never falls back to shape[0] (= c dim)
            if not (hasattr(img, 'dims') and 'z' in img.dims):
                continue
            n_z = img.sizes['z']
            if n_z < 2:
                continue

            mps = sections_depth / (n_z - 1)  # microns per z-step at this level
            z_offset = -(n_z - 1) / 2.0 * mps if center_z else 0.0

            # Update every coord system present — not just a hardcoded subset
            for cs, existing_tf in get_transformation(img, get_all=True).items():
                set_transformation(img, _rescale_z(existing_tf, mps, z_offset),
                                   to_coordinate_system=cs)
    return sdata


def adjust_transcripts_z_scaling(sdata, sections_depth, center_z=True):
    """
    Rescale transcript z-coordinates so they span the known section thickness.

    If center_z=True (default), z=0 is the section midplane
    (range: -sections_depth/2 .. +sections_depth/2 µm), matching
    adjust_3d_images_z_scaling so transcripts and the DAPI z-stack share the
    same z=0 reference point.
    """
    tx = sdata['transcripts']
    if 'original_z_coords' not in tx.columns:
        tx['original_z_coords'] = tx['z']
    z = tx['original_z_coords']

    # Single compute call — avoids triggering the dask graph twice
    z_stats = z.agg(['min', 'max']).compute()
    z_min, z_max = float(z_stats['min']), float(z_stats['max'])
    z_span = z_max - z_min

    if z_span == 0:
        import warnings
        warnings.warn("Transcript z-span is zero; skipping z scaling.")
        return sdata

    scaled_z = (z - z_min) * (sections_depth / z_span)
    if center_z:
        scaled_z = scaled_z - sections_depth / 2.0
    tx['z'] = scaled_z
    return sdata

def _shift_transform_origin_along_z(tf, z_offset):
    mat = np.array(
        tf.to_affine_matrix(input_axes=('x', 'y', 'z'), output_axes=('x', 'y', 'z')),
        dtype=float,
    )
    mat[:, 3] = mat[:, 3] + (mat[:, 2] * float(z_offset))
    return Affine(mat, input_axes=('x', 'y', 'z'), output_axes=('x', 'y', 'z'))


def _get_lifted_element_transforms(reference_element, z_start=0.0):
    reference_transforms = get_transformation(reference_element, get_all=True)
    lifted_transforms = {}
    for coord_sys, tf in reference_transforms.items():
        lifted_transforms[coord_sys] = _shift_transform_origin_along_z(tf, z_start)
    return lifted_transforms


def make_element_3d(
    sdata,
    element_2d,
    reference_3d,
    use_model,
    keep_2d=False,
    lift_mode='centered_plane',
    lift_n_z=None,
):
    """
    Lift a 2D element into 3D with an explicit z-placement convention.

    If keep_2d=True, the original 2D element is preserved under its original name and the
    lifted 3D element is written as "{element_2d}_3d".

    lift_mode:
      - 'reference_slab': repeat across the full z support of `reference_3d`
      - 'centered_plane': place the 2D element on a single plane at the section center
      - 'centered_slab': repeat across `lift_n_z` planes centered within the section
    """
    if lift_mode not in {'reference_slab', 'centered_plane', 'centered_slab'}:
        raise ValueError("lift_mode must be 'reference_slab', 'centered_plane', or 'centered_slab'")

    scale_dict = xr.DataTree()
    for scale_level in sdata[element_2d].keys():
        if 'z' in sdata[element_2d][scale_level].dims:
            return sdata
        shape_3d = sdata[reference_3d][scale_level]['z'].shape[0]

        if lift_mode == 'reference_slab':
            target_n_z = int(shape_3d)
        elif lift_mode == 'centered_plane':
            target_n_z = 1
        else:
            if lift_n_z is None:
                raise ValueError("lift_n_z must be provided when lift_mode='centered_slab'")
            target_n_z = int(lift_n_z)

        if target_n_z < 1 or target_n_z > int(shape_3d):
            raise ValueError(f"target_n_z must be in [1, {shape_3d}], got {target_n_z}")

        z_start = (shape_3d - target_n_z) / 2.0
        transforms_3d = _get_lifted_element_transforms(
            sdata[reference_3d][scale_level].image,
            z_start=z_start,
        )
        
        element = sdata[element_2d][scale_level].image
        el_3d = xr.concat([element] * target_n_z, dim='z')
        
        if 'c' in el_3d.dims:
            el_3d = el_3d.transpose('c', 'z', 'y', 'x')
        else:
            el_3d = el_3d.transpose('z', 'y', 'x')
        
        parsed = use_model.parse(el_3d, c_coords=list(el_3d.coords['c'].values) if 'c' in el_3d.coords else None)
        parsed.attrs['transform'] = transforms_3d
        parsed.attrs['z_lift_mode'] = lift_mode
        parsed.attrs['z_lift_start_index'] = float(z_start)
        parsed.attrs['z_lift_n_planes'] = int(target_n_z)
        parsed.attrs['z_ref_n_planes'] = int(shape_3d)
        scale_dict[scale_level] = xr.Dataset({'image': parsed})
    
    if use_model == Image3DModel:
        if keep_2d:
            sdata.images[f'{element_2d}_3d'] = scale_dict
        else:
            del sdata.images[element_2d]
            sdata.images[element_2d] = scale_dict
    else:
        if keep_2d:
            sdata.labels[f'{element_2d}_3d'] = scale_dict
        else:
            del sdata.labels[element_2d]
            sdata.labels[element_2d] = scale_dict
    
    return sdata



######## Main function for taking processed SpatialData object and aligning to cortical z-stack
def align_section_to_zstack(sdata, 
                            czstack_sdata, 
                            affines, 
                            section_n=None,
                            landmarks=None,
                            sections_um=20.0, 
                            drop_elements=True,
                            make_single_scale=True,
                            add_tx_cell_labels=True,
                            czstack_microns_scale_tf=None,
                            center_z=True,
                            include_flat=False
                            ):
    # Czstack scale to microns
    if czstack_microns_scale_tf is None:
        czstack_microns_scale_tf = get_microns_scales(czstack_sdata, 'zstack')
    xy_scale = czstack_microns_scale_tf.scale[list(czstack_microns_scale_tf.axes).index('x')]
    czstack_xyz_microns_tf = Scale([xy_scale, xy_scale, 1.0], axes=('x', 'y', 'z'))

    # Add landmarks to sdata
    if landmarks is not None:
        sdata.points['landmarks'] = PointsModel.parse(landmarks)

    # Drop unwanted elements
    if drop_elements:
        sdata = drop_sdata_elements(sdata)

    # Add section micron coord system
    sdata = add_micron_coord_sys(sdata)

    # Make dapi-zstack z-scaling fit 20um, centered around z=0
    sdata = adjust_3d_images_z_scaling(sdata, sections_um, center_z=center_z)

    # Make transcript z-scaling fit 20um, centered around z=0
    sdata = adjust_transcripts_z_scaling(sdata, sections_um, center_z=center_z)
        
    # Add transform to czstack/czstack microns if not present
    if 'czstack_microns' not in sdata.coordinate_systems:
        spatial_els = get_spatial_elements(sdata)
        for el in spatial_els:
            add_affine_to_element(sdata[el],
                    affine_tf=affines['section_to_czstack_full_affine'],
                    coord_sys_name='czstack')
            add_affine_to_element(sdata[el],
                affine_tf=affines['section_to_czstack_full_affine'],
                coord_sys_name='czstack_microns',
                microns_tf=czstack_xyz_microns_tf,
                microns_tf_position='after')
            if include_flat:
                add_affine_to_element(sdata[el],
                    affine_tf=affines['section_to_czstack_affine_flat'],
                    coord_sys_name='czstack_flat')
                add_affine_to_element(sdata[el],
                    affine_tf=affines['section_to_czstack_affine_flat'],
                    coord_sys_name='czstack_microns_flat',
                    microns_tf=czstack_xyz_microns_tf,
                    microns_tf_position='after')
    
    # Keep only one scale level instead of multiscale
    if make_single_scale:
        sdata = get_single_scale(sdata)

    # Add section to table and transcripts
    if section_n is not None:
        sdata['table'].obs['section'] = section_n
        sdata['transcripts']['section'] = section_n

    if add_tx_cell_labels:
        cell_id_labels_dict = dict(zip(sdata['table'].obs['cell_id'], sdata['table'].obs['cell_labels']))
        sdata['transcripts']['cell_labels'] = sdata['transcripts']['cell_id'].map(cell_id_labels_dict,  meta=('cell_id', 'float64'))

    # Update region table is annotating
    sdata['table'].obs['region'] = f'cell_labels'
    sdata['table'].obs['region'] = sdata['table'].obs['region'].astype('category')
    sdata['table'].uns['spatialdata_attrs']['instance_key'] = 'cell_labels'
    sdata['table'].obs = sdata['table'].obs.reset_index(drop=True)
    
    sd.SpatialData.update_annotated_regions_metadata(sdata['table'])

    return sdata

def write_sdata_elements(sdata, sdata_path, overwrite=False, num_workers=4):
    """
    Write a SpatialData object element-by-element with a progress bar.
    
    If overwrite=False, skips elements that already exist on disk.
    If overwrite=True, rewrites all elements.
    If writing fails, the partially-written element is deleted to avoid corrupted zarr.
    """
    import shutil

    sdata_path = Path(sdata_path)
    all_elements = list(sdata.gen_elements())  # [(etype, name, element), ...]

    # --- Step 1: create/open zarr store and write root metadata ---
    store = _resolve_zarr_store(sdata_path)
    if sdata_path.exists():
        zarr_group = zarr.open_group(store=store, mode='a')
    else:
        zarr_group = zarr.create_group(store=store, overwrite=True)
    
    sdata.write_attrs(zarr_group=zarr_group)
    store.close()
    sdata.path = sdata_path

    # --- Step 2: determine which elements to write ---
    etype_to_folder = {
        'images': 'images',
        'labels': 'labels', 
        'points': 'points',
        'shapes': 'shapes',
        'tables': 'tables',
    }

    def _element_exists(sdata_path, etype, name):
        folder = etype_to_folder.get(etype, etype)
        return (sdata_path / folder / name).exists()

    def _delete_element(sdata_path, etype, name):
        folder = etype_to_folder.get(etype, etype)
        el_path = sdata_path / folder / name
        if el_path.exists():
            shutil.rmtree(el_path)
            tqdm_nb.write(f"    🗑  Deleted incomplete element at {el_path}")

    to_write = []
    skipped = []
    for etype, name, el in all_elements:
        if not overwrite and _element_exists(sdata_path, etype, name):
            skipped.append((etype, name))
        else:
            to_write.append((etype, name, el))

    if skipped:
        print(f"Skipping {len(skipped)} already-written elements:")
        for etype, name in skipped:
            print(f"  [{etype}] {name} (already exists)")

    if not to_write:
        print("All elements already written. Nothing to do.")
        return

    print(f"Writing {len(to_write)} elements...")

    # --- Step 3: write elements with progress bar ---
    t0 = time.time()
    failed = []
    with dask.config.set(scheduler='threads', num_workers=num_workers):
        for etype, name, _ in tqdm_nb(to_write, desc='Writing elements', unit='element'):
            tqdm_nb.write(f"  [{etype}] {name}...")
            t1 = time.time()
            try:
                sdata.write_element(name, overwrite=overwrite)
                tqdm_nb.write(f"    done in {time.time()-t1:.1f}s")
            except Exception as e:
                tqdm_nb.write(f"    ✗ FAILED: {e}")
                _delete_element(sdata_path, etype, name)
                failed.append((etype, name, str(e)))

        # --- Step 4: consolidate metadata ---
        sdata.write_consolidated_metadata()

    if failed:
        print(f"\n⚠  {len(failed)} element(s) failed to write and were deleted:")
        for etype, name, err in failed:
            print(f"  [{etype}] {name}: {err}")
    
    print(f"Total: {(time.time()-t0)/60:.1f} min")


def compare_affines(derived_affine, calculated_affines, xy_scale=1.75,
                    derived_input_axes='yx', derived_output_axes='yx',
                    verbose=True):
    """
    Compare a previously derived 2D affine (3×3, old BigWarp/tiff pipeline) against
    newly calculated spatialdata Affines.

    Primary question answered:
        "Does applying section_to_czstack_affine_flat to transform a Xenium section into
         czstack coordinates produce equivalent results to the previously found affines?"

    The new pipeline (section_to_czstack_affine_flat) maps:
        Xenium section pixels (sdata_x, sdata_y) → new zarr czstack pixels (x,y)

    Both were derived from the same BigWarp landmarks, but the czstack pixel spaces differ
    (different pixel sizes / FOV origins). An isotropic scale ratio between them is therefore
    expected and does NOT indicate misalignment. Equivalence means:

        1. Same rotation/orientation  (Δangle < 0.5°)
        2. Isotropic scale ratio      (|sx_ratio − sy_ratio| / max(sx, sy) < 2%)
               → the two czstack pixel spaces differ only in pixel pitch, not in alignment
        3. Translation comparison is NOT valid for section→czstack direction, because the
           two czstack pixel spaces have different origins. For czstack→section direction
           (shared section pixel space), translation differences DO reflect real offsets.

    Parameters
    ----------
    derived_affine : np.ndarray (3,3)
        Old 2D affine in homogeneous form.  Its axis conventions are given by
        ``derived_input_axes`` and ``derived_output_axes``.
    calculated_affines : dict
        Output of get_alignment_transforms – spatialdata Affine objects keyed by name.
    xy_scale : float
        Pixel scale factor applied to z-stack coords in the previously found affines (default 1.75).
        Divides out the czstack pixel size so scale ratios are dimensionless.
    derived_input_axes : str, ``'yx'`` (default) or ``'xy'``
        Axis convention of the *input* (section) side of ``derived_affine``.

        ``'yx'`` — input stored as (sdata_y, sdata_x).  A column-swap P is applied on
            the right (``M @ P``) to convert to (sdata_x, sdata_y) before comparison.
            Use when the TIFF and SpatialData image share the same row/column orientation.

        ``'xy'`` — input already in (sdata_x, sdata_y) convention.  No right-swap.
            Use when the TIFF was **transposed** relative to SpatialData (e.g. 797371
            where NCC returns 'transpose' and TIF rows = sdata_x, TIF cols = sdata_y).

    derived_output_axes : str, ``'yx'`` (default) or ``'xy'``
        Axis convention of the *output* (czstack) side of ``derived_affine``.

        ``'yx'`` — output stored as (czstack_y, czstack_x).  A row-swap P is applied on
            the left (``P @ M``) to convert to (czstack_x, czstack_y).

        ``'xy'`` — output already in (czstack_x, czstack_y) convention.  No left-swap.
    verbose : bool
        Print a summary comparison table.

    Returns
    -------
    dict with keys:
        'derived_decomposed'         – scale/angle/translation of the Normalized derived affine
        'derived_lifted_4x4'         – derived affine lifted to 4×4 (x,y,z) convention
        'interpretation'             – human-readable verdict
        'equivalent_to_old_pipeline' – bool: True if rotation matches and scale is isotropic
        'ok_to_proceed'              – same as equivalent_to_old_pipeline
        <affine_key>                 – per-affine comparison metrics
    """
    results = {}

    # ── 1. Normalize derived affine to (sdata_x, sdata_y) → czstack(x,y) ────
    P = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 1]], dtype=float)
    M_xy = np.array(derived_affine, dtype=float)
    if derived_output_axes == 'yx':
        M_xy = P @ M_xy      # swap output rows: (czstack_y, czstack_x) → (czstack_x, czstack_y)
    if derived_input_axes == 'yx':
        M_xy = M_xy @ P      # swap input cols:  (sdata_y, sdata_x) → (sdata_x, sdata_y)
    S_inv  = np.diag([1.0 / xy_scale, 1.0 / xy_scale, 1.0])
    M_norm = S_inv @ M_xy    # section(sdata_x, sdata_y) → old_czstack(x,y)

    M4 = np.eye(4)
    M4[0, 0] = M_norm[0, 0]; M4[0, 1] = M_norm[0, 1]; M4[0, 3] = M_norm[0, 2]
    M4[1, 0] = M_norm[1, 0]; M4[1, 1] = M_norm[1, 1]; M4[1, 3] = M_norm[1, 2]
    results['derived_lifted_4x4'] = M4

    A_d  = M_norm[:2, :2]
    t_d  = M_norm[:2, 2]
    sx_d = np.linalg.norm(A_d[:, 0])
    sy_d = np.linalg.norm(A_d[:, 1])
    theta_d = np.degrees(np.arctan2(A_d[1, 0], A_d[0, 0]))
    # Pure rotation matrix of derived affine (normalize out scale)
    R_d = np.column_stack([A_d[:, 0] / sx_d, A_d[:, 1] / sy_d])
    results['derived_decomposed'] = dict(
        scale_x=sx_d, scale_y=sy_d, angle_deg=theta_d,
        translation_x=t_d[0], translation_y=t_d[1],
        det_sign=float(np.sign(np.linalg.det(A_d)))
    )

    # ── 2. Compare against each calculated affine ────────────────────────────
    for key, calc_affine in calculated_affines.items():
        mat4 = np.array(calc_affine.to_affine_matrix(
            input_axes=('x', 'y', 'z'), output_axes=('x', 'y', 'z')
        ))
        C = np.array([
            [mat4[0, 0], mat4[0, 1], mat4[0, 3]],
            [mat4[1, 0], mat4[1, 1], mat4[1, 3]],
            [0.0,        0.0,        1.0        ],
        ])
        A_c = C[:2, :2]
        t_c = C[:2, 2]
        sx_c = np.linalg.norm(A_c[:, 0])
        sy_c = np.linalg.norm(A_c[:, 1])
        theta_c = np.degrees(np.arctan2(A_c[1, 0], A_c[0, 0]))

        is_same_dir = 'section_to_czstack' in key
        ref = M_norm if is_same_dir else np.linalg.inv(M_norm)

        sx_ref = np.linalg.norm(ref[:2, 0])
        sy_ref = np.linalg.norm(ref[:2, 1])
        sx_ratio = sx_c / sx_ref if sx_ref > 0 else float('nan')
        sy_ratio = sy_c / sy_ref if sy_ref > 0 else float('nan')

        theta_ref = np.degrees(np.arctan2(ref[1, 0], ref[0, 0]))
        angle_diff = abs(theta_ref - theta_c)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        frob   = np.linalg.norm(ref - C)
        t_diff = np.linalg.norm(ref[:2, 2] - t_c)

        # Scale-invariant rotation comparison: normalize columns, then compare.
        # This is the meaningful metric for section→czstack, where the two czstack
        # pixel spaces differ in scale but not in orientation.
        R_c = np.column_stack([A_c[:, 0] / sx_c, A_c[:, 1] / sy_c]) if (sx_c > 0 and sy_c > 0) else A_c
        if is_same_dir:
            rot_only_frob = np.linalg.norm(R_d - R_c)
        else:
            # czstack→section: compare inv(R_d) vs R_c  (R_d.T = inv for pure rotation)
            rot_only_frob = np.linalg.norm(R_d.T - R_c)

        results[key] = dict(
            direction           = 'section→czstack' if is_same_dir else 'czstack→section',
            scale_x             = sx_c,
            scale_y             = sy_c,
            angle_deg           = theta_c,
            det_sign            = float(np.sign(np.linalg.det(A_c))),
            translation_x       = t_c[0],
            translation_y       = t_c[1],
            frobenius_diff      = frob,
            rot_only_frob       = rot_only_frob,   # scale-invariant rotation difference
            translation_diff_px = t_diff,
            angle_diff_deg      = angle_diff,
            scale_x_ratio       = sx_ratio,
            scale_y_ratio       = sy_ratio,
            scale_x_diff        = abs(sx_ref - sx_c),
            scale_y_diff        = abs(sy_ref - sy_c),
        )

    # ── 3. Equivalence verdict focused on section→czstack_affine_flat ────────
    # This directly answers: "will applying section_to_czstack_affine_flat produce
    # the same result as the previously found affines?"
    s2c_key = next(
        (k for k in ['section_to_czstack_affine_flat', 'section_to_czstack_full_affine']
         if k in calculated_affines),
        next((k for k in calculated_affines if 'section_to_czstack' in k), None)
    )
    if s2c_key:
        s2c     = results[s2c_key]
        sx_r, sy_r = s2c['scale_x_ratio'], s2c['scale_y_ratio']
        rot_ok      = s2c['angle_diff_deg'] < 0.5
        rot_exact   = s2c['rot_only_frob'] < 0.01   # pure rotation should be ~0
        scale_uniform = (not np.isnan(sx_r) and not np.isnan(sy_r)
                         and abs(sx_r - sy_r) / max(abs(sx_r), abs(sy_r), 1e-9) < 0.02)
        equivalent = rot_ok and scale_uniform

        if equivalent:
            scale_note = (f'isotropic pixel-size difference between old tiff and new zarr czstack '
                          f'(sx={sx_r:.3f}, sy={sy_r:.3f}) — expected, does not affect alignment')
            verdict = (f'EQUIVALENT ✓ — applying {s2c_key} to transform section→czstack '
                       f'produces the same physical alignment as the previously found affines  |  '
                       f'Δangle={s2c["angle_diff_deg"]:.3f}°  rot_frob={s2c["rot_only_frob"]:.5f}  '
                       f'{scale_note}')
        elif rot_ok and not scale_uniform:
            verdict = (f'PARTIAL — rotation matches (Δangle={s2c["angle_diff_deg"]:.3f}°) but '
                       f'anisotropic scale difference (sx={sx_r:.3f} ≠ sy={sy_r:.3f}) — '
                       f'possible shear between pipelines, review landmarks')
            equivalent = False
        else:
            verdict = (f'MISMATCH ✗ — rotation differs (Δangle={s2c["angle_diff_deg"]:.3f}°, '
                       f'rot_frob={s2c["rot_only_frob"]:.5f}) — '
                       f'transforms describe different physical alignments')
            equivalent = False
    else:
        verdict    = 'No section_to_czstack affine found in calculated_affines'
        equivalent = False

    results['interpretation']             = verdict
    results['equivalent_to_old_pipeline'] = equivalent
    results['ok_to_proceed']              = equivalent

    # ── 4. Print summary ─────────────────────────────────────────────────────
    if verbose:
        def _fmt_transform(sx, sy, theta_deg, det_sign):
            """One-liner describing a 2D linear transform geometrically."""
            iso = abs(sx - sy) / max(abs(sx), abs(sy), 1e-9) < 0.02
            sc  = (f"×{0.5*(sx + sy):.3f}  (isotropic)"
                   if iso else f"sx=×{sx:.3f}, sy=×{sy:.3f}  (anisotropic)")
            fl  = "none" if det_sign > 0 else "yes"
            return f"  rotation = {theta_deg:+.1f}°   scale = {sc}   reflection = {fl}"

        d = results['derived_decomposed']

        if s2c_key and s2c_key in results:
            s2c       = results[s2c_key]
            sx_r, sy_r = s2c['scale_x_ratio'], s2c['scale_y_ratio']
            rot_ok    = s2c['angle_diff_deg'] < 0.5
            ratio_iso = (not np.isnan(sx_r) and not np.isnan(sy_r)
                         and abs(sx_r - sy_r) / max(abs(sx_r), abs(sy_r), 1e-9) < 0.02)
            flip_match = int(s2c['det_sign']) == int(d['det_sign'])

            print(f"  Transform  (section → czstack):")
            print(_fmt_transform(s2c['scale_x'], s2c['scale_y'],
                                 s2c['angle_deg'], s2c['det_sign']))
            print()

            print(f"  vs. previously found affines  (Normalized):")
            rot_sym   = '✓' if rot_ok    else '✗'
            scale_sym = '✓' if ratio_iso else '✗'
            print(f"    rotation = {d['angle_deg']:+.1f}°   "
                  f"Δ = {s2c['angle_diff_deg']:.3f}°  {rot_sym}")
            avg_r      = 0.5 * (sx_r + sy_r)
            ratio_note = "  (czstack pixel-size offset — expected)" if ratio_iso else ""
            print(f"    scale ratio = {avg_r:.3f}×  {scale_sym}{ratio_note}")
            if not flip_match:
                new_fl = 'yes' if s2c['det_sign'] < 0 else 'none'
                old_fl = 'yes' if d['det_sign']   < 0 else 'none'
                print(f"    reflection: new={new_fl}, old={old_fl}  ✗  mismatch")
        else:
            print("  (no section→czstack affine found in calculated_affines)")

        print()
        if equivalent:
            print("  → PASS ✓  same physical alignment as previously found affines")
        elif s2c_key and s2c_key in results and results[s2c_key]['angle_diff_deg'] >= 0.5:
            print("  → FAIL ✗  rotation mismatch — transforms describe different physical alignments")
        else:
            print("  → FAIL ✗  anisotropic scale difference — possible shear, review landmarks")

    return results