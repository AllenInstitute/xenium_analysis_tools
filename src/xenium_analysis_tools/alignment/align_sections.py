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
        print(f"Multiple z-stacks available: {[z for z in zstacks_zarrs]}")
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

def get_element_bytes(el):
    try:
        if hasattr(el, 'data') and hasattr(el.data, 'nbytes'):
            return el.data.nbytes
        elif hasattr(el, 'nbytes') and not callable(el.nbytes):
            return el.nbytes
        elif hasattr(el, 'compute'):  # Dask DataFrame (points)
            return el.compute().memory_usage(deep=True).sum()
        elif hasattr(el, '__sizeof__'):  # AnnData
            return el.__sizeof__()
    except Exception:
        pass
    return 0

def print_sdata_size_summary(sdata):
    # --- Size summary ---
    print("\n=== Combined SpatialData size summary ===")
    total_bytes = 0
    for element_type, container in [('images', sdata.images),
                                     ('labels', sdata.labels),
                                     ('points', sdata.points),
                                     ('tables', sdata.tables)]:
        # Group by prefix (e.g. 'dapi_zstack', 'boundary', 'cell_labels')
        groups = {}
        for name, el in container.items():
            el_bytes = get_element_bytes(el)
            # Extract prefix: 'dapi_zstack-3' → 'dapi_zstack', 'gcamp' → 'gcamp'
            prefix = name.rsplit('-', 1)[0] if '-' in name and name.rsplit('-', 1)[-1].isdigit() else name
            if prefix not in groups:
                groups[prefix] = {'bytes': 0, 'count': 0}
            groups[prefix]['bytes'] += el_bytes
            groups[prefix]['count'] += 1
            total_bytes += el_bytes

        print(f"\n  [{element_type}]")
        for prefix, info in groups.items():
            n = info['count']
            gb = info['bytes'] / 1e9
            label = f"({n} sections)" if n > 1 else ""
            print(f"    {prefix} {label}: {gb:.2f} GB")

    print(f"\n  Total (uncompressed, in-memory): {total_bytes / 1e9:.2f} GB")
    print(f"  On-disk (zarr, ~3-5x compression): ~{total_bytes / 1e9 / 4:.2f}–{total_bytes / 1e9 / 3:.2f} GB estimated")
    print("=========================================\n")

####### Functions for manipulating spatialdata objects #########
def get_spatial_elements(sdata):
    spatial_elements = []
    spatial_elements.extend(sdata.images.keys())
    spatial_elements.extend(sdata.labels.keys())
    spatial_elements.extend(sdata.points.keys())
    spatial_elements.extend(sdata.shapes.keys())
    return spatial_elements


def rename_coordinate_systems_manual(sdata, rename_dict):
    from geopandas import GeoDataFrame

    def _rename_tfs(tfs):
        return {rename_dict.get(k, k): v for k, v in tfs.items()}

    def _is_multiscale_element(el):
        keys_attr = getattr(el, "keys", None)
        if not callable(keys_attr):
            return False
        try:
            ks = list(el.keys())
            if len(ks) == 0:
                return False
            # multiscale nodes usually have .image at each scale
            first = el[ks[0]]
            return hasattr(first, "image")
        except Exception:
            return False

    for store in [sdata.images, sdata.labels, sdata.points, sdata.shapes]:
        for el_name in list(store.keys()):
            el = store[el_name]
            try:
                if _is_multiscale_element(el):
                    for scale in el.keys():
                        node = el[scale]
                        img = node.image if hasattr(node, "image") else node
                        img.attrs["transform"] = _rename_tfs(
                            dict(img.attrs.get("transform", {}))
                        )
                else:
                    # points/shapes/geodataframe/single-scale elements
                    if hasattr(el, "attrs"):
                        el.attrs["transform"] = _rename_tfs(
                            dict(el.attrs.get("transform", {}))
                        )
            except Exception as e:
                print(f"  Warning: could not rename transforms for {el_name}: {e}")

    return sdata

def rename_elements_section(sdata, section_n, rename_tables=True):
    for el in list(sdata.images.keys()):
        section_el = sdata[el]
        del sdata[el]
        sdata.images[f"{el}_{section_n}"] = section_el

    for el in list(sdata.labels.keys()):
        section_el = sdata[el]
        del sdata[el]
        sdata.labels[f"{el}_{section_n}"] = section_el

    for el in list(sdata.points.keys()):
        section_el = sdata[el]
        del sdata[el]
        sdata.points[f"{el}_{section_n}"] = section_el
    
    for el in list(sdata.shapes.keys()):
        section_el = sdata[el]
        del sdata[el]
        sdata.shapes[f"{el}_{section_n}"] = section_el
    if rename_tables:
        for el in list(sdata.tables.keys()):
            section_el = sdata[el]
            del sdata[el]
            sdata.tables[f"{el}_{section_n}"] = section_el
    return sdata

def adjust_3d_images_z_scaling(sdata, sections_depth, elements_3d=['dapi_zstack'], center_z=True):
    """
    Rescale the z-axis of 3D section images to match the known section thickness.

    If center_z=True (default), the slab is further shifted so that z=0 in global/microns
    space corresponds to the section midplane.  This is the physically correct convention
    because the 2D landmark affine was fit to a projection of the whole section volume,
    so z=0 should represent the center of that volume rather than its bottom edge.

    The z-step uses ``sections_depth / (z_planes - 1)`` so that the first and last planes
    sit exactly at ±sections_depth/2, matching the transcript z-range produced by
    ``adjust_transcripts_z_scaling``.  This prevents empty planes from appearing in napari
    after the transcripts have ended.
    """
    for el in elements_3d:
        if el not in sdata:
            continue
            
        for scale in sdata[el].keys():
            img = sdata[el][scale].image if hasattr(sdata[el][scale], 'image') else sdata[el][scale]
            z_planes = img.sizes.get('z', img.shape[0])
            if z_planes < 2:
                continue  # single-plane: nothing meaningful to rescale
            microns_per_slice = sections_depth / (z_planes - 1)

            def update_z_scale(tf):
                if isinstance(tf, Identity):
                    return Scale([microns_per_slice, 1.0, 1.0], axes=('z', 'y', 'x'))
                elif isinstance(tf, Scale):
                    if 'z' in tf.axes:
                        new_scale = list(tf.scale)
                        new_scale[list(tf.axes).index('z')] = microns_per_slice
                        return Scale(new_scale, axes=tf.axes)
                    return Sequence([Scale([microns_per_slice], axes=('z',)), tf])
                elif isinstance(tf, Sequence):
                    new_tfs = [
                        Scale(
                            [microns_per_slice if ax == 'z' else s for ax, s in zip(t.axes, t.scale)],
                            axes=t.axes
                        ) if isinstance(t, Scale) and 'z' in t.axes else t
                        for t in tf.transformations
                    ]
                    if not any(isinstance(t, Scale) and 'z' in t.axes for t in new_tfs):
                        new_tfs = [Scale([microns_per_slice, 1.0, 1.0], axes=('z', 'y', 'x'))] + new_tfs
                    return Sequence(new_tfs)
                return Sequence([Scale([microns_per_slice], axes=('z',)), tf])

            for cs in ['global', 'microns']:
                existing_tf = get_transformation(img, to_coordinate_system=cs)
                if existing_tf is not None:
                    new_tf = update_z_scale(existing_tf)
                    set_transformation(img, new_tf, to_coordinate_system=cs)

            # Center the slab: shift z so the midplane (z_image=(z_planes-1)/2) maps to z=0.
            # With mps = sections_depth/(z_planes-1), plane 0 → -sections_depth/2 and
            # plane (z_planes-1) → +sections_depth/2, exactly matching transcript z range.
            if center_z:
                z_center_offset = -(z_planes - 1) / 2.0 * microns_per_slice
                center_mat = np.eye(4)
                center_mat[2, 3] = z_center_offset
                center_tf_3d = Affine(center_mat, input_axes=('x', 'y', 'z'), output_axes=('x', 'y', 'z'))
                for cs in ['global', 'microns']:
                    existing_tf = get_transformation(img, to_coordinate_system=cs)
                    if existing_tf is not None:
                        set_transformation(img, Sequence([existing_tf, center_tf_3d]), to_coordinate_system=cs)
    return sdata

def adjust_transcripts_z_scaling(sdata, sections_depth, center_z=True):
    """
    Rescale transcript z-coordinates so they span the known section thickness.

    If center_z=True (default), z-coordinates are further shifted so that z=0 corresponds
    to the section midplane (range: -sections_depth/2 .. +sections_depth/2 µm).  This
    matches the centered convention used by adjust_3d_images_z_scaling so that transcripts
    and the DAPI z-stack share the same z=0 reference point.
    """
    if 'original_z_coords' not in sdata['transcripts'].columns:
        z_coords = sdata['transcripts']['z']
        sdata['transcripts']['original_z_coords'] = z_coords
    else:
        z_coords = sdata['transcripts']['original_z_coords']
    
    z_min, z_max = z_coords.min().compute(), z_coords.max().compute()
    tx_z_span = z_max - z_min
    
    if tx_z_span == 0:
        print(f"  Warning: transcript z-span is zero, skipping z scaling.")
        return sdata
    
    microns_thickness_scale = sections_depth / tx_z_span
    scaled_z = (z_coords - z_min) * microns_thickness_scale
    # Center around z=0 so the middle of the transcript volume coincides with
    # the 2D section plane (z=0 in section space = the landmark-fitted position).
    if center_z:
        scaled_z = scaled_z - sections_depth / 2.0
    sdata['transcripts']['z'] = scaled_z
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

def get_single_scale(sdata, keep_scale=2, zstack_scale=0):
    single_scale_sdata = sd.SpatialData()
    for el_name in sdata.images.keys():
        if el_name in ['zstack', 'gcamp', 'dextran']:
            single_scale_sdata.images[el_name] = sd.get_pyramid_levels(sdata[el_name], n=zstack_scale)
        else:
            single_scale_sdata.images[el_name] = sd.get_pyramid_levels(sdata[el_name], n=keep_scale)
    for el_name in sdata.labels.keys():
        if el_name in ['zstack_label', 'gcamp_labels', 'dextran_labels']:
            single_scale_sdata.labels[el_name] = sd.get_pyramid_levels(sdata[el_name], n=zstack_scale)
        else:
            single_scale_sdata.labels[el_name] = sd.get_pyramid_levels(sdata[el_name], n=keep_scale)
    for el_name in sdata.points.keys():
        single_scale_sdata.points[el_name] = sdata.points[el_name]
    for el_name in sdata.tables.keys():
        single_scale_sdata.tables[el_name] = sdata.tables[el_name]
    for el_name in sdata.shapes.keys():
        single_scale_sdata.shapes[el_name] = sdata.shapes[el_name]
    return single_scale_sdata

def drop_sdata_elements(sdata, drop_elements=['nucleus_labels', 'cell_boundaries', 'cell_circles', 'nucleus_boundaries']):
    for el_name in drop_elements:
        if el_name in sdata.labels:
            del sdata.labels[el_name]
        if el_name in sdata.images:
            del sdata.images[el_name]
        if el_name in sdata.shapes:
            del sdata.shapes[el_name]
        if el_name in sdata.points:
            del sdata.points[el_name]
    return sdata

def separate_channels(sdata, element='morphology_focus', section_n=None, drop_source=True):
    channel_name_map = {
        'DAPI': 'dapi',
        'ATP1A1/CD45/E-Cadherin': 'boundary',
        '18S': 'rna',
        'AlphaSMA/Vimentin': 'protein'
    }

    el = sdata.images[element]

    # Get channel names from scale0
    if hasattr(el, 'keys'):
        scale_levels = list(el.keys())
    else:
        scale_levels = None

    if scale_levels:
        c_coords = el[scale_levels[0]].image.coords['c'].values
    else:
        c_coords = el.coords['c'].values

    for ch in c_coords:
        ch_name = channel_name_map.get(str(ch), str(ch))

        if scale_levels:
            scale_dict = {}

            for scale_level in scale_levels:
                img = el[str(scale_level)].image          # (c, z, y, x) or (c, y, x)
                el_tf = get_transformation(img, get_all=True)

                chan_img = img.sel(c=ch)                  # (z, y, x) or (y, x)
                chan_img = chan_img.expand_dims('c', axis=0)
                chan_img = chan_img.assign_coords(c=[ch_name])

                # Determine dims based on actual shape
                if chan_img.ndim == 4:
                    parse_dims = ('c', 'z', 'y', 'x')
                else:
                    parse_dims = ('c', 'y', 'x')
                    use_model = Image3DModel if 'z' in img.dims else Image3DModel

                parsed = Image3DModel.parse(
                    chan_img.values,
                    dims=parse_dims,
                    c_coords=[ch_name],
                    scale_factors=None,
                    transformations=el_tf
                ) if chan_img.ndim == 4 else __import__(
                    'spatialdata.models', fromlist=['Image2DModel']
                ).Image2DModel.parse(
                    chan_img.values,
                    dims=parse_dims,
                    c_coords=[ch_name],
                    scale_factors=None,
                    transformations=el_tf
                )
                scale_dict[str(scale_level)] = parsed

            new_dt = xr.DataTree.from_dict({
                scale: xr.Dataset({'image': arr})
                for scale, arr in scale_dict.items()
            })
        else:
            el_tf = get_transformation(el, get_all=True)
            chan_img = el.sel(c=ch)
            if 'c' not in chan_img.dims:
                chan_img = chan_img.expand_dims('c', axis=0)
            chan_img = chan_img.assign_coords(c=[ch_name])

            # Determine if this is 3D (c, z, y, x) or 2D (c, y, x)
            if chan_img.ndim == 4:
                parse_dims = ('c', 'z', 'y', 'x')
                new_dt = Image3DModel.parse(
                    chan_img.values,
                    dims=parse_dims,
                    c_coords=[ch_name],
                    transformations=el_tf
                )
            else:
                from spatialdata.models import Image2DModel
                parse_dims = ('c', 'y', 'x')
                new_dt = Image2DModel.parse(
                    chan_img.values,
                    dims=parse_dims,
                    c_coords=[ch_name],
                    transformations=el_tf
                )

        if ch_name in sdata.images:
            del sdata.images[ch_name]
        if section_n is not None:
            ch_name = f"{ch_name}_{section_n}"
        sdata.images[ch_name] = new_dt

    if drop_source and element in sdata.images:
        del sdata.images[element]
    return sdata

####### Functions to handle landmarks #########
def get_biwarp_params(bigwarp_json_path):
    import json
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

def get_section_landmarks(landmarks_path, dims_order=['x','y','z'], bigwarp_project_path=None, moving_img=None):
    if bigwarp_project_path is not None:
        bigwarp_params = get_biwarp_params(bigwarp_project_path)
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

    return landmarks, bigwarp_params

def get_landmarked_image_props(landmarked_image_path, sdata, landmarks, 
                                section_n, invert_y=False):
    with tifffile.TiffFile(landmarked_image_path) as tif:
        landmarked_image_shape = tif.pages[0].shape
    print(f"Landmarked image shape: {landmarked_image_shape}")

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

    return landmarks

def format_landmarks(sdata, 
                    landmarks_path, 
                    section_n, 
                    bigwarp_project_paths=None,
                    moving_img=None,
                    czstack_buffer=None,
                    invert_lm_y=False, 
                    fix_cropped_landmarks=False,
                    landmarked_image_path=None,
                    dims_order=['x','y','z']):

    landmarks, bigwarp_params = get_section_landmarks(
                                    landmarks_path=landmarks_path, 
                                    bigwarp_project_path=bigwarp_project_paths,
                                    moving_img=moving_img,
                                    dims_order=dims_order)

    if czstack_buffer is not None:
        landmarks = remove_landmark_buffer(landmarks, 
                                            czstack_buffer=czstack_buffer)

    if fix_cropped_landmarks and landmarked_image_path is not None:
        # invert_y is handled inside get_landmarked_image_props at the right point
        landmarks = get_landmarked_image_props(
                            landmarked_image_path, 
                            sdata, 
                            landmarks, 
                            section_n,
                            invert_y=invert_lm_y)
    elif invert_lm_y and landmarked_image_path is not None:
        # sections where fix_cropped_landmarks=False
        landmarks = invert_xenium_y_landmarks(landmarks, landmarked_image_path)

    # Adjust landmarks resolutions
    full_scale_pixel_size = sdata['table'].uns['section_metadata']['pixel_size']
    landmarks = landmarks.rename(columns={'xenium_x': 'x', 'xenium_y': 'y', 'xenium_z': 'z'})
    landmarks = PointsModel.parse(landmarks)
    set_transformation(landmarks, Identity(), to_coordinate_system='global')
    set_transformation(
        landmarks,
        Scale([full_scale_pixel_size, full_scale_pixel_size], axes=('x', 'y')),
        to_coordinate_system='microns'
    )
    return landmarks

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

def _is_multiscale(element):
    return (
        hasattr(element, 'keys')
        and callable(element.keys)
        and not isinstance(element, GeoDataFrame)
    )

def get_microns_scales(sdata, element_name):
    el = sdata[element_name]
    if _is_multiscale(el):
        img = sd.get_pyramid_levels(el, n=0)
        img = img.image if hasattr(img, 'image') else img
    else:
        img = el.image if hasattr(el, 'image') else el

    # Get transforms from the actual image DataArray, not the DataTree
    el_transforms = get_transformation(img, get_all=True)
    microns_tf = el_transforms.get('microns', None)
    if microns_tf is None:
        ps = sdata['table'].uns['section_metadata']['pixel_size']
        microns_tf = Scale([ps, ps], axes=('x', 'y'))
        set_transformation(img, microns_tf, to_coordinate_system='microns')
    if len(microns_tf.scale) >= 2:
        x_y_axes = ('x', 'y')
        x_y_tf = [microns_tf.axes.index(axis) for axis in x_y_axes if axis in microns_tf.axes]
        microns_tf = Scale([microns_tf.scale[i] for i in x_y_tf], x_y_axes)
    return microns_tf

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

####### QC alignment #########
def get_section_z_stats(affines_dict, landmarks_dict, czstack_shape_yx=(512, 512)):
    y_max, x_max = czstack_shape_yx
    
    # FOV corners in czstack pixels
    fov_corners_czstack = np.array([
        [0,     0,     0, 1],
        [x_max, 0,     0, 1],
        [0,     y_max, 0, 1],
        [x_max, y_max, 0, 1],
    ])  # columns: x, y, z, 1

    results = {}
    for s_n, affines in affines_dict.items():
        tilt = affines['section_to_czstack_full_affine']
        if hasattr(landmarks_dict[s_n], 'compute'):
            landmarks = landmarks_dict[s_n].compute()
        else:
            landmarks = landmarks_dict[s_n]
        czstack_pts = landmarks[['czstack_x', 'czstack_y', 'czstack_z']].values
        if isinstance(tilt, dict):
            tilt = Affine(
                matrix=np.array(tilt['matrix']),
                input_axes=tuple(tilt['input_axes']),
                output_axes=tuple(tilt['output_axes'])
            )
        mat = tilt.matrix  # input_axes=(x,y,z)

        # Invert to go czstack -> section pixels, then forward to get z at FOV corners
        inv_mat = np.linalg.inv(mat)
        
        # Map czstack FOV corners -> section pixels
        section_corners = (inv_mat @ fov_corners_czstack.T).T  # (4, 4)
        
        # Map those section pixels -> czstack z via forward affine
        fwd = np.zeros((4, 4))
        fwd[:, 3] = 1
        xi = list(tilt.input_axes).index('x')
        yi = list(tilt.input_axes).index('y')
        zi = list(tilt.input_axes).index('z')
        for i in range(4):
            fwd[i, xi] = section_corners[i, xi]
            fwd[i, yi] = section_corners[i, yi]
            fwd[i, zi] = 0
        z_out = (mat @ fwd.T).T[:, list(tilt.output_axes).index('z')]
        
        # Landmark-derived z centroid
        z_centroid_from_landmarks = float(np.mean(czstack_pts[:, 2]))
        
        # Z at the center of the FOV
        center = np.array([[x_max/2, y_max/2, 0, 1]])
        center_section = (inv_mat @ center.T).T
        center_fwd = np.zeros((1, 4)); center_fwd[0, 3] = 1
        center_fwd[0, xi] = center_section[0, xi]
        center_fwd[0, yi] = center_section[0, yi]
        center_fwd[0, zi] = 0
        z_at_center = float((mat @ center_fwd.T).T[0, list(tilt.output_axes).index('z')])
        
        results[s_n] = {
            'z_centroid_landmarks': z_centroid_from_landmarks,
            'z_at_fov_center':      z_at_center,
            'z_min_in_fov':         float(z_out.min()),
            'z_max_in_fov':         float(z_out.max()),
            'z_span_in_fov':        float(z_out.max() - z_out.min()),
            'tilt_deg':             float(np.degrees(np.arctan(
                                        np.sqrt(mat[2,0]**2 + mat[2,1]**2) / abs(mat[2,2])
                                    ))),
        }
    return results

def print_z_stats_and_check_overlaps(results, 
                                      czstack_pixel_um=0.78125,
                                      czstack_z_um_per_plane=1.0,
                                      sections_depth_um=20.0,
                                      z_units='planes'):
    """
    Parameters
    ----------
    results : dict from get_section_z_stats
        z values are in czstack planes (z_um_per_plane = 1.0 for this dataset)
    czstack_pixel_um : float
        xy pixel size in µm (default 0.78125)
    czstack_z_um_per_plane : float
        z step size in µm per plane (default 1.0)
    sections_depth_um : float
        expected physical section thickness in µm
    z_units : str
        'planes' or 'microns' — controls display units for z values.
        For this dataset they are equivalent (1 plane = 1 µm) but
        labeling them correctly avoids ambiguity.
    """
    assert z_units in ('planes', 'microns'), "z_units must be 'planes' or 'microns'"
    zu = 'µm' if z_units == 'microns' else 'pl'

    # Convert planes → µm if requested
    scale = czstack_z_um_per_plane if z_units == 'microns' else 1.0

    print(f"{'Sec':>4} {'z_lm_centroid':>16} {'z_fov_center':>14} "
          f"{'z_min':>9} {'z_max':>9} {'z_span':>9} {'tilt°':>6}")
    print(f"{'':>4} {'('+zu+')':>16} {'('+zu+')':>14} "
          f"{'('+zu+')':>9} {'('+zu+')':>9} {'('+zu+')':>9}")
    print("-" * 68)
    for s_n, r in sorted(results.items()):
        print(f"{s_n:>4} "
              f"{r['z_centroid_landmarks']*scale:>16.1f} "
              f"{r['z_at_fov_center']*scale:>14.1f} "
              f"{r['z_min_in_fov']*scale:>9.1f} "
              f"{r['z_max_in_fov']*scale:>9.1f} "
              f"{r['z_span_in_fov']*scale:>9.1f} "
              f"{r['tilt_deg']:>6.2f}°")

    print(f"\nOverlap check (consecutive sections):")
    print(f"Note: z values are in czstack {z_units} "
          f"({czstack_z_um_per_plane} µm/plane, "
          f"xy pixel = {czstack_pixel_um} µm)")

    sorted_sections = sorted(results.keys())
    for i in range(len(sorted_sections) - 1):
        s_a = sorted_sections[i]
        s_b = sorted_sections[i + 1]

        # Separation in planes (raw from affine), then convert to µm
        sep_planes = abs(results[s_b]['z_at_fov_center'] - 
                         results[s_a]['z_at_fov_center'])
        sep_um     = sep_planes * czstack_z_um_per_plane
        min_sep_um = sections_depth_um

        # Overlap test in planes (raw units, no conversion needed)
        fov_overlap = results[s_a]['z_max_in_fov'] > results[s_b]['z_min_in_fov']

        print(f"  Sections {s_a}→{s_b}: "
              f"separation = {sep_planes:.1f} pl "
              f"= {sep_um:.1f} µm  "
              f"(min expected {min_sep_um:.1f} µm)  "
              f"{'⚠  FOV ranges overlap' if fov_overlap else '✓  non-overlapping'}")

        results[s_a]['overlap_with_next']       = fov_overlap
        results[s_a]['separation_planes_to_next'] = sep_planes
        results[s_a]['separation_um_to_next']     = sep_um

    results[sorted_sections[-1]]['overlap_with_next']         = False
    results[sorted_sections[-1]]['separation_planes_to_next'] = None
    results[sorted_sections[-1]]['separation_um_to_next']     = None

    return results

def find_landmark_outliers(landmarks):
      if isinstance(landmarks, dd.DataFrame):
            lm = landmarks.compute()
      else:
            lm = landmarks
      
      cz_z = lm['czstack_z'].values

      # Identify outliers: landmarks more than 2 std from mean
      mean_z, std_z = cz_z.mean(), cz_z.std()
      outlier_mask = np.abs(cz_z - mean_z) > 2 * std_z
      print(f"Outlier landmarks ({outlier_mask.sum()} found):")
      print(lm[outlier_mask][['czstack_x', 'czstack_y', 'czstack_z', 'x', 'y']])

      # Compare affine with and without outliers
      xenium_pts_all = lm[['x', 'y', 'z']].values
      czstack_pts_all = lm[['czstack_x', 'czstack_y', 'czstack_z']].values

      xenium_pts_clean = xenium_pts_all[~outlier_mask]
      czstack_pts_clean = czstack_pts_all[~outlier_mask]

      flat_all   = get_affine_from_landmarks_flat(xenium_pts_all, czstack_pts_all)
      flat_clean = get_affine_from_landmarks_flat(xenium_pts_clean, czstack_pts_clean)
      tilt_all   = tilt_affines(xenium_pts_all, czstack_pts_all, flat_all)
      tilt_clean = tilt_affines(xenium_pts_clean, czstack_pts_clean, flat_clean)

      print(f"\nWith all landmarks:     z_offset = {flat_all.matrix[2,3]:.3f}, "
            f"tilt = {np.degrees(np.arctan(np.sqrt(tilt_all.matrix[2,0]**2 + tilt_all.matrix[2,1]**2))):.3f}°")
      print(f"Without outliers:       z_offset = {flat_clean.matrix[2,3]:.3f}, "
            f"tilt = {np.degrees(np.arctan(np.sqrt(tilt_clean.matrix[2,0]**2 + tilt_clean.matrix[2,1]**2))):.3f}°")

      print(f"\nz_center separation from section 1:")
      print(f"  With all landmarks:  {flat_all.matrix[2,3] - 67.4:.1f} planes")
      print(f"  Without outliers:    {flat_clean.matrix[2,3] - 67.4:.1f} planes")


def plot_section_positions(results,
                           czstack_xy_um=0.78125,
                           czstack_depth_um=450,
                           czstack_x_shape=512,
                           pairs=None,
                           save_path=None):
    sections = sorted(results.keys())

    if pairs is None:
        pairs = []
    paired = {s for p in pairs for s in p}
    
    # ── Color assignment ──────────────────────────────────────────────────
    pair_hues = [
        ('#4e9af1', '#1a5fa8'),
        ('#3db87a', '#1a6e46'),
        ('#e05c5c', '#8a1f1f'),
        ('#b07fd4', '#6a3a9a'),
        ('#50b8c4', '#1e6e7a'),
        ('#f4a742', '#9a5e10'),
    ]
    # Bright palette for standalones (or all sections when no pairs given)
    bright_palette = [
        '#4e9af1', '#3db87a', '#e05c5c', '#b07fd4',
        '#50b8c4', '#f4a742', '#f17c4e', '#a8d44e',
        '#f14eb0', '#4ef1d4', '#d4c44e', '#7a4ef1',
        '#f1d44e', '#4ef17a', '#f14e7a', '#4eaff1',
    ]

    colors = {}
    if pairs:
        pair_idx = 0
        for p in pairs:
            if pair_idx < len(pair_hues):
                colors[p[0]] = pair_hues[pair_idx][0]
                colors[p[1]] = pair_hues[pair_idx][1]
                pair_idx += 1
        # Standalones get remaining bright colors (skip hues already used by pairs)
        bright_idx = pair_idx * 2  # rough offset past used pair colors
        for s_n in sections:
            if s_n not in colors:
                colors[s_n] = bright_palette[bright_idx % len(bright_palette)]
                bright_idx += 1
    else:
        # No pairs — assign bright colors directly, evenly spaced
        for i, s_n in enumerate(sections):
            colors[s_n] = bright_palette[i % len(bright_palette)]

    # ── Pair membership lookup ────────────────────────────────────────────
    pair_of = {}  # section → (s_a, s_b) tuple it belongs to
    for p in pairs:
        pair_of[p[0]] = p
        pair_of[p[1]] = p

    fig = plt.figure(figsize=(15, 6.5), facecolor='#0f1117')
    gs = GridSpec(1, 2, figure=fig, wspace=0.44)

    # ── Left: side view (XZ) ─────────────────────────────────────────────
    ax_xz = fig.add_subplot(gs[0])
    ax_xz.set_facecolor('#0f1117')
    x_fov = czstack_xy_um * czstack_x_shape

    ax_xz.set_xlim(-5, x_fov + 90)
    ax_xz.set_ylim(czstack_depth_um + 10, -10)
    ax_xz.set_xlabel('x position in czstack (µm)', color='#8b9ab0', fontsize=9)
    ax_xz.set_ylabel('z depth (µm)', color='#8b9ab0', fontsize=9)
    ax_xz.set_title('Side view: section positions and tilt',
                    color='#e2e8f0', fontsize=10, pad=8)
    for spine in ax_xz.spines.values():
        spine.set_edgecolor('#2d3748')
    ax_xz.tick_params(colors='#8b9ab0', labelsize=8)
    ax_xz.yaxis.grid(True, color='#2d3748', linewidth=0.5, linestyle='--')
    ax_xz.set_axisbelow(True)

    # Shade paired z-bands with the pair's light color
    for s_a, s_b in pairs:
        if s_a not in results or s_b not in results:
            continue
        z_lo = min(results[s_a]['z_min_in_fov'], results[s_b]['z_min_in_fov'])
        z_hi = max(results[s_a]['z_max_in_fov'], results[s_b]['z_max_in_fov'])
        ax_xz.axhspan(z_lo, z_hi, xmin=0, xmax=x_fov / (x_fov + 90),
                      color=colors[s_a], alpha=0.07, zorder=0)

    for s_n in sections:
        r = results[s_n]
        col = colors[s_n]
        is_paired = s_n in paired
        ax_xz.plot([0, x_fov], [r['z_min_in_fov'], r['z_max_in_fov']],
                   color=col,
                   linewidth=2.5 if is_paired else 1.2,
                   linestyle='-',   # all solid — color encodes pairing
                   solid_capstyle='round', alpha=0.95)
        ax_xz.text(x_fov + 5, r['z_at_fov_center'],
                   f'S{s_n}  {r["z_at_fov_center"]:.0f}µm  {r["tilt_deg"]:.2f}°',
                   color=col, fontsize=6.5, va='center',
                   fontweight='bold' if is_paired else 'normal')

    # Legend: one entry per pair + standalones
    legend_handles = []
    for idx, (s_a, s_b) in enumerate(pairs):
        if s_a not in results:
            continue
        h = mlines.Line2D([], [], color=colors[s_a], linewidth=2.5,
                          label=f'S{s_a}+S{s_b} (paired)')
        legend_handles.append(h)
    legend_handles.append(
        mlines.Line2D([], [], color='#9b9b9b', linewidth=1.2,
                      label='standalone sections')
    )
    ax_xz.legend(handles=legend_handles, fontsize=7,
                 facecolor='#1a2130', edgecolor='#2d3748',
                 labelcolor='#8b9ab0', loc='lower left')

    # ── Right: z-position chart ───────────────────────────────────────────
    ax_z = fig.add_subplot(gs[1])
    ax_z.set_facecolor('#0f1117')
    for spine in ax_z.spines.values():
        spine.set_edgecolor('#2d3748')
    ax_z.tick_params(colors='#8b9ab0', labelsize=8)
    ax_z.xaxis.grid(True, color='#2d3748', linewidth=0.5, linestyle='--')
    ax_z.set_axisbelow(True)

    y_pos     = np.arange(len(sections))
    z_centers = [results[s]['z_at_fov_center'] for s in sections]
    z_mins    = [results[s]['z_min_in_fov']    for s in sections]
    z_spans   = [results[s]['z_span_in_fov']   for s in sections]
    bar_h = 0.5

    # Background band per pair (full x-width, subtle)
    for s_a, s_b in pairs:
        if s_a not in sections or s_b not in sections:
            continue
        i_a = sections.index(s_a)
        i_b = sections.index(s_b)
        ax_z.axhspan(i_a - bar_h, i_b + bar_h,
                     color=colors[s_a], alpha=0.07, zorder=0)

    # Bars and dots
    for i, s_n in enumerate(sections):
        col = colors[s_n]
        is_paired = s_n in paired
        ax_z.barh(i, z_spans[i], left=z_mins[i], height=bar_h,
                  color=col, alpha=0.5,
                  linewidth=1.5 if is_paired else 0.3,
                  edgecolor=col if is_paired else 'none')
        ax_z.scatter(z_centers[i], i, color=col,
                     s=50 if is_paired else 25, zorder=5,
                     marker='D' if is_paired else 'o')

    # Pair connectors: bracket between the two center dots
    for s_a, s_b in pairs:
        if s_a not in sections or s_b not in sections:
            continue
        i_a = sections.index(s_a)
        i_b = sections.index(s_b)
        # Use the shared hue (lighter of the two) for the bracket
        bracket_col = colors[s_a]
        x_bracket = z_mins[i_a] - 3  # just left of bars
        ax_z.plot([x_bracket, x_bracket], [i_a, i_b],
                  color=bracket_col, linewidth=1.5, alpha=0.6, zorder=2)
        ax_z.plot([x_bracket, x_bracket + 1.5], [i_a, i_a],
                  color=bracket_col, linewidth=1.5, alpha=0.6, zorder=2)
        ax_z.plot([x_bracket, x_bracket + 1.5], [i_b, i_b],
                  color=bracket_col, linewidth=1.5, alpha=0.6, zorder=2)

    # Amber overlap boxes (in data coords before invert_yaxis)
    overlap_labeled = False
    for i in range(len(sections) - 1):
        s_a, s_b = sections[i], sections[i + 1]
        z_max_a = results[s_a]['z_max_in_fov']
        z_min_b = results[s_b]['z_min_in_fov']
        if z_max_a > z_min_b:
            rect = plt.Rectangle(
                (z_min_b, i - bar_h / 2),
                z_max_a - z_min_b,
                (i + 1 + bar_h / 2) - (i - bar_h / 2),
                color='#f4a742', alpha=0.22, zorder=1,
                label='z-range overlap' if not overlap_labeled else None
            )
            ax_z.add_patch(rect)
            overlap_labeled = True

    ax_z.set_yticks(y_pos)
    ax_z.set_yticklabels([f'S{s}' for s in sections],
                          fontsize=7.5, color='#8b9ab0')
    ax_z.set_xlabel('z depth (µm)', color='#8b9ab0', fontsize=9)
    ax_z.set_title('Section z-positions  (bar = z-span, dot = center)',
                   color='#e2e8f0', fontsize=10, pad=8)

    z_all_min = min(z_mins) - 5
    z_all_max = max(r + s for r, s in zip(z_mins, z_spans)) + 5
    ax_z.set_xlim(z_all_min, z_all_max)
    ax_z.invert_yaxis()

    paired_bar    = mpatches.Patch(color='#8b9ab0', alpha=0.6,
                                    label='paired (edged bar, ◆)')
    solo_bar      = mpatches.Patch(color='#8b9ab0', alpha=0.25,
                                    label='standalone (plain bar, ●)')
    overlap_patch = mpatches.Patch(color='#f4a742', alpha=0.4,
                                    label='z-range overlap')
    ax_z.legend(handles=[paired_bar, solo_bar, overlap_patch],
                fontsize=7, facecolor='#1a2130',
                edgecolor='#2d3748', labelcolor='#8b9ab0',
                loc='lower right')

    fig.suptitle('Xenium section positions within cortical z-stack',
                 color='#e2e8f0', fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Saved to {save_path}")
    return fig

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


###### QC
def _pick_existing_key(container, candidates):
    for k in candidates:
        if k in container:
            return k
    return None


def _unwrap_da(element):
    da = element
    for _ in range(5):
        if hasattr(da, 'data') and hasattr(da.data, 'shape'):
            return da
        if hasattr(da, 'keys') and callable(da.keys):
            da = da[next(iter(da.keys()))]
        else:
            break
    return da


def _as_pandas(df_like):
    if hasattr(df_like, 'compute'):
        return df_like.compute()
    return df_like


def _resolve_channel_index(da, channel=None):
    """Resolve channel index for a potentially multi-channel image DataArray-like object."""
    if not hasattr(da, 'data'):
        return 0, None

    arr = da.data
    if arr.ndim != 4:
        return 0, None

    # Expected order in this pipeline is typically (c, z, y, x).
    # channel can be int (index) or str (name).
    if channel is None:
        return 0, None

    if isinstance(channel, int):
        idx = int(channel)
        if idx < 0 or idx >= arr.shape[0]:
            raise ValueError(f"channel index {idx} out of range [0, {arr.shape[0]-1}]")
        return idx, None

    if isinstance(channel, str):
        try:
            names = list(get_channel_names(da))
        except Exception:
            names = []
        if len(names) == 0:
            raise ValueError("channel specified by name, but channel names are unavailable for this image")
        if channel not in names:
            raise ValueError(f"channel '{channel}' not found. Available: {names}")
        return int(names.index(channel)), channel

    raise TypeError("channel must be None, int, or str")


def _resolve_lift_slab_bounds(sdata, img_da, n_z, section_n, ref_3d_key_type='dapi_zstack'):
    """
    Determine the full-slab z bounds for a (possibly lifted) 2D-to-3D image element.

    Priority:
      1. Read z_ref_n_planes / z_lift_start_index from img_da.attrs (only available
         in-memory before a zarr write/read round-trip).
      2. Infer from the reference 3D image (dapi_zstack for that section): because
         make_element_3d always places the lifted element at
         z_start = (ref_nz - target_nz) / 2, we can reconstruct it from n_z alone.

    Returns (z_ref_n_planes, z_lift_start, lift_mode, z_slab_lo, z_slab_hi).
    """
    img_attrs = getattr(img_da, 'attrs', {}) or {}

    if img_attrs.get('z_ref_n_planes') is not None:
        # attrs survived (in-memory, not yet round-tripped through zarr)
        z_ref_n_planes = int(img_attrs['z_ref_n_planes'])
        z_lift_start   = float(img_attrs.get('z_lift_start_index', 0.0))
        lift_mode      = img_attrs.get('z_lift_mode', 'reference_slab')
    else:
        # attrs lost after zarr round-trip: infer from dapi_zstack reference
        ref_key = _pick_existing_key(
            sdata.images,
            [
                f'{ref_3d_key_type}-{section_n}' if section_n is not None else None,
                f'{ref_3d_key_type}_{section_n}' if section_n is not None else None,
                ref_3d_key_type,
            ],
        )
        if ref_key is not None:
            ref_da  = _unwrap_da(sdata.images[ref_key])
            ref_arr = ref_da.data
            z_ref_n_planes = int(ref_arr.shape[1]) if ref_arr.ndim == 4 else int(ref_arr.shape[0])
        else:
            z_ref_n_planes = n_z  # no reference found; treat as full slab

        # Reconstruct z_lift_start from the make_element_3d formula
        z_lift_start = (z_ref_n_planes - n_z) / 2.0

        if n_z == z_ref_n_planes:
            lift_mode = 'reference_slab'
        elif n_z == 1:
            lift_mode = 'centered_plane'
        else:
            lift_mode = 'centered_slab'

    # Express slab bounds in *centered* index space where 0 = section midplane.
    # Every lift mode yields the same full-reference-slab bounds:
    #   -(z_ref_n_planes-1)/2  …  +(z_ref_n_planes-1)/2
    # Callers that compute z_idx via mat_cz_inv must convert raw index → centered index by
    # subtracting  z_idx_center_offset = (z_ref_n_planes-1)/2 - z_lift_start
    # before comparing against these bounds.
    center_offset = (z_ref_n_planes - 1) / 2.0
    z_slab_lo = -center_offset
    z_slab_hi = +center_offset
    return z_ref_n_planes, z_lift_start, lift_mode, z_slab_lo, z_slab_hi


def alignment_qc_dashboard(
    sdata,
    section_n=None,
    sections_um=20.0,
    coord_sys='czstack_microns',
    img_key_type='dapi_zstack',
    channel=None,
    y_tolerance=1.0,
    dapi_signal_percentile=1.0,
    thickness_tol_um=1.0,
    max_dapi_pts=120000,
    max_tx_pts=10000,
    overlay_space='index',
    overlay_y_tolerance_idx=0.5,
    max_tx_overlay_pts=20000,
    make_plot=True,
):
    """
    Compact QC for one section (for any image channel):
      1) global z-thickness check (image vs transcripts)
      2) czstack z-overlap check in matched y-slab
      3) transform-only center alignment metric in czstack (median z offset)
      4) XZ overlay in either world (`overlay_space='world'`) or image index (`overlay_space='index'`)
      5) summary metrics panel
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import spatialdata as sd

    if overlay_space not in ('world', 'index'):
        raise ValueError("overlay_space must be 'world' or 'index'")

    # Resolve keys for either per-section or single-section sdata
    img_key = _pick_existing_key(
        sdata.images,
        [
            f'{img_key_type}_{section_n}' if section_n is not None else None,
            f'{img_key_type}-{section_n}' if section_n is not None else None,
            img_key_type,
        ],
    )
    tx_key = _pick_existing_key(
        sdata.points,
        [
            f'transcripts_{section_n}' if section_n is not None else None,
            f'transcripts-{section_n}' if section_n is not None else None,
            'transcripts',
        ],
    )

    if img_key is None:
        raise KeyError(f"Could not find image key for '{img_key_type}' in sdata.images")
    if tx_key is None:
        raise KeyError('Could not find a transcript points key in sdata.points')

    img_da = _unwrap_da(sdata.images[img_key])
    arr = img_da.data

    # Select image channel (if multichannel)
    ch_idx, ch_name = _resolve_channel_index(img_da, channel=channel)
    selected_channel = ch_name if ch_name is not None else ch_idx

    if arr.ndim == 4:
        n_z, n_y, n_x = arr.shape[1], arr.shape[2], arr.shape[3]
        mid_y = n_y // 2
        img_xz = arr[ch_idx, :, mid_y, :].compute().astype(np.float32)
    else:
        n_z, n_y, n_x = arr.shape[0], arr.shape[1], arr.shape[2]
        mid_y = n_y // 2
        img_xz = arr[:, mid_y, :].compute().astype(np.float32)

    # Resolve lift metadata (survives zarr round-trip via dapi_zstack inference)
    _z_ref_n_planes, _z_lift_start, _lift_mode, _z_slab_lo, _z_slab_hi = \
        _resolve_lift_slab_bounds(sdata, img_da, n_z, section_n)

    # Affines
    mat_global = np.array(get_transformation(img_da, to_coordinate_system='global').to_affine_matrix(
        input_axes=('x', 'y', 'z'), output_axes=('x', 'y', 'z')
    ))
    mat_cz = np.array(get_transformation(img_da, to_coordinate_system=coord_sys).to_affine_matrix(
        input_axes=('x', 'y', 'z'), output_axes=('x', 'y', 'z')
    ))
    mat_cz_inv = np.linalg.inv(mat_cz)

    # Image global z support (center-based)
    z_idx = np.arange(n_z, dtype=float)
    x0, y0 = n_x / 2.0, n_y / 2.0
    img_z_global = mat_global[2, 0] * x0 + mat_global[2, 1] * y0 + mat_global[2, 2] * z_idx + mat_global[2, 3]
    img_span_centers = float(img_z_global.max() - img_z_global.min())
    # With mps = sections_depth/(n_z-1), the span between plane-centre extremes equals
    # sections_um exactly (plane 0 at -sections_um/2, plane n_z-1 at +sections_um/2).
    img_expected_center_span = float(sections_um)
    img_expected_full_span = float(sections_um)

    # Transcript global support
    tx_global = _as_pandas(sd.transform(sdata.points[tx_key], to_coordinate_system='global'))
    tx_span_global = float(tx_global['z'].max() - tx_global['z'].min())

    pass_img_center = abs(img_span_centers - img_expected_center_span) <= thickness_tol_um
    pass_tx_global = abs(tx_span_global - sections_um) <= thickness_tol_um

    # Build image world XZ at fixed image y-index
    xx, zz = np.meshgrid(np.arange(n_x, dtype=float), np.arange(n_z, dtype=float))
    yy = np.full_like(xx, float(mid_y))
    img_x_cz = mat_cz[0, 0] * xx + mat_cz[0, 1] * yy + mat_cz[0, 2] * zz + mat_cz[0, 3]
    img_z_cz = mat_cz[2, 0] * xx + mat_cz[2, 1] * yy + mat_cz[2, 2] * zz + mat_cz[2, 3]

    y_slice_world = float(
        mat_cz[1, 0] * (n_x / 2.0) + mat_cz[1, 1] * mid_y + mat_cz[1, 2] * (n_z / 2.0) + mat_cz[1, 3]
    )

    flat_i = img_xz.ravel()
    flat_x = img_x_cz.ravel()
    flat_z = img_z_cz.ravel()
    thr = np.percentile(flat_i, dapi_signal_percentile)
    keep_sig = flat_i > thr
    sig_idx = np.flatnonzero(keep_sig)

    img_sig_x = flat_x[keep_sig]
    img_sig_z = flat_z[keep_sig]
    img_sig_i = flat_i[keep_sig]

    if len(img_sig_i) > max_dapi_pts:
        rng = np.random.default_rng(42)
        choose = rng.choice(np.arange(len(img_sig_i)), size=max_dapi_pts, replace=False)
        img_sig_x = img_sig_x[choose]
        img_sig_z = img_sig_z[choose]
        img_sig_i = img_sig_i[choose]
        sig_idx = sig_idx[choose]

    # Transcript czstack support in matched y-slab + image x support (world-based QC metric)
    tx_cz = _as_pandas(sd.transform(sdata.points[tx_key], to_coordinate_system=coord_sys))
    n_tx_total = int(len(tx_cz))
    tx_mask_y = np.ones(n_tx_total, dtype=bool)
    if 'y' in tx_cz.columns:
        tx_mask_y &= np.abs(tx_cz['y'].values - y_slice_world) <= y_tolerance

    tx_mask_x = np.ones(n_tx_total, dtype=bool)
    if len(img_sig_x) > 0:
        tx_mask_x &= (tx_cz['x'].values >= img_sig_x.min()) & (tx_cz['x'].values <= img_sig_x.max())

    tx_mask = tx_mask_y & tx_mask_x
    tx_slab = tx_cz.loc[tx_mask].copy()
    n_tx_after_y = int(tx_mask_y.sum())

    if len(tx_slab) > max_tx_pts:
        tx_slab = tx_slab.sample(max_tx_pts, random_state=42)

    img_cz_rng = (float(np.min(img_sig_z)), float(np.max(img_sig_z))) if len(img_sig_z) else (np.nan, np.nan)
    tx_cz_rng = (float(tx_slab['z'].min()), float(tx_slab['z'].max())) if len(tx_slab) else (np.nan, np.nan)

    if len(img_sig_z) > 10 and len(tx_slab) > 10:
        i_lo, i_hi = np.percentile(img_sig_z, [0.5, 99.5])
        t_lo, t_hi = np.percentile(tx_slab['z'].values, [0.5, 99.5])
        overlap = max(0.0, min(i_hi, t_hi) - max(i_lo, t_lo))
        union = max(i_hi, t_hi) - min(i_lo, t_lo)
        overlap_frac = float(overlap / union) if union > 0 else np.nan

        i_med = float(np.median(img_sig_z))
        t_med = float(np.median(tx_slab['z'].values))
        median_z_offset_cz = float(t_med - i_med)

        i_q25, i_q75 = np.percentile(img_sig_z, [25, 75])
        i_iqr = max(float(i_q75 - i_q25), 1e-6)
        median_z_offset_iqr_norm = float(median_z_offset_cz / i_iqr)
    else:
        overlap_frac = np.nan
        median_z_offset_cz = np.nan
        median_z_offset_iqr_norm = np.nan

    # Overlay-specific transcript selection (to prevent opposite tilt artifacts)
    if overlay_space == 'world':
        overlay_img_x = img_sig_x
        overlay_img_z = img_sig_z
        overlay_img_i = img_sig_i
        overlay_tx_x = tx_slab['x'].values if len(tx_slab) else np.array([])
        overlay_tx_z = tx_slab['z'].values if len(tx_slab) else np.array([])
        n_tx_overlay = int(len(overlay_tx_x))
        overlay_desc = f"world y-slab ±{y_tolerance}"
    else:
        # Image points in index XZ (centered: 0 = section midplane)
        x_idx_flat = xx.ravel()
        z_idx_flat = zz.ravel()
        overlay_img_x = x_idx_flat[sig_idx]
        overlay_img_z = z_idx_flat[sig_idx]  # will be centered below after _z_idx_center_offset is known
        overlay_img_i = flat_i[sig_idx]

        # Transform a sampled set of transcripts to image index space and filter by y-index
        if len(tx_cz) > 300000:
            tx_overlay_df = tx_cz.sample(300000, random_state=42)
        else:
            tx_overlay_df = tx_cz

        xyz = tx_overlay_df[['x', 'y', 'z']].to_numpy(dtype=np.float64)
        xyz1 = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1)
        idx_pts = xyz1 @ mat_cz_inv.T
        tx_xi = idx_pts[:, 0]
        tx_yi = idx_pts[:, 1]
        tx_zi = idx_pts[:, 2]

        # Convert raw image-index z to centered-index space (0 = section midplane).
        _z_idx_center_offset = (_z_ref_n_planes - 1) / 2.0 - _z_lift_start
        tx_zi = tx_zi - _z_idx_center_offset
        overlay_img_z = overlay_img_z - _z_idx_center_offset  # center image z indices for plotting

        mask_i = np.abs(tx_yi - mid_y) <= overlay_y_tolerance_idx
        mask_i &= (tx_xi >= 0) & (tx_xi <= (n_x - 1))
        mask_i &= (tx_zi >= _z_slab_lo) & (tx_zi <= _z_slab_hi)

        overlay_tx_x = tx_xi[mask_i]
        overlay_tx_z = tx_zi[mask_i]

        if len(overlay_tx_x) > max_tx_overlay_pts:
            rng = np.random.default_rng(42)
            keep = rng.choice(np.arange(len(overlay_tx_x)), size=max_tx_overlay_pts, replace=False)
            overlay_tx_x = overlay_tx_x[keep]
            overlay_tx_z = overlay_tx_z[keep]

        n_tx_overlay = int(len(overlay_tx_x))
        overlay_desc = f"index y-slab ±{overlay_y_tolerance_idx}"

    metrics = {
        'section_n': section_n,
        'img_key_type': img_key_type,
        'img_key': img_key,
        'channel': selected_channel,
        'tx_key': tx_key,
        'sections_um_target': float(sections_um),
        'n_z_planes': int(n_z),
        'img_expected_center_span_um': float(img_expected_center_span),
        'img_expected_full_span_um': float(img_expected_full_span),
        'img_global_span_um': float(img_span_centers),
        'tx_global_span_um': float(tx_span_global),
        'img_center_span_pass': bool(pass_img_center),
        'tx_global_span_pass': bool(pass_tx_global),
        'coord_sys': coord_sys,
        'img_cz_range': img_cz_rng,
        'tx_cz_range': tx_cz_rng,
        'overlap_fraction': overlap_frac,
        'median_z_offset_cz': median_z_offset_cz,
        'median_z_offset_iqr_norm': median_z_offset_iqr_norm,
        'n_tx_total': n_tx_total,
        'n_tx_after_y': n_tx_after_y,
        'n_tx_in_slab': int(len(tx_slab)),
        'overlay_space': overlay_space,
        'n_tx_overlay': n_tx_overlay,
        'y_tolerance': float(y_tolerance),
    }

    if make_plot:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))

        label_img = f"{img_key_type}[{selected_channel}]" if arr.ndim == 4 else img_key_type

        # (A) Global thickness check — centre both distributions for a span-only comparison
        # (absolute z positions differ because DAPI global transform was shifted by center_z=True
        # while transcript native z may have a different z-reference in global space)
        ax = axes[0, 0]
        img_z_c = img_z_global - img_z_global.mean()
        tx_z_c  = tx_global['z'].values - tx_global['z'].values.mean()
        ax.hist(img_z_c, bins=min(30, n_z), alpha=0.6, density=True, label=f'{label_img} (global, centred)')
        ax.hist(tx_z_c, bins=40, alpha=0.5, density=True, label='TX (global, centred)')
        ax.axvline( img_expected_center_span / 2, color='tab:blue',   linestyle='--', linewidth=1, label=f'img span ={img_expected_center_span:.1f}µm')
        ax.axvline(-img_expected_center_span / 2, color='tab:blue',   linestyle='--', linewidth=1)
        ax.axvline( sections_um / 2,              color='tab:orange', linestyle='--', linewidth=1, label=f'TX span ={sections_um:.1f}µm')
        ax.axvline(-sections_um / 2,              color='tab:orange', linestyle='--', linewidth=1)
        ax.set_title('Global z thickness (centred)')
        ax.set_xlabel('z − mean(z)  (µm)')
        ax.set_ylabel('density')
        ax.legend(fontsize=8)

        # (B) czstack overlap histogram
        ax = axes[0, 1]
        if len(img_sig_z) and len(tx_slab):
            bins = np.linspace(min(np.min(img_sig_z), tx_slab['z'].min()),
                               max(np.max(img_sig_z), tx_slab['z'].max()), 45)
            ax.hist(img_sig_z, bins=bins, alpha=0.55, density=True, label=f'{label_img} signal')
            ax.hist(tx_slab['z'].values, bins=bins, alpha=0.55, density=True, label='TX slab')
        ax.set_title(f'czstack z overlap (y±{y_tolerance})')
        ax.set_xlabel(f'z ({coord_sys})')
        ax.set_ylabel('density')
        ax.legend(fontsize=8)

        # (C) XZ overlay
        ax = axes[1, 0]
        if len(overlay_img_z):
            ax.scatter(overlay_img_x, overlay_img_z, c=overlay_img_i, s=1, cmap='magma', alpha=0.35, linewidths=0)
        if len(overlay_tx_z):
            ax.scatter(overlay_tx_x, overlay_tx_z, s=3, alpha=0.6, c='cyan', linewidths=0)
        if overlay_space == 'index':
            ax.set_title('XZ overlay in image index space')
            ax.set_xlabel('x (image index)')
            ax.set_ylabel('z (image index)')
        else:
            ax.set_title('XZ overlay in czstack world')
            ax.set_xlabel(f'x ({coord_sys})')
            ax.set_ylabel(f'z ({coord_sys})')

        # (D) Text summary
        ax = axes[1, 1]
        ax.axis('off')
        _lift_summary = (
            f"{_lift_mode}  ref_nz={_z_ref_n_planes}  start={_z_lift_start:.1f}\n"
            f"Slab idx bounds: [{_z_slab_lo:.1f}, {_z_slab_hi:.1f}]\n"
        ) if _lift_mode != 'reference_slab' else f"{_lift_mode}\n"
        txt = (
            f"Section: {section_n if section_n is not None else 'single'}\n"
            f"Image key: {img_key}\nChannel: {selected_channel}\nTX key: {tx_key}\n"
            f"Lift: {_lift_summary}\n"
            f"Image global span: {img_span_centers:.3f} µm\n"
            f"Image expected (center-span): {img_expected_center_span:.3f} µm\n"
            f"TX global span: {tx_span_global:.3f} µm\n"
            f"TX expected: {sections_um:.3f} µm\n\n"
            f"PASS image center-span: {pass_img_center}\n"
            f"PASS TX span: {pass_tx_global}\n\n"
            f"Image cz range: [{img_cz_rng[0]:.2f}, {img_cz_rng[1]:.2f}]\n"
            f"TX cz range: [{tx_cz_rng[0]:.2f}, {tx_cz_rng[1]:.2f}]\n"
            f"Overlap fraction: {overlap_frac:.3f}\n"
            f"Median z offset (TX-image): {median_z_offset_cz:.3f}\n"
            f"Offset / image IQR: {median_z_offset_iqr_norm:.3f}\n"
            f"Overlay space: {overlay_space} ({overlay_desc})\n"
            f"n TX total: {n_tx_total}\n"
            f"n TX after y-filter: {n_tx_after_y}\n"
            f"n TX in slab: {len(tx_slab)}\n"
            f"n TX in overlay: {n_tx_overlay}"
        )
        ax.text(0.02, 0.98, txt, va='top', ha='left', fontsize=10, family='monospace')

        plt.tight_layout()
        plt.show()

    return metrics

def diagnose_transcript_z_within_section(
    sdata,
    section_n,
    img_key_type='rna',
    coord_sys='czstack_microns',
):
    """
    Check whether transcripts for a section lie within the section's z slab.

    Correctly handles 'centered_plane' and 'centered_slab' lift modes: the out-of-slab
    count is measured against the FULL REFERENCE SLAB bounds (not the image's n_z planes),
    so a single-plane centered image does not report 99%+ out of bounds.
    """
    import numpy as np
    import spatialdata as sd

    # Resolve keys
    img_key = None
    for k in [f'{img_key_type}-{section_n}', f'{img_key_type}_{section_n}', img_key_type]:
        if k in sdata.images:
            img_key = k
            break
    tx_key = None
    for k in [f'transcripts-{section_n}', f'transcripts_{section_n}', 'transcripts']:
        if k in sdata.points:
            tx_key = k
            break

    if img_key is None or tx_key is None:
        raise KeyError(f'Could not resolve keys. img_key={img_key}, tx_key={tx_key}')

    img_da = _unwrap_da(sdata.images[img_key])
    arr = img_da.data
    if arr.ndim == 4:
        n_z = int(arr.shape[1])
    elif arr.ndim == 3:
        n_z = int(arr.shape[0])
    else:
        raise ValueError(f'Unsupported image ndim={arr.ndim}')

    # Resolve lift metadata (survives zarr round-trip via dapi_zstack inference)
    z_ref_n_planes, z_lift_start, lift_mode, z_slab_lo, z_slab_hi = \
        _resolve_lift_slab_bounds(sdata, img_da, n_z, section_n)

    mat = np.array(
        get_transformation(img_da, to_coordinate_system=coord_sys).to_affine_matrix(
            input_axes=('x', 'y', 'z'), output_axes=('x', 'y', 'z')
        ),
        dtype=np.float64,
    )
    mat_inv = np.linalg.inv(mat)

    tx = sd.transform(sdata.points[tx_key], to_coordinate_system=coord_sys)
    tx_pdf = tx.compute() if hasattr(tx, 'compute') else tx

    xyz = tx_pdf[['x', 'y', 'z']].to_numpy(dtype=np.float64)
    xyz1 = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1)
    idx = xyz1 @ mat_inv.T
    z_idx = idx[:, 2]

    # Convert raw image-index z to centered-index space (0 = section midplane).
    # Raw z_idx from mat_inv includes the lift offset; subtracting z_idx_center_offset
    # aligns it with the centered slab bounds from _resolve_lift_slab_bounds.
    _z_idx_center_offset = (z_ref_n_planes - 1) / 2.0 - z_lift_start
    z_idx = z_idx - _z_idx_center_offset

    # Out-of-bounds relative to the full reference slab, not just this image's n_z planes.
    out_low  = int(np.sum(z_idx < z_slab_lo))
    out_high = int(np.sum(z_idx > z_slab_hi))
    out_any  = out_low + out_high
    frac_out = float(out_any / len(z_idx)) if len(z_idx) else np.nan

    print(f'Section {section_n}  | image={img_key}  tx={tx_key}')
    print(f'n_tx={len(z_idx)}  n_z_image={n_z}  lift_mode={lift_mode}')
    print(f'z_ref_n_planes={z_ref_n_planes}  z_lift_start={z_lift_start:.1f}')
    print(f'Slab bounds (centered idx space): [{z_slab_lo:.1f}, {z_slab_hi:.1f}]')
    print(f'z_idx min/max: {z_idx.min():.3f} / {z_idx.max():.3f}')
    print(f'out_of_slab: {out_any} ({frac_out:.3%})  [low={out_low}, high={out_high}]')

    return {
        'section_n': section_n,
        'img_key': img_key,
        'tx_key': tx_key,
        'n_tx': int(len(z_idx)),
        'n_z_image': int(n_z),
        'lift_mode': lift_mode,
        'z_ref_n_planes': int(z_ref_n_planes),
        'z_lift_start': float(z_lift_start),
        'z_slab_lo': float(z_slab_lo),
        'z_slab_hi': float(z_slab_hi),
        'z_idx_min': float(z_idx.min()),
        'z_idx_max': float(z_idx.max()),
        'out_of_slab_n': int(out_any),
        'out_of_slab_fraction': float(frac_out),
    }


def alignment_qc_summary_by_section(
    combined_data,
    sections,
    sections_um=20.0,
    img_key_type='dapi_zstack',
    channel=None,
    coord_sys='czstack_microns',
    y_tolerance=1.0,
):
    """Run compact QC across sections and plot a concise summary."""
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    rows = []
    for s_n in sections:
        try:
            m = alignment_qc_dashboard(
                combined_data,
                section_n=s_n,
                sections_um=sections_um,
                img_key_type=img_key_type,
                channel=channel,
                coord_sys=coord_sys,
                y_tolerance=y_tolerance,
                make_plot=False,
            )
            rows.append(m)
        except Exception as e:
            rows.append({'section_n': s_n, 'error': str(e)})

    df = pd.DataFrame(rows).sort_values('section_n').reset_index(drop=True)

    ok = df[~df.get('error', pd.Series([None] * len(df))).notna()].copy()
    if len(ok) == 0:
        display(df)
        return df

    fig, axes = plt.subplots(1, 4, figsize=(19, 4))

    # 1) Thickness spans
    axes[0].plot(ok['section_n'], ok['img_global_span_um'], 'o-', label='Image global span')
    axes[0].plot(ok['section_n'], ok['tx_global_span_um'], 'o-', label='TX global span')
    axes[0].axhline(sections_um, linestyle='--', color='k', linewidth=1, label='20 µm target')
    axes[0].set_title('Global thickness by section')
    axes[0].set_xlabel('section')
    axes[0].set_ylabel('span (µm)')
    axes[0].legend(fontsize=8)

    # 2) Overlap fraction
    axes[1].bar(ok['section_n'].astype(str), ok['overlap_fraction'])
    axes[1].set_ylim(0, 1)
    axes[1].set_title('czstack z overlap fraction')
    axes[1].set_xlabel('section')
    axes[1].set_ylabel('overlap (0-1)')

    # 3) transform-only metric (center offset)
    axes[2].axhline(0.0, linestyle='--', color='k', linewidth=1)
    axes[2].plot(ok['section_n'], ok['median_z_offset_cz'], 'o-', color='tab:green')
    axes[2].set_title('Median z offset (TX - image)')
    axes[2].set_xlabel('section')
    axes[2].set_ylabel(f'offset ({coord_sys} z units)')

    # 4) czstack ranges
    i_mins = ok['img_cz_range'].apply(lambda x: x[0])
    i_maxs = ok['img_cz_range'].apply(lambda x: x[1])
    t_mins = ok['tx_cz_range'].apply(lambda x: x[0])
    t_maxs = ok['tx_cz_range'].apply(lambda x: x[1])

    y = np.arange(len(ok))
    axes[3].hlines(y + 0.12, i_mins, i_maxs, linewidth=3, label='Image')
    axes[3].hlines(y - 0.12, t_mins, t_maxs, linewidth=3, label='TX')
    axes[3].set_yticks(y)
    axes[3].set_yticklabels(ok['section_n'].astype(str))
    axes[3].set_title(f'czstack z-ranges ({coord_sys})')
    axes[3].set_xlabel('z')
    axes[3].set_ylabel('section')
    axes[3].legend(fontsize=8)

    plt.suptitle(f"QC summary: img_key_type={img_key_type}, channel={channel}", y=1.02, fontsize=11)
    plt.tight_layout()
    plt.show()
    return df