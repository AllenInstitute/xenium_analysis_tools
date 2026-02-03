import spatialdata as sd
from xenium_analysis_tools.utils.sd_utils import add_micron_coord_sys
from spatialdata.models import Image2DModel, Image3DModel, Labels3DModel, Labels2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations import get_transformation, set_transformation
import anndata as ad
from spatialdata import get_element_instances
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import tifffile
import json
import re
from xenium_analysis_tools.utils.sd_utils import (
    add_mapped_cells_cols,
    get_transcripts_bboxes
)

def create_zstack_array(tif_path,
                        add_chan=True,
                        fov_z_um=450.0,
                        fov_x_um=400.0, 
                        fov_y_um=400.0):

    data = tifffile.imread(tif_path)
    pixels_z, pixels_y, pixels_x = data.shape
    
    # Pixel sizes
    pixel_size_z = fov_z_um / pixels_z 
    pixel_size_y = fov_y_um / pixels_y 
    pixel_size_x = fov_x_um / pixels_x 
    
    # Create coordinate arrays with proper spacing
    z_coords = np.arange(pixels_z) * pixel_size_z
    y_coords = np.arange(pixels_y) * pixel_size_y
    x_coords = np.arange(pixels_x) * pixel_size_x
    coords = {"z": z_coords,
                "y": y_coords, 
                "x": x_coords}

    if add_chan:
        data = np.expand_dims(data, axis=0)
        coords["c"] = np.arange(data.shape[0])
        dims = ("c", "z", "y", "x")
    else:
        dims = ("z", "y", "x")

    # Create xarray DataArray with improved metadata
    da = xr.DataArray(
        data,
        coords=coords,
        dims=dims,
        attrs={
            "pixel_size_um_z": pixel_size_z,
            "pixel_size_um_y": pixel_size_y,
            "pixel_size_um_x": pixel_size_x,
            "fov_um_z": fov_z_um,
            "fov_um_y": fov_y_um,
            "fov_um_x": fov_x_um,
            "units": "micrometers",
        }
    )
    return da

def get_zstacks_dict(zstacks_folder, channels=['gcamp', 'dextran']):
    zstacks_dict = {}
    
    # Process only directories
    stack_dirs = [d for d in zstacks_folder.iterdir() if d.is_dir()]
    
    for stack_ind, stack_folder in enumerate(stack_dirs):
        stack_info = {
            'zstack_name': stack_folder.name,
            'zstack_size': _extract_zstack_size(stack_folder.name),
            'zstack_channels': [ch for ch in channels if ch in stack_folder.name.lower()],
            'metadata_jsons': {'registration': None, 'roi_groups': None, 'scanimage': None},
            'channel_tifs': {}
        }
        
        # Process files in stack folder
        chan_ind = 0
        for item in sorted(stack_folder.iterdir()):
            if item.is_file() and item.suffix.lower() == '.json':
                # Categorize JSON metadata files
                json_type = _categorize_json_file(item.name.lower())
                if json_type:
                    stack_info['metadata_jsons'][json_type] = item
                    
            elif item.is_dir() and 'channel' in item.name.lower():
                # Process channel directories
                tif_files = [f for f in item.iterdir() if f.suffix.lower() == '.tif']
                stack_info['channel_tifs'][chan_ind] = {
                    'chan_name': item.name,
                    'chan_tif_path': tif_files[0] if len(tif_files) == 1 else tif_files
                }
                chan_ind += 1
        
        zstacks_dict[stack_ind] = stack_info
    
    return zstacks_dict

def _extract_zstack_size(zstack_name):
    """Extract width x height x depth from stack name."""
    size_pattern = re.search(r'(\d+)x(\d+)x(\d+)', zstack_name)
    if size_pattern:
        width, height, depth = map(int, size_pattern.groups())
        return {"width": width, "height": height, "depth": depth}
    return {"width": None, "height": None, "depth": None}

def _categorize_json_file(filename_lower):
    """Categorize JSON file by its name."""
    if 'registration' in filename_lower:
        return 'registration'
    elif 'roi_groups' in filename_lower:
        return 'roi_groups' 
    elif 'scanimage' in filename_lower:
        return 'scanimage'
    return None

def get_zstack(zstacks_dict, zstack_ind=None, zstack_name=None, zstack_size=None, zstack_channels=None):    
    if zstack_ind is not None:
        if zstack_ind not in zstacks_dict:
            raise ValueError(f"Z-stack index {zstack_ind} not found in zstacks_dict.")
        return zstacks_dict[zstack_ind]
    
    # Helper function to find matches
    def _find_matches(criterion_func, criterion_name, criterion_value):
        matches = [i for i, stack in zstacks_dict.items() if criterion_func(stack)]
        
        if not matches:
            raise ValueError(f"{criterion_name} {criterion_value} not found in zstacks_dict.")
        
        if len(matches) == 1:
            return zstacks_dict[matches[0]]
            
        # Handle multiple matches with optional channel filtering
        if zstack_channels is not None:
            channel_matches = [
                i for i in matches 
                if set(zstacks_dict[i]['zstack_channels']) == set(zstack_channels)
            ]
            if len(channel_matches) == 1:
                return zstacks_dict[channel_matches[0]]
            elif len(channel_matches) > 1:
                raise ValueError(f"Multiple z-stacks found with {criterion_name} {criterion_value} and channels {zstack_channels}. Found {len(channel_matches)} matches.")
            else:
                raise ValueError(f"No z-stack found with {criterion_name} {criterion_value} and channels {zstack_channels}.")
        raise ValueError(f"Multiple z-stacks found with {criterion_name} {criterion_value}. Found {len(matches)} matches. Consider specifying channels parameter.")
    
    if zstack_name is not None:
        return _find_matches(
            lambda stack: stack['zstack_name'] == zstack_name,
            "Z-stack name", zstack_name
        )
    
    if zstack_size is not None:
        return _find_matches(
            lambda stack: (
                stack['zstack_size']['width'] == zstack_size['width'] and
                stack['zstack_size']['height'] == zstack_size['height'] and
                stack['zstack_size']['depth'] == zstack_size['depth']
            ),
            "Stack size", zstack_size
        )
    
    raise ValueError("Either zstack_ind, zstack_name, or zstack_size must be provided.")

def get_label_params(label_obj, id_name='cell'):
    from skimage.measure import regionprops
    labels = label_obj.values
    props = regionprops(labels)
    data = [
        {f'{id_name}_id': p.label, 
        'z': p.centroid[0] if len(p.centroid)==3 else None,
        'y': p.centroid[1] if len(p.centroid)==3 else p.centroid[0],
        'x': p.centroid[2] if len(p.centroid)==3 else p.centroid[1],
        'area': p.area,
        'bbox': p.bbox}
        for p in props
    ]
    df = pd.DataFrame(data)
    return df

def get_zstack_sdata(stack, zstack_masks=None):
    # Create the z-stack image array
    chan_arrays = {}
    for ch_ind, chan_name in enumerate(stack['zstack_channels']):
        chan_img = create_zstack_array(tif_path=stack['channel_tifs'][ch_ind]['chan_tif_path'], 
                    fov_x_um=stack['zstack_size']['width'], 
                    fov_y_um=stack['zstack_size']['height'], 
                    fov_z_um=stack['zstack_size']['depth'])
        chan_img = Image3DModel.parse(
                chan_img,
                dims=['c', 'z', 'y', 'x'],
                c_coords=chan_name,
                chunks='auto',
            )
        chan_arrays[chan_name] = chan_img

    if zstack_masks is not None:
        zstack_labels = {}
        # Get labels for each channel
        for mask_ind, masks in zstack_masks['channel_tifs'].items():
            channel_name = zstack_masks['zstack_channels'][mask_ind]
            zstack_label = create_zstack_array(tif_path=masks['chan_tif_path'], 
                        fov_x_um=zstack_masks['zstack_size']['width'], 
                        fov_y_um=zstack_masks['zstack_size']['height'], 
                        fov_z_um=zstack_masks['zstack_size']['depth'],
                        add_chan=False)

            zstack_label = Labels3DModel.parse(
                        zstack_label,
                        dims=['z', 'y', 'x'],
                        chunks='auto',
                    )
            zstack_labels[f"{channel_name}_labels"] = zstack_label
    
        tables = {}
        for label_name, labels_obj in zstack_labels.items():
            chan_name = label_name.replace('_labels','')
            label_type_id = f'{chan_name}_id'
            chan_label_ids = get_element_instances(labels_obj).values
            obs = pd.DataFrame(chan_label_ids, columns=[label_type_id])
            cells_df = get_label_params(labels_obj, id_name=chan_name)
            cells_df['region'] = label_name
            obs = obs.merge(cells_df, left_on=label_type_id, right_on=chan_name+'_id', how='left')
            table = ad.AnnData(obs=obs, obsm={'spatial': obs[['z','y','x']].values})
            table = TableModel.parse(table, region=label_name, region_key='region', instance_key=label_type_id)
            tables[f'{chan_name}_cells'] = table

    # Assemble SpatialData
    zstack_sdata = sd.SpatialData(
        images={**chan_arrays},
        labels={**zstack_labels} if zstack_masks is not None else {},
        tables={**tables} if zstack_masks is not None else {}
    )    

    # Determine pixel sizes
    zstack_chan = zstack_sdata[stack['zstack_channels'][0]] # Use first channel for pixel size reference
    if zstack_chan.attrs['pixel_size_um_x'] == zstack_chan.attrs['pixel_size_um_y']:
        pixel_size = zstack_chan.attrs['pixel_size_um_x']
    if zstack_chan.attrs['fov_um_z']==zstack_chan.shape[1]:
        z_step_size = 1

    # Add micron coordinate system if not already present
    if 'microns' not in zstack_sdata.coordinate_systems:
        zstack_sdata = add_micron_coord_sys(zstack_sdata, pixel_size=pixel_size, z_step=z_step_size)
    else:
        print("Micron coordinate system already exists")
    return zstack_sdata

def get_alignment_spatial_elements(sdata, scale_from_level=2, channel_names=['dapi', 'boundary', 'rna', 'protein'], keep_coord_systems=['global', 'microns']):
    # Technically should only need to replace morphology focus transforms, but doing for all elements just in case
    # For elements, get at a specific scale level (if multi-scale) and set transform to global coordinate system
    # Images
    # Dapi z-stack
    dapi_zstack_level = sdata['dapi_zstack'][f'scale{scale_from_level}'].image
    el_transforms = {}
    for cs in list(dapi_zstack_level.attrs['transform'].keys()):
        if cs in keep_coord_systems:
            el_transforms[cs] = get_transformation(dapi_zstack_level, to_coordinate_system=cs)
    # dapi_zstack_global_tf = get_transformation(dapi_zstack_level, to_coordinate_system='global')
    dapi_zstack = Image3DModel.parse(sdata['dapi_zstack'][f'scale{scale_from_level}'].image,
                                                        dims=['c', 'z', 'y', 'x'],
                                                        c_coords=['DAPI'],
                                                        chunks='auto',
                                                    )
    for cs, tf in el_transforms.items():
        set_transformation(dapi_zstack, tf, to_coordinate_system=cs)
    # set_transformation(dapi_zstack, dapi_zstack_global_tf, to_coordinate_system='global')

    # Morphology focus channels    
    mf_chans_level = sdata['morphology_focus'][f'scale{scale_from_level}'].image
    el_transforms = {}
    for cs in list(mf_chans_level.attrs['transform'].keys()):
        if cs in keep_coord_systems:
            el_transforms[cs] = get_transformation(mf_chans_level, to_coordinate_system=cs)
    # mf_img_global_tf = get_transformation(mf_chans_level, to_coordinate_system='global')
    chans_arrays = {}
    for chan_ind, chan in enumerate(channel_names):
        chan_img = sdata['morphology_focus'][f'scale{scale_from_level}'].image[chan_ind]
        chan_img = np.expand_dims(chan_img.data, axis=0)
        chans_arrays[chan] = Image2DModel.parse(chan_img,
                                                dims=['c', 'y', 'x'],
                                                c_coords=chan,
                                                chunks='auto',
                                            )
        for cs, tf in el_transforms.items():    
            set_transformation(chans_arrays[chan], tf, to_coordinate_system=cs)
        # set_transformation(chans_arrays[chan], mf_img_global_tf, to_coordinate_system='global')

    images = {'dapi_zstack': dapi_zstack, **chans_arrays}

    # Labels
    cell_labels_level = sdata['cell_labels'][f'scale{scale_from_level}'].image
    for cs in list(cell_labels_level.attrs['transform'].keys()):
        if cs in keep_coord_systems:
            el_transforms[cs] = get_transformation(cell_labels_level, to_coordinate_system=cs)
    # cell_labels_tf = get_transformation(cell_labels_level, to_coordinate_system='global')
    cell_labels = Labels2DModel.parse(cell_labels_level, dims=['y', 'x'], chunks='auto')
    for cs, tf in el_transforms.items():
        set_transformation(cell_labels, tf, to_coordinate_system=cs)
    # set_transformation(cell_labels, cell_labels_tf, to_coordinate_system='global')
    nucleus_labels_level = sdata['nucleus_labels'][f'scale{scale_from_level}'].image
    nucleus_labels = Labels2DModel.parse(nucleus_labels_level, dims=['y', 'x'], chunks='auto')
    # set_transformation(nucleus_labels, cell_labels_tf, to_coordinate_system='global')
    for cs, tf in el_transforms.items():
        set_transformation(nucleus_labels, tf, to_coordinate_system=cs)

    labels = {
        'cell_labels': cell_labels,
        'nucleus_labels': nucleus_labels,
    }

    return images, labels

def get_alignment_shapes_tables(sdata, 
                    transcripts_qv_thresh=20, 
                    annotate_spatial_elements='cell_boundaries',
                    cell_id_name='cell_id',
                    mask_id_name='cell_labels',
                    keep_coord_systems=['global', 'microns']):
    # Make cell_id to cell_label mapping dictionary
    cell_id_label_dict = dict(zip(sdata['table'].obs[cell_id_name].values, sdata['table'].obs[mask_id_name].values))
    # Transcripts transforms
    points_transforms = {}
    for cs in list(sdata['transcripts'].attrs['transform'].keys()):
        if cs in keep_coord_systems:
            points_transforms[cs] = get_transformation(sdata['transcripts'], to_coordinate_system=cs)
    
    transcripts = sdata['transcripts'].compute()
    # Drop transcripts not included in counts
    transcripts = transcripts[transcripts['qv']>=transcripts_qv_thresh]
    # Add cell_labels to transcripts based on cell_id
    transcripts[mask_id_name] = transcripts[cell_id_name].map(cell_id_label_dict).fillna(0).astype('int64')
    # Annotate spatial elements (e.g., cell_boundaries) with cell_labels
    sdata[annotate_spatial_elements][cell_id_name] = sdata[annotate_spatial_elements].index.values
    sdata[annotate_spatial_elements][mask_id_name] = sdata[annotate_spatial_elements][cell_id_name].map(cell_id_label_dict).values
    sdata[annotate_spatial_elements].set_index(mask_id_name, inplace=True, drop=False)
    # Update annotation regions
    table = sdata['table'].copy()
    table.obs['region'] = annotate_spatial_elements
    table.obs['region'] = pd.Categorical(table.obs['region'])
    table.uns['spatialdata_attrs'].update({
        'region_key': 'region',
        'region': [annotate_spatial_elements],
        'instance_key': mask_id_name
    })

    # Parse shapes
    annotated_shape = ShapesModel.parse(sdata[annotate_spatial_elements])
    shapes = sdata.shapes
    shapes[annotate_spatial_elements] = annotated_shape
    # Parse table
    table = TableModel.parse(table)
    # Parse transcripts
    transcripts = PointsModel.parse(transcripts)
    for cs, tf in points_transforms.items():
        set_transformation(transcripts, tf, to_coordinate_system=cs)
    
    return table, transcripts, shapes

def generate_zstack(zstack_path, zstack_masks_path, zstack_size=None, zstack_ind=None, zstack_channels=None, save_folder=None):
    # Make the dictionary for the available z-stacks
    zstacks_dict = get_zstacks_dict(zstack_path)
    zstacks_masks_dict = get_zstacks_dict(zstack_masks_path)
    print(f'Number of z-stacks found: {len(zstacks_dict)}\n')
    for stack_ind, stack_info in zstacks_dict.items():
        print(f"Stack {stack_ind}: {stack_info['zstack_name']}")
        zstack_width = stack_info['zstack_size']['width']
        zstack_height = stack_info['zstack_size']['height']
        zstack_depth = stack_info['zstack_size']['depth']
        print(f"  Size: {zstack_width} W x {zstack_height} H x {zstack_depth} D")
        print(f"  Channels: {stack_info['zstack_channels']}\n")

    # Select the zstack that matches the criteria
    if len(zstacks_dict) == 1:
        zstack_ind = 0
        zstack_size = None
        zstack_channels = None
    zstack_info = get_zstack(zstacks_dict, zstack_ind=zstack_ind, zstack_size=zstack_size, zstack_channels=zstack_channels)
    zstack_size = zstack_info['zstack_size']
    zstack_save_name = f"zstack_{zstack_size['width']}x{zstack_size['height']}x{zstack_size['depth']}.zarr"
    zstack_save_path = save_folder / zstack_save_name
    zstack_masks = get_zstack(zstacks_masks_dict, zstack_size=zstack_size)
    zstack_sdata = get_zstack_sdata(zstack_info, zstack_masks=zstack_masks)
    for table_name, table in zstack_sdata.tables.items():
        table.uns = {'zstack_name': zstack_info['zstack_name']}
    zstack_sdata.write(zstack_save_path)
    print(f"Zstack saved at: {zstack_save_path}")
   
def generate_section_alignment_data(section_n, 
                                    paths, 
                                    save_folder):
    section_sdata_path = paths['sdata_path'] / f'section_{section_n}.zarr'
    save_section_path = save_folder / f'xenium_section_{section_n}.zarr'

    if save_section_path.exists():
        print(f"Section alignment data already exists: {save_section_path}")
        xenium_section = sd.read_zarr(save_section_path)
    else:
        section_sdata = sd.read_zarr(section_sdata_path)

        # Add micron coordinate system
        section_sdata = add_micron_coord_sys(section_sdata)

        # Add mapped cell type columns if mapped data is available
        mapped_h5ad_path = paths['data_root'] / f"{paths['xenium_dataset_name']}_mapped" / f'section_{section_n}.h5ad'
        if mapped_h5ad_path.exists():
            section_sdata = add_mapped_cells_cols(section_sdata, mapped_h5ad_path)
        else:
            print(f"Mapped h5ad file not found: {mapped_h5ad_path}")

        # Reformat section data to only include elements needed for alignment
        print("Generating alignment spatial data...")
        alignment_images, alignment_labels = get_alignment_spatial_elements(section_sdata)
        alignment_table, alignment_transcripts, alignment_shapes = get_alignment_shapes_tables(section_sdata)
        xenium_section = sd.SpatialData(
            images={**alignment_images},
            labels={**alignment_labels},
            tables={'table': alignment_table},
            points={'transcripts': alignment_transcripts},
            shapes={**alignment_shapes}
        )

        # Get transcript bounding boxes
        cell_label_bboxes = get_transcripts_bboxes(xenium_section['transcripts'], id_col='cell_labels')
        xenium_section['table'].obs['transcripts_bbox'] = xenium_section['table'].obs['cell_labels'].map(cell_label_bboxes)

        # Save the xenium section data for alignment, then reload
        xenium_section.write(save_section_path)
        del section_sdata, xenium_section
        xenium_section = sd.read_zarr(save_section_path)
    return xenium_section


def generate_channel_tifs(sdata,
                          channels,
                          save_folder,
                          overwrite=False):
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    for chan_name in channels:
        if chan_name not in sdata:
            print(f"Skipping missing channel: {chan_name}")
            continue

        out_path = save_folder / f"{chan_name}.tif"
        if out_path.exists() and not overwrite:
            print(f"Exists (skip): {out_path}")
            continue
        
        da = sdata[chan_name]
        
        # 1. Extract the resolution from the 'microns' transform sequence
        microns_transform = get_transformation(da, to_coordinate_system='microns')
        if hasattr(microns_transform, 'transformations'):
            scale_factors = [t.scale for t in microns_transform.transformations if hasattr(t, 'scale')]
        else:
            scale_factors = [microns_transform.scale] if hasattr(microns_transform, 'scale') else []
        
        # final_resolution is [y_scale, x_scale] in microns/pixel
        final_resolution = np.prod(scale_factors, axis=0)
        
        # 2. Convert to pixels per micron for TIFF metadata (1 / microns_per_pixel)
        # tifffile expects (x_resolution, y_resolution)
        res_yx = 1.0 / final_resolution
        tif_res = (res_yx[1], res_yx[0]) 

        if hasattr(da, "dims") and 'c' in da.dims and da.sizes.get('c', 0) == 1:
            da = da.isel(c=0)
            
        arr = da.values if hasattr(da, 'values') else np.asarray(da)
        axes = ''.join(da.dims) if hasattr(da, 'dims') else \
                    ('zyx' if arr.ndim == 3 else ('yx' if arr.ndim == 2 else 'c' + ''.join(map(str, range(arr.ndim)))))

        # 3. Add 'unit' to metadata so ImageJ recognizes 'um'
        meta = {
            'axes': axes,
            'unit': 'um'
        }

        tifffile.imwrite(
            str(out_path),
            arr.astype('uint16', copy=False), # Ensure consistent dtype for ImageJ
            imagej=True,
            resolution=tif_res,
            metadata=meta
        )
        print(f"Wrote: {out_path} with resolution {final_resolution} um/px")

def generate_annotated_masks(sdata, 
                            label_key, 
                            column_name, 
                            categories, 
                            save_folder,
                            table_name='table',
                            table_labels_key='cell_labels',
                            overwrite=False):
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    out_path = save_folder / f"{'_'.join(categories)}_mask.tif"
    if out_path.exists() and not overwrite:
        print(f"Exists (skip): {out_path}")
        return

    # 1. Identify valid IDs from the table
    table = sdata[table_name]
    mask_indices = table.obs[column_name].isin(categories)
    # Ensure this column corresponds to the integer values in your label array
    valid_ids = table.obs[table_labels_key][mask_indices].values.astype(int)

    # 2. Get the labels element
    labels_el = sdata.labels[label_key]
    
    # 3. Extract the resolution from the 'microns' transform sequence
    microns_transform = get_transformation(labels_el, to_coordinate_system='microns')
    scale_factors = [t.scale for t in microns_transform.transformations if hasattr(t, 'scale')]
    
    # final_resolution is [y_scale, x_scale] in microns/pixel
    final_resolution = np.prod(scale_factors, axis=0)
    
    # Convert to pixels per micron for TIFF (tifffile expects x, y)
    res_yx = 1.0 / final_resolution
    tif_res = (res_yx[1], res_yx[0]) 

    # 4. Resolve the numpy array
    label_arr = labels_el.values
    if label_arr.ndim == 3: # Handle (C, Y, X)
        label_arr = label_arr[0]

    # 5. Create the mask: keep original ID if it's in our list, else 0
    filtered_mask = np.where(np.isin(label_arr, valid_ids), label_arr, 0).astype('uint16')
    
    # 6. Save with ImageJ physical units
    tifffile.imwrite(
        str(out_path),
        filtered_mask,
        imagej=True,
        resolution=tif_res,
        metadata={'unit': 'um', 'axes': 'YX'}
    )
    print(f"Wrote: {out_path} with resolution {final_resolution} um/px")
