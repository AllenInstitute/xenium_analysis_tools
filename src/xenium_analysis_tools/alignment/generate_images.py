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

def get_alignment_data_paths(dataset_id, 
                            data_root=Path('/root/capsule/data'),
                            scratch_root=Path('/root/capsule/scratch'),
                            results_root=Path('/root/capsule/results'),
                            code_root=Path('/root/capsule/code')):
    datasets_naming_dict_path = code_root / 'datasets_names_dict.json'
    with open(datasets_naming_dict_path) as f:
        datasets_naming_dict = json.load(f)
    dataset_id = str(dataset_id)  # Ensure string format
    dataset_config = datasets_naming_dict[dataset_id]
    
    paths = {
        "data_root": data_root,
        "scratch_root": scratch_root,
        "results_root": results_root,
        "xenium_dataset_name": dataset_config["xenium_name"],
        "sdata_path": data_root / f'{dataset_config["xenium_name"]}_processed',
        "confocal_path": data_root / dataset_config["confocal_name"],
        "zstack_path": data_root / dataset_config["zstack_name"],
        "zstack_masks": data_root / dataset_config["zstack_masks_name"]
    }
    
    return paths

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

def get_alignment_spatial_elements(sdata, scale_from_level=2, channel_names=['dapi', 'boundary', 'rna', 'protein']):
    # Technically should only need to replace morphology focus transforms, but doing for all elements just in case
    # For elements, get at a specific scale level (if multi-scale) and set transform to global coordinate system
    # Images
    # Dapi z-stack
    dapi_zstack_level = sdata['dapi_zstack'][f'scale{scale_from_level}'].image
    dapi_zstack_global_tf = get_transformation(dapi_zstack_level, to_coordinate_system='global')
    dapi_zstack = Image3DModel.parse(sdata['dapi_zstack'][f'scale{scale_from_level}'].image,
                                                        dims=['c', 'z', 'y', 'x'],
                                                        c_coords=['DAPI'],
                                                        chunks='auto',
                                                    )
    set_transformation(dapi_zstack, dapi_zstack_global_tf, to_coordinate_system='global')

    # Morphology focus channels    
    mf_chans_level = sdata['morphology_focus'][f'scale{scale_from_level}'].image
    mf_img_global_tf = get_transformation(mf_chans_level, to_coordinate_system='global')
    chans_arrays = {}
    for chan_ind, chan in enumerate(channel_names):
        chan_img = sdata['morphology_focus'][f'scale{scale_from_level}'].image[chan_ind]
        chan_img = np.expand_dims(chan_img.data, axis=0)
        chans_arrays[chan] = Image2DModel.parse(chan_img,
                                                dims=['c', 'y', 'x'],
                                                c_coords=chan,
                                                chunks='auto',
                                            )
        set_transformation(chans_arrays[chan], mf_img_global_tf, to_coordinate_system='global')

    images = {'dapi_zstack': dapi_zstack, **chans_arrays}

    # Labels
    cell_labels_level = sdata['cell_labels'][f'scale{scale_from_level}'].image
    cell_labels_tf = get_transformation(cell_labels_level, to_coordinate_system='global')
    cell_labels = Labels2DModel.parse(cell_labels_level, dims=['y', 'x'], chunks='auto')
    set_transformation(cell_labels, cell_labels_tf, to_coordinate_system='global')
    nucleus_labels_level = sdata['nucleus_labels'][f'scale{scale_from_level}'].image
    nucleus_labels = Labels2DModel.parse(nucleus_labels_level, dims=['y', 'x'], chunks='auto')
    set_transformation(nucleus_labels, cell_labels_tf, to_coordinate_system='global')

    labels = {
        'cell_labels': cell_labels,
        'nucleus_labels': nucleus_labels,
    }

    return images, labels

def get_alignment_shapes_tables(sdata, 
                    transcripts_qv_thresh=20, 
                    annotate_spatial_elements='cell_boundaries',
                    cell_id_name='cell_id',
                    mask_id_name='cell_labels'):
    # Make cell_id to cell_label mapping dictionary
    cell_id_label_dict = dict(zip(sdata['table'].obs[cell_id_name].values, sdata['table'].obs[mask_id_name].values))
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
    
    return table, transcripts, shapes

def generate_zstack(zstack_path, zstack_masks_path, zstack_size=None, zstack_ind=None, zstack_channels=None, alignment_folder=None):
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

    if alignment_folder:
        zstack_save_path = alignment_folder / zstack_save_name
        if zstack_save_path.exists():
            print(f"Zstack already generated at: {zstack_save_path}")
            return zstack_save_path
        else:
            alignment_folder.mkdir(parents=True, exist_ok=True)
            zstack_masks = get_zstack(zstacks_masks_dict, zstack_size=zstack_size)
            zstack_sdata = get_zstack_sdata(zstack_info, zstack_masks=zstack_masks)
            for table_name, table in zstack_sdata.tables.items():
                table.uns = {'zstack_name': zstack_info['zstack_name']}
            zstack_sdata.write(zstack_save_path)
            print(f"Zstack saved at: {zstack_save_path}")
            return zstack_save_path
    else:
        zstack_masks = get_zstack(zstacks_masks_dict, zstack_size=zstack_size)
        zstack_sdata = get_zstack_sdata(zstack_info, zstack_masks=zstack_masks)
        for table_name, table in zstack_sdata.tables.items():
            table.uns = {'zstack_name': zstack_info['zstack_name']}
        return zstack_sdata
