import spatialdata as sd
from xenium_analysis_tools.utils.sd_utils import add_micron_coord_sys
from spatialdata.models import Image3DModel, Labels3DModel
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
            'stack_name': stack_folder.name,
            'stack_size': _extract_stack_size(stack_folder.name),
            'stack_channels': [ch for ch in channels if ch in stack_folder.name.lower()],
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

def _extract_stack_size(stack_name):
    """Extract width x height x depth from stack name."""
    size_pattern = re.search(r'(\d+)x(\d+)x(\d+)', stack_name)
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

def get_zstack(zstacks_dict, zstack_ind=None, zstack_name=None, stack_size=None, channels=None):    
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
        if channels is not None:
            channel_matches = [
                i for i in matches 
                if set(zstacks_dict[i]['stack_channels']) == set(channels)
            ]
            if len(channel_matches) == 1:
                return zstacks_dict[channel_matches[0]]
            elif len(channel_matches) > 1:
                raise ValueError(f"Multiple z-stacks found with {criterion_name} {criterion_value} and channels {channels}. Found {len(channel_matches)} matches.")
            else:
                raise ValueError(f"No z-stack found with {criterion_name} {criterion_value} and channels {channels}.")
        
        raise ValueError(f"Multiple z-stacks found with {criterion_name} {criterion_value}. Found {len(matches)} matches. Consider specifying channels parameter.")
    
    if zstack_name is not None:
        return _find_matches(
            lambda stack: stack['stack_name'] == zstack_name,
            "Z-stack name", zstack_name
        )
    
    if stack_size is not None:
        return _find_matches(
            lambda stack: (
                stack['stack_size']['width'] == stack_size['width'] and
                stack['stack_size']['height'] == stack_size['height'] and
                stack['stack_size']['depth'] == stack_size['depth']
            ),
            "Stack size", stack_size
        )
    
    raise ValueError("Either zstack_ind, zstack_name, or stack_size must be provided.")

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
        "sdata_path": data_root / f'{dataset_config["xenium_name"]}_processed',
        "confocal_path": data_root / dataset_config["confocal_name"],
        "zstack_path": data_root / dataset_config["zstack_name"],
        "zstack_masks": data_root / dataset_config["zstack_masks_name"]
    }
    
    return paths

def get_zstack_sdata(stack, zstack_masks=None, use_shared_coords=True):
    # Create the z-stack image array
    num_channels = len(stack['stack_channels'])
    chans = []
    if num_channels > 1:
        for ch_ind in range(num_channels):
            chan_array = create_zstack_array(tif_path=stack['channel_tifs'][ch_ind]['chan_tif_path'], 
                    fov_x_um=stack['stack_size']['width'], 
                    fov_y_um=stack['stack_size']['height'], 
                    fov_z_um=stack['stack_size']['depth'])
            chans.append(chan_array)
        zstack_img = xr.concat(chans, dim='c')
        zstack_img['c'] = stack['stack_channels']
    else:
        zstack_img = create_zstack_array(tif_path=stack['channel_tifs'][0]['chan_tif_path'], 
                    fov_x_um=stack['stack_size']['width'], 
                    fov_y_um=stack['stack_size']['height'], 
                    fov_z_um=stack['stack_size']['depth'])
        zstack_img['c'] = stack['stack_channels']

    if use_shared_coords:   
        reg_json_path = stack['metadata_jsons']['registration']
        with open(reg_json_path) as f:
            reg_json = json.load(f) 
        if 'z_steps' in reg_json.keys() and len(reg_json['z_steps'])==zstack_img.sizes['z']:
            print("Using shared z coordinates for images")
            zstack_img.coords['z'] = reg_json['z_steps']

    # Parse into Image3DModel
    zstack_img = Image3DModel.parse(
                zstack_img,
                dims=['c', 'z', 'y', 'x'],
                c_coords=stack['stack_channels'],
                chunks='auto',
            )

    # Make the SpatialData object
    zstack_sdata = sd.SpatialData(
            images={'zstack': zstack_img},
        )

    if zstack_masks is not None:
        # Get labels for each channel
        for mask_ind, masks in zstack_masks['channel_tifs'].items():
            channel_name = zstack_masks['stack_channels'][mask_ind]
            zstack_label = create_zstack_array(tif_path=masks['chan_tif_path'], 
                        fov_x_um=zstack_masks['stack_size']['width'], 
                        fov_y_um=zstack_masks['stack_size']['height'], 
                        fov_z_um=zstack_masks['stack_size']['depth'],
                        add_chan=False)

            if use_shared_coords:   
                if 'z_steps' in reg_json.keys() and len(reg_json['z_steps'])==zstack_label.sizes['z']:
                    print("Using shared z coordinates for labels")
                    zstack_label.coords['z'] = reg_json['z_steps']

            zstack_label = Labels3DModel.parse(
                        zstack_label,
                        dims=['z', 'y', 'x'],
                        chunks='auto',
                    )
            zstack_sdata.labels[f"{channel_name}_labels"] = zstack_label

    # Determine pixel sizes
    if zstack_sdata['zstack'].attrs['pixel_size_um_x'] == zstack_sdata['zstack'].attrs['pixel_size_um_y']:
        pixel_size = zstack_sdata['zstack'].attrs['pixel_size_um_x']
    if zstack_sdata['zstack'].attrs['fov_um_z']==zstack_sdata['zstack'].shape[1]:
        z_step_size = 1

    # Add micron coordinate system if not already present
    if 'microns' not in zstack_sdata.coordinate_systems:
        zstack_sdata = add_micron_coord_sys(zstack_sdata, pixel_size=pixel_size, z_step=z_step_size)
    else:
        print("Micron coordinate system already exists")
    return zstack_sdata

