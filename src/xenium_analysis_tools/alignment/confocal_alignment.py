from xenium_analysis_tools.utils.sd_utils import add_micron_coord_sys
from spatialdata.transformations import Scale, Identity, Sequence, set_transformation, get_transformation
from spatialdata.models import Image3DModel
import spatialdata as sd
import xarray as xr
import dask.array as da
import zarr
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

def get_confocal_image_sizes(img_name, cf_raw_path, overlap=0.1):
    confocal_notes = pd.read_csv(cf_raw_path / 'notes.csv')
    capture_name = confocal_notes.loc[confocal_notes['note'] == img_name, 'capture names'].values[0]
    sldy_dir = list(cf_raw_path.glob("*.dir"))[0]
    imgdir_path = sldy_dir / f"{capture_name}.imgdir"
    sample_npy = list(imgdir_path.glob("ImageData_*.npy"))[0]
    shape = np.load(sample_npy, mmap_mode='r').shape # (Z, Y, X)
    yaml_path = imgdir_path / 'StagePositionData.yaml'

    with open(yaml_path, 'r') as f:
        stage_data = yaml.safe_load(f)
    coords = np.array(stage_data['StructArrayValues']).reshape(-1, 3)
    step_x = np.abs(np.diff(coords[:, 0]))
    step_x = np.median(step_x[step_x > 1.0])
    phys_x = step_x / (shape[2] * (1 - overlap))

    return {
        'sizeZ': shape[0],
        'sizeY': shape[1],
        'sizeX': shape[2],
        'sizeC': 1,
        'physical_pixel_size_x': phys_x,
        'physical_pixel_size_y': phys_x,
        'physical_pixel_size_z': 1.0
    }

def generate_confocal_sdata(zarr_path, raw_confocal_path=None, select_scales=['0','1','2','3'], 
                           chunk_size=(1, 64, 512, 512)):
    cf_name = zarr_path.stem
    cf_dt = create_datatree_from_zarr(zarr_path, chan_name=cf_name, 
                                     select_scales=select_scales, chunk_size=chunk_size)

    cf_sdata = sd.SpatialData(
            images={cf_name: cf_dt}
    )

    if raw_confocal_path:
        cf_sizes = get_confocal_image_sizes(cf_name, raw_confocal_path)
        cf_sdata[cf_name].attrs.update(cf_sizes)
        cf_sdata = add_micron_coord_sys(cf_sdata, pixel_size=[cf_sizes['physical_pixel_size_y'], cf_sizes['physical_pixel_size_x']], z_step=cf_sizes['physical_pixel_size_z'])

    return cf_sdata

def create_datatree_from_zarr(zarr_path, chan_name='chan', select_scales=['0','1','2','3'], 
                             chunk_size=(1, 64, 512, 512)):
    root = zarr.open_group(zarr_path, mode='r')
    data_tree_obj = xr.DataTree()
    
    # Pre-calculate which scales to process
    available_scales = sorted(list(root.keys()))
    if select_scales is not None:
        # Always include scale0 for size calculations, filter later
        scales_to_process = ['0'] + [s for s in select_scales if s != '0']
    else:
        scales_to_process = available_scales
    
    scale0_shape = None
    
    for scale_level in scales_to_process:
        if scale_level not in available_scales:
            continue
            
        if scale_level != '0':
            if select_scales is not None and scale_level not in select_scales:
                continue
                
        print(f"Adding scale level: {scale_level}")
        
        # Optimize zarr loading with explicit chunking
        level_array = da.from_zarr(str(zarr_path / scale_level), chunks=chunk_size[1:])  # Skip c dim
        level_array = da.expand_dims(level_array, axis=0)  # Add c dimension
        
        # Store scale0 shape for later calculations
        if scale_level == '0':
            scale0_shape = level_array.shape
        
        # Convert to xarray DataArray with optimized chunking
        data_array = xr.DataArray(
            level_array,
            dims=['c', 'z', 'y', 'x']
        )
        
        parsed_array = Image3DModel.parse(
            data_array,
            dims=['c', 'z', 'y', 'x'],
            c_coords=[chan_name],  # Use list for consistency
            chunks=chunk_size,  # Explicit chunking
        )
        
        scale_key = f'scale{scale_level}'
        data_tree_obj[scale_key] = xr.Dataset({'image': parsed_array})
        
        # Set up transformations more efficiently
        if scale_level != '0' and scale0_shape is not None:
            current_shape = level_array.shape
            scale_factors = np.array(scale0_shape) / np.array(current_shape)
            scale_transform = Scale(scale_factors, axes=parsed_array.dims)
            sequence = Sequence([scale_transform, Identity()])
            set_transformation(parsed_array, sequence, to_coordinate_system="global")
        else:
            set_transformation(parsed_array, Identity(), to_coordinate_system="global")

    # Handle scale removal and renaming more efficiently
    if select_scales is not None and '0' not in select_scales:
        print("Removing scale0 from data tree as it's not in select_scales")
        del data_tree_obj['scale0']
    
    # Optimize renaming logic
    if select_scales is not None:
        orig_keys = list(data_tree_obj.keys())
        desired_keys = [f"scale{idx}" for idx in range(len(orig_keys))]
        
        if orig_keys != desired_keys:
            print("Renaming scales to ensure sequential order")
            # Use dict comprehension for faster renaming
            renamed_datasets = {}
            for n, old_key in enumerate(orig_keys):
                new_key = f"scale{n}"
                print(f"  Renaming {old_key} -> {new_key}")
                dataset = data_tree_obj[old_key].copy(deep=False)  # Shallow copy
                dataset.image.attrs['original_scale_level'] = old_key
                renamed_datasets[new_key] = dataset
            
            # Rebuild tree from dict
            data_tree_obj = xr.DataTree.from_dict(renamed_datasets)
            
    return data_tree_obj

def get_confocal_sdata(confocal_zarr_path, raw_confocal_path, select_scales=['0','1','2','3'], 
                       chunk_size=(1, 64, 512, 512)):
    sdatas = []
    
    # Use pathlib for more efficient path operations
    confocal_path = Path(confocal_zarr_path)
    zarr_files = [f.stem for f in confocal_path.glob('*.zarr')]
    
    if 'deep' in zarr_files:
        print("Generating sdata for deep confocal...")
        deep_sdata = generate_confocal_sdata(
            zarr_path=confocal_path / 'deep.zarr',
            raw_confocal_path=raw_confocal_path,
            select_scales=select_scales,
            chunk_size=chunk_size
        )
        sdatas.append(deep_sdata)
        
    if 'surface' in zarr_files:
        print("Generating sdata for surface confocal...")
        surface_sdata = generate_confocal_sdata(
            zarr_path=confocal_path / 'surface.zarr',
            raw_confocal_path=raw_confocal_path,
            select_scales=select_scales,
            chunk_size=chunk_size
        )
        sdatas.append(surface_sdata)
        
    confocal_sdata = sd.concatenate(sdatas, merge_coord_systems=True)
    return confocal_sdata