
from spatialdata.transformations import Scale, Identity, Sequence, set_transformation, get_transformation
from spatialdata.models import Image3DModel
import spatialdata as sd
import xarray as xr
import dask.array as da
import zarr
from xenium_analysis_tools.utils.sd_utils import add_micron_coord_sys
import pandas as pd
import numpy as np
# from bioio_sldy import Reader
import pandas as pd
import yaml
import os

def get_confocal_image_sizes(img_name, cf_raw_path, overlap=0.1):
    confocal_notes = pd.read_csv(cf_raw_path / 'notes.csv')
    capture_name = confocal_notes.loc[confocal_notes['note'] == img_name, 'capture names'].values[0]
    imgdir_path = cf_raw_path / f"{capture_name}.imgdir"
    if not imgdir_path.exists():
        imgdir_path = list(cf_raw_path.glob(f"*{capture_name}*.imgdir"))[0]

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
        'sizeC': 1, # Confocal captures are usually single channel per dir
        'physical_pixel_size_x': phys_x,
        'physical_pixel_size_y': phys_x, # Typically square
        'physical_pixel_size_z': 1.0     # Placeholder if not in YAML
    }

# def get_confocal_image_sizes(img_name, cf_raw_path):
#     sld_path = list(cf_raw_path.glob('*.sldy'))[0]
#     r = Reader(str(sld_path))
#     confocal_notes = pd.read_csv(cf_raw_path / 'notes.csv')
#     capture_name = confocal_notes.loc[confocal_notes['note']==img_name,'capture names'].values[0]
#     img_reader = None
#     for i in range(len(r._images)):
#         if r._images[i].image_directory.stem==capture_name:
#             img_reader = r._images[i]
#             break
#     if img_reader is None:
#         raise ValueError(f"Could not find capture name {capture_name} in confocal sldy file")

#     reader_attrs = list(img_reader.__dict__.keys())
#     size_attrs = [attr for attr in reader_attrs if 'size' in attr]
#     size_attrs = {attr: getattr(img_reader, attr) for attr in size_attrs}
#     return size_attrs

def generate_confocal_sdata(zarr_path, raw_confocal_path=None):
    cf_name = zarr_path.stem
    cf_dt = create_datatree_from_zarr(zarr_path, chan_name=cf_name)

    cf_sdata = sd.SpatialData(
            images={cf_name: cf_dt}
    )

    if raw_confocal_path:
        # cf_sizes = get_confocal_image_sizes(cf_name, raw_confocal_path)
        cf_sizes = get_confocal_image_sizes(cf_name, raw_confocal_path)
        cf_sdata[cf_name].attrs.update(cf_sizes)
        if not cf_sizes['physical_pixel_size_x']==cf_sizes['physical_pixel_size_y']:
            raise ValueError(f"Confocal pixel sizes in X and Y do not match for {cf_name}!")
        else:
            cf_sdata = add_micron_coord_sys(cf_sdata, pixel_size=cf_sizes['physical_pixel_size_x'])

    return cf_sdata

def create_datatree_from_zarr(zarr_path, chan_name='chan'):
    root = zarr.open_group(zarr_path, mode='r')
    data_tree_obj = xr.DataTree()

    for scale_level in sorted(list(root.keys())):
        # Load the image data at this scale level
        level_array = da.from_zarr(str(zarr_path / scale_level))
        level_array = np.expand_dims(level_array, axis=0)  # Add c dimension
        # Convert to xarray DataArray
        data_array = xr.DataArray(
                level_array,
                dims=['c', 'z', 'y', 'x']
            )
        parsed_array = Image3DModel.parse(
                            data_array,
                            dims=['c', 'z', 'y', 'x'],
                            c_coords=chan_name,
                            chunks='auto', 
        )
        scale_key = f'scale{scale_level}'
        data_tree_obj[scale_key] = xr.Dataset({'image': parsed_array})
        # Set up scale transformation for non-zero scales
        if scale_key != 'scale0':
            scale_factors = np.array(data_tree_obj[f'scale0'].image.shape) / np.array(data_tree_obj[scale_key].image.shape)
            scale_transform = Scale(scale_factors, axes=data_tree_obj[scale_key].image.dims)
            sequence = Sequence([scale_transform, Identity()])
            set_transformation(data_tree_obj[scale_key].image, sequence, to_coordinate_system="global")
        else:
            set_transformation(data_tree_obj[scale_key].image, Identity(), to_coordinate_system="global")
    return data_tree_obj



# Coped from capsule 4 to keep track of overlap blending code
def generate_fused_confocal_images(data_asset, overlap=0.1, img_layers=6):
    notes = pd.read_csv(os.path.join(data_asset, 'notes.csv'))
    notes=notes[notes['note']!='qc'].reset_index(drop=True)
    today_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    processed_dir = os.path.join(data_asset.replace('/data/','/scratch/')+'_processed_'+today_str)
    for idx, row in notes.iterrows():
        image_dir = os.path.join(data_asset,[d for d in os.listdir(data_asset) if d.endswith('.dir')][0] ,row['capture names']+'.imgdir')
        zarr_filename = os.path.join(processed_dir, row['note'] + '.zarr') 
        tif_filename = zarr_filename.replace('.zarr','.tif')
        if os.path.exists(zarr_filename):
            print('skipping', data_asset, row['note'])
            continue
        
        try:
            yaml_file = os.path.join(image_dir,'StagePositionData.yaml')
            StagePositionData = open_yaml(yaml_file)
            positions = (np.array(StagePositionData['StructArrayValues'])/100).astype(int)
        except:
            print('no position data for', data_asset, row['note'])
            continue
        for p in range(3):
            positions[p::3] = rankdata(-positions[p::3], method='dense')-1
            
        locs = positions.reshape(-1,3)[:,:-1][:,::-1].astype(int)
        n_rows = np.max(locs[:,0])+1
        n_cols = np.max(locs[:,1])+1
        
        if n_rows*n_cols == 1:
            print('no fusion needed for', data_asset, row['note'])
            file = os.path.join(image_dir, [f for f in os.listdir(image_dir) if f.endswith('.npy') and f.startswith('ImageData')][0])
            image = np.load(file)    

        else:   
            files = [f for f in os.listdir(image_dir) if f.endswith('.npy') and f.startswith('ImageData')]
            files = np.sort(files)
            image_ = np.load(os.path.join(image_dir, files[0]))
            n_tiles = len(locs)
            x_size = image_.shape[1]
            y_size = image_.shape[2]
            z_size = image_.shape[0]
            image = np.zeros((z_size,int(x_size*(n_rows-overlap*(n_rows-1))), int(y_size*(n_cols-overlap*(n_cols-1)))),dtype=np.uint16)
            print('fusing', data_asset, row['note'])
            for ind_tile in range(n_tiles):
                tile_ = np.load(os.path.join(image_dir, files[ind_tile]))
                for z in tqdm(range(z_size), desc=f'Fusing tile {ind_tile+1}/{n_tiles}'):

                    tile = tile_[z]
                    if locs[ind_tile][0] == 0:
                        x_start = 0
                        x_end = x_size  
                    else:
                        x_start = int(x_size*(locs[ind_tile][0]*(1-overlap)))
                        x_end = x_start+x_size
                        
                        
                    if locs[ind_tile][0] < np.max(locs,axis=0)[0]:
                        tile[-int(overlap*x_size):, :] = tile[-int(overlap*x_size):, :]*(1-sigmoid_vector(int(overlap*x_size), y_size))
                    
                    if locs[ind_tile][0] > 0:
                        tile[:int(overlap*x_size), :] = tile[:int(overlap*x_size), :]*sigmoid_vector(int(overlap*x_size), y_size)
                    
                    if locs[ind_tile][1] == 0:
                        y_start = 0
                        y_end = y_size
                    else:
                        y_start = int(y_size*(locs[ind_tile][1]*(1-overlap)))
                        y_end = y_start+y_size
                        
                        
                    if locs[ind_tile][1] < np.max(locs,axis=0)[1]:
                        tile[:, -int(overlap*y_size):] = tile[:, -int(overlap*y_size):]*(1-sigmoid_vector(int(overlap*y_size),x_size).T)
                        
                    if locs[ind_tile][1] > 0:
                        tile[:, :int(overlap*y_size)] = tile[:, :int(overlap*y_size)]*sigmoid_vector(int(overlap*y_size),x_size).T

                    image[z,x_start:x_end, y_start:y_end] += tile
        os.makedirs(processed_dir, exist_ok=True)
        tiff.imwrite(tif_filename, image)
        store = zarr.storage.LocalStore(zarr_filename)
        root = zarr.group(store=store, zarr_format=2)

        # Define a scaler for creating the image pyramid
        scaler = Scaler(method='nearest', max_layer=img_layers)  # Create 4 levels in the pyramid
        # Write the image data with pyramid
        write_image(image, root, scaler=scaler, axes = 'zyx')