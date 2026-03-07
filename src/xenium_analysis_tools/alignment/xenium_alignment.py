import json
import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import xarray as xr

from xenium_analysis_tools.utils.sd_utils import (
    add_mapped_cells_cols,
    add_micron_coord_sys
)

from pathlib import Path
import spatialdata as sd
from spatialdata.models import (
    Image3DModel, 
    Labels3DModel, 
    Image2DModel, 
    Labels2DModel, 
    PointsModel, 
    TableModel
)
from spatialdata.transformations import (
    Affine,
    Identity,
    MapAxis,
    Scale,
    Sequence,
    Translation,
    BaseTransformation,
    get_transformation,
    set_transformation,
    align_elements_using_landmarks,
    get_transformation_between_landmarks
)

def get_ophys_sdata_filtered(zstack_sdata, confocal_sdata=None, zstack_scale_level=0, confocal_scale_level=3):
    if isinstance(zstack_sdata, str) or isinstance(zstack_sdata, Path):
        zstack_sdata = sd.read_zarr(zstack_sdata)
    # Z-stack
    chans = sd.models.get_channel_names(zstack_sdata['zstack'][f'scale{zstack_scale_level}'].image)
    images = {}
    labels = {}
    for c in chans:
        channel_data = zstack_sdata['zstack'][f'scale{zstack_scale_level}'].image.sel(c=c)
        channel_data = channel_data.expand_dims('c')
        images[c] = Image3DModel.parse(channel_data, c_coords=[c])
    for c in zstack_sdata.labels.keys():
        channel_data = zstack_sdata.labels[c][f'scale{zstack_scale_level}'].image
        labels[c] = Labels3DModel.parse(channel_data)
    czstack = sd.SpatialData(images=images, labels=labels)

    # Confocal
    if confocal_sdata is None:
        return czstack
    images = {}
    labels = {}
    for chan in confocal_sdata.images.keys():
        channel_data = confocal_sdata[chan][f'scale{confocal_scale_level}'].image
        images[f'{chan}_confocal'] = Image3DModel.parse(channel_data, c_coords=[chan])
    for c in confocal_sdata.labels.keys():
        channel_data = confocal_sdata.labels[c][f'scale{confocal_scale_level}'].image
        labels[f'{c}_confocal'] = Labels3DModel.parse(channel_data)
    confocal = sd.SpatialData(images=images, labels=labels)

    sdata = sd.concatenate([czstack, confocal], merge_coordinate_systems_on_name=True)
    return sdata

def _add_dapi_zstack(sdata, chan='dapi_zstack', labels_name='cell_labels', scale_level=2):
    # Transform
    original_element = sdata[chan][f'scale{scale_level}']
    transformations = {}
    for cs in original_element.attrs.get('transform', {}):
        transformations[cs] = original_element.attrs['transform'][cs]

    # Make image element
    image = Image3DModel.parse(sdata[chan][f'scale{scale_level}'].image, c_coords=[chan], transformations=transformations)

    # Make labels element 
    adata = sdata['table'].obs.copy()
    z_levels = range(int(adata['z_level'].min()), int(adata['z_level'].max())+1)
    labels = sdata[labels_name][f'scale{scale_level}'].image
    dapi_zstack_shape = (len(z_levels),) + labels.shape
    dapi_zstack_data = np.zeros(dapi_zstack_shape, dtype=labels.dtype)
    for i, z_lv in enumerate(z_levels):
        z_level_cell_ids = adata[adata['z_level'] == z_lv]['cell_labels'].values.astype(int)
        level_labels = labels.values.copy()
        level_labels[~np.isin(level_labels, z_level_cell_ids)] = 0
        dapi_zstack_data[i] = level_labels

    dapi_zstack_cell_labels = xr.DataArray(
        dapi_zstack_data,
        dims=['z', 'y', 'x'],
        coords={
            'z': z_levels,
            'y': labels.coords['y'],
            'x': labels.coords['x']
        },
        attrs=labels.attrs
    )
    dapi_zstack_cell_labels = Labels3DModel.parse(dapi_zstack_cell_labels)
    dapi_zstack_cell_labels.attrs['transform'] = image.attrs['transform'].copy()
    return image, dapi_zstack_cell_labels

def get_section_sdata_filtered(section_sdata_path, 
                            section_n,
                            chan_order_names = ['dapi', 'boundary', 'rna', 'protein'], 
                            get_labels = ['cell_labels'], 
                            scale_level=2,
                            add_micron_cs=True,
                            include_dapi_zstack=False):
    # Load section sdata
    section_sdata = sd.read_zarr(section_sdata_path) 

    # Add micron coordinate system
    if add_micron_cs:
        section_sdata = add_micron_coord_sys(section_sdata) 

    # Images
    images = {}
    channel_names = sd.models.get_channel_names(section_sdata['morphology_focus'][f'scale{scale_level}'].image)
    original_element = section_sdata['morphology_focus'][f'scale{scale_level}']
    transformations = {}
    for cs in original_element.attrs.get('transform', {}):
        transformations[cs] = original_element.attrs['transform'][cs]
    for i, chan in enumerate(chan_order_names):
        channel_data = section_sdata['morphology_focus'][f'scale{scale_level}'].sel(c=channel_names[i]).image.copy()
        channel_data = channel_data.expand_dims('c')
        images[chan] = Image2DModel.parse(channel_data, c_coords=[chan], transformations=transformations)
    
    # Labels
    labels = {}
    for c in get_labels:
        channel_data = section_sdata.labels[c][f'scale{scale_level}'].image.copy()
        labels[c] = Labels2DModel.parse(channel_data, transformations=transformations)

    # Points
    ids_labels_dict = section_sdata['table'].obs[['cell_id','cell_labels']]
    transcripts = section_sdata['transcripts'].copy()
    transcripts['cell_labels'] = transcripts['cell_id'].map(ids_labels_dict.set_index('cell_id')['cell_labels'])
    transcripts['section'] = section_n
    transcripts = PointsModel.parse(transcripts)

    # Table
    table = section_sdata['table'].copy()
    table.obs['section'] = section_n
    table.obs['x_centroid_microns'] = table.obsm['spatial'][:,0]
    table.obs['y_centroid_microns'] = table.obsm['spatial'][:,1]
    table.obs['region'] = 'cell_labels' 
    table.obs['region'] = table.obs['region'].astype('category')
    table.uns['spatialdata_attrs'] = {
        'region': 'cell_labels',
        'instance_key': 'cell_labels',
        'region_key': 'region'
    }
    table = TableModel.parse(table)

    # Combine into SpatialData object
    x_section = sd.SpatialData(images=images, labels=labels, points={'transcripts': transcripts}, tables={'table': table})

    if include_dapi_zstack:
        dapi_image, dapi_zstack_labels = _add_dapi_zstack(section_sdata)
        x_section.images['dapi_zstack'] = dapi_image
        x_section.labels['dapi_zstack_labels'] = dapi_zstack_labels

    return x_section

def get_biwarp_params(bigwarp_json_path):
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
    
def get_section_landmarks(bigwarp_project_path, landmarks_path, dims_order=['x','y','z']):
    bigwarp_params = get_biwarp_params(bigwarp_project_path)
    landmarks = pd.read_csv(landmarks_path, header=None)
    if bigwarp_params['moving_image']=='czstack':
        # Flatten the lists using + operator to concatenate them
        landmarks.columns = ['landmark_name', 'active'] + [f'czstack_{dim}' for dim in dims_order] + [f'xenium_{dim}' for dim in dims_order]
    else:
        landmarks.columns = ['landmark_name', 'active'] + [f'xenium_{dim}' for dim in dims_order] + [f'czstack_{dim}' for dim in dims_order]

    return landmarks, bigwarp_params

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

def invert_xenium_y_landmarks(landmarks, landmarked_image_path):
    with tifffile.TiffFile(landmarked_image_path) as tif:
        landmarked_image_shape = tif.pages[0].shape
    full_y_size = landmarked_image_shape[0]
    landmarks['xenium_y'] = full_y_size - landmarks['xenium_y']
    return landmarks

def fix_landmarks_cropped_section(landmarks, sdata, section_n=None):
    if section_n is None:
        section_n = np.unique(sdata['table'].obs['section'])[0]
    
    # Bounding box used for getting the section from multisection slide image
    section_bbox = sdata['table'].uns['sections_bboxes'][str(section_n)]

    # Scale level to pixel full resolution
    scale_factor_y = 1
    scale_factor_x = 1
    level_transform = get_transformation(sdata[list(sdata.images.keys())[0]], to_coordinate_system='global')
    if hasattr(level_transform, 'transformations'):
        if hasattr(level_transform.transformations[0],'scale'):
            for i, ax in enumerate(level_transform.transformations[0].axes):
                if ax == 'y':
                    scale_factor_y = level_transform.transformations[0].scale[i]
                elif ax == 'x':
                    scale_factor_x = level_transform.transformations[0].scale[i]

    # Adjust landmarks for cropping and scaling
    landmarks['xenium_x'] = landmarks['xenium_x'] - (section_bbox['x_min'] / scale_factor_x)
    landmarks['xenium_y'] = landmarks['xenium_y'] - (section_bbox['y_min'] / scale_factor_y)

    return landmarks

def validate_landmarks(section_image, zstack_image, section_landmarks, section_n):
    section_landmarks['landmark_label'] = section_landmarks['landmark_name'].str.split('-').str[1]
    fig,ax = plt.subplots(1,3,figsize=(10,10))
    ax[0].imshow(section_image)
    # Keep both points and labels
    ax[0].scatter(section_landmarks['xenium_x'], section_landmarks['xenium_y'], c='r', s=10)
    for i, row in section_landmarks.iterrows():
        ax[0].text(row['xenium_x'], row['xenium_y'] - 20, row['landmark_label'], 
                color='red', fontsize=8, ha='center')
    ax[0].set_title(f'Xenium Section {section_n}')

    landmarks_bbox = {'ymin': int(section_landmarks['xenium_y'].min()) - 50,
                     'ymax': int(section_landmarks['xenium_y'].max()) + 50, 
                     'xmin': int(section_landmarks['xenium_x'].min()) - 50, 
                     'xmax': int(section_landmarks['xenium_x'].max()) + 50}
    ax[1].imshow(section_image)
    ax[1].set_xlim(landmarks_bbox['xmin'], landmarks_bbox['xmax'])
    ax[1].set_ylim(landmarks_bbox['ymin'], landmarks_bbox['ymax'])
    # Keep both points and labels
    ax[1].scatter(section_landmarks['xenium_x'], section_landmarks['xenium_y'], c='r', s=10)
    for i, row in section_landmarks.iterrows():
        ax[1].text(row['xenium_x'], row['xenium_y'] - 20, row['landmark_label'], 
                color='red', fontsize=8, ha='center')
    ax[1].set_title(f'Zoomed to Landmarks')

    zstack_image = zstack_image[int(section_landmarks['czstack_z'].min()):int(section_landmarks['czstack_z'].max())+1]
    zstack_image = np.max(zstack_image, axis=0)
    ax[2].imshow(zstack_image)
    ax[2].scatter(section_landmarks['czstack_x'], section_landmarks['czstack_y'], c='r', s=10)
    for i, row in section_landmarks.iterrows():
        ax[2].text(row['czstack_x'], row['czstack_y'] - 20, row['landmark_label'], 
                color='red', fontsize=8, ha='center')
    ax[2].set_title(f'Cortical z-stack \n(max projection across covered z-planes)')
    plt.tight_layout()
    return fig

def add_xenium_landmarks_to_sdata(section_landmarks, sdata):
    xenium_landmarks = section_landmarks.copy()
    xenium_landmarks.rename(columns={'xenium_x': 'x', 'xenium_y': 'y', 'xenium_z': 'z'}, inplace=True)
    xenium_landmarks = PointsModel.parse(xenium_landmarks)
    xenium_landmarks.attrs['transform'] = sdata[list(sdata.images.keys())[0]].attrs['transform'].copy()
    sdata.points['landmarks'] = xenium_landmarks
    return sdata

def simplify_coord_systems(sdata, section_n):
    for el_name in ['dapi','boundary','rna','protein','cell_labels', 'landmarks', 'dapi_zstack', 'dapi_zstack_labels']:
        if el_name not in sdata:
            continue
        tf_to_full_res = get_transformation(sdata[el_name], to_coordinate_system='global')
        tf_to_microns = get_transformation(sdata[el_name], to_coordinate_system='microns')
        if hasattr(tf_to_full_res, 'transformations'):
            scale_to_full_res = tf_to_full_res.transformations[0]
        else:
            scale_to_full_res = tf_to_full_res
        if hasattr(tf_to_microns, 'transformations'):
            scale_to_microns = tf_to_microns.transformations[1]
        else:
            scale_to_microns = tf_to_microns
        set_transformation(sdata[el_name], Identity(), to_coordinate_system=f'global')
        set_transformation(sdata[el_name], scale_to_full_res, to_coordinate_system='full_res')
        set_transformation(sdata[el_name], Scale(scale_to_full_res.scale*scale_to_microns.scale, scale_to_full_res.axes), to_coordinate_system='microns')

    set_transformation(sdata['transcripts'], get_transformation(sdata['transcripts'], to_coordinate_system='global'), to_coordinate_system='full_res')
    set_transformation(sdata['transcripts'], get_transformation(sdata['boundary'], to_coordinate_system='microns').inverse(), to_coordinate_system='global')
    sd.SpatialData.rename_coordinate_systems(sdata, {'global': f'section_{section_n}', 'microns': f'section_{section_n}_microns', 'full_res': f'section_{section_n}_full_res'})
    return sdata

# import spatialdata as sd
# from xenium_analysis_tools.utils.sd_utils import add_micron_coord_sys
# from spatialdata.models import Image2DModel, Image3DModel, Labels3DModel, Labels2DModel, PointsModel, ShapesModel, TableModel
# from spatialdata.transformations import get_transformation, set_transformation
# import anndata as ad
# from spatialdata import get_element_instances
# from pathlib import Path
# import pandas as pd
# import numpy as np
# import xarray as xr
# import tifffile
# import json
# import re
# from xenium_analysis_tools.utils.sd_utils import (
#     add_mapped_cells_cols,
#     get_transcripts_bboxes
# )

# def create_zstack_array(tif_path,
#                         add_chan=True,
#                         fov_z_um=450.0,
#                         fov_x_um=400.0, 
#                         fov_y_um=400.0):

#     data = tifffile.imread(tif_path)
#     pixels_z, pixels_y, pixels_x = data.shape
    
#     # Pixel sizes
#     pixel_size_z = fov_z_um / pixels_z 
#     pixel_size_y = fov_y_um / pixels_y 
#     pixel_size_x = fov_x_um / pixels_x 
    
#     # Create coordinate arrays with proper spacing
#     z_coords = np.arange(pixels_z) * pixel_size_z
#     y_coords = np.arange(pixels_y) * pixel_size_y
#     x_coords = np.arange(pixels_x) * pixel_size_x
#     coords = {"z": z_coords,
#                 "y": y_coords, 
#                 "x": x_coords}

#     if add_chan:
#         data = np.expand_dims(data, axis=0)
#         coords["c"] = np.arange(data.shape[0])
#         dims = ("c", "z", "y", "x")
#     else:
#         dims = ("z", "y", "x")

#     # Create xarray DataArray with improved metadata
#     da = xr.DataArray(
#         data,
#         coords=coords,
#         dims=dims,
#         attrs={
#             "pixel_size_um_z": pixel_size_z,
#             "pixel_size_um_y": pixel_size_y,
#             "pixel_size_um_x": pixel_size_x,
#             "fov_um_z": fov_z_um,
#             "fov_um_y": fov_y_um,
#             "fov_um_x": fov_x_um,
#             "units": "micrometers",
#         }
#     )
#     return da

# def get_zstacks_dict(zstacks_folder, channels=['gcamp', 'dextran']):
#     zstacks_dict = {}
    
#     # Process only directories
#     stack_dirs = [d for d in zstacks_folder.iterdir() if d.is_dir()]
    
#     for stack_ind, stack_folder in enumerate(stack_dirs):
#         stack_info = {
#             'zstack_name': stack_folder.name,
#             'zstack_size': _extract_zstack_size(stack_folder.name),
#             'zstack_channels': [ch for ch in channels if ch in stack_folder.name.lower()],
#             'metadata_jsons': {'registration': None, 'roi_groups': None, 'scanimage': None},
#             'channel_tifs': {}
#         }
        
#         # Process files in stack folder
#         chan_ind = 0
#         for item in sorted(stack_folder.iterdir()):
#             if item.is_file() and item.suffix.lower() == '.json':
#                 # Categorize JSON metadata files
#                 json_type = _categorize_json_file(item.name.lower())
#                 if json_type:
#                     stack_info['metadata_jsons'][json_type] = item
                    
#             elif item.is_dir() and 'channel' in item.name.lower():
#                 # Process channel directories
#                 tif_files = [f for f in item.iterdir() if f.suffix.lower() == '.tif']
#                 stack_info['channel_tifs'][chan_ind] = {
#                     'chan_name': item.name,
#                     'chan_tif_path': tif_files[0] if len(tif_files) == 1 else tif_files
#                 }
#                 chan_ind += 1
        
#         zstacks_dict[stack_ind] = stack_info
    
#     return zstacks_dict

# def _extract_zstack_size(zstack_name):
#     """Extract width x height x depth from stack name."""
#     size_pattern = re.search(r'(\d+)x(\d+)x(\d+)', zstack_name)
#     if size_pattern:
#         width, height, depth = map(int, size_pattern.groups())
#         return {"width": width, "height": height, "depth": depth}
#     return {"width": None, "height": None, "depth": None}

# def _categorize_json_file(filename_lower):
#     """Categorize JSON file by its name."""
#     if 'registration' in filename_lower:
#         return 'registration'
#     elif 'roi_groups' in filename_lower:
#         return 'roi_groups' 
#     elif 'scanimage' in filename_lower:
#         return 'scanimage'
#     return None

# def get_zstack(zstacks_dict, zstack_ind=None, zstack_name=None, zstack_size=None, zstack_channels=None):    
#     if zstack_ind is not None:
#         if zstack_ind not in zstacks_dict:
#             raise ValueError(f"Z-stack index {zstack_ind} not found in zstacks_dict.")
#         return zstacks_dict[zstack_ind]
    
#     # Helper function to find matches
#     def _find_matches(criterion_func, criterion_name, criterion_value):
#         matches = [i for i, stack in zstacks_dict.items() if criterion_func(stack)]
        
#         if not matches:
#             raise ValueError(f"{criterion_name} {criterion_value} not found in zstacks_dict.")
        
#         if len(matches) == 1:
#             return zstacks_dict[matches[0]]
            
#         # Handle multiple matches with optional channel filtering
#         if zstack_channels is not None:
#             channel_matches = [
#                 i for i in matches 
#                 if set(zstacks_dict[i]['zstack_channels']) == set(zstack_channels)
#             ]
#             if len(channel_matches) == 1:
#                 return zstacks_dict[channel_matches[0]]
#             elif len(channel_matches) > 1:
#                 raise ValueError(f"Multiple z-stacks found with {criterion_name} {criterion_value} and channels {zstack_channels}. Found {len(channel_matches)} matches.")
#             else:
#                 raise ValueError(f"No z-stack found with {criterion_name} {criterion_value} and channels {zstack_channels}.")
#         raise ValueError(f"Multiple z-stacks found with {criterion_name} {criterion_value}. Found {len(matches)} matches. Consider specifying channels parameter.")
    
#     if zstack_name is not None:
#         return _find_matches(
#             lambda stack: stack['zstack_name'] == zstack_name,
#             "Z-stack name", zstack_name
#         )
    
#     if zstack_size is not None:
#         return _find_matches(
#             lambda stack: (
#                 stack['zstack_size']['width'] == zstack_size['width'] and
#                 stack['zstack_size']['height'] == zstack_size['height'] and
#                 stack['zstack_size']['depth'] == zstack_size['depth']
#             ),
#             "Stack size", zstack_size
#         )
    
#     raise ValueError("Either zstack_ind, zstack_name, or zstack_size must be provided.")

# def get_label_params(label_obj, id_name='cell'):
#     from skimage.measure import regionprops
#     labels = label_obj.values
#     props = regionprops(labels)
#     data = [
#         {f'{id_name}_id': p.label, 
#         'z': p.centroid[0] if len(p.centroid)==3 else None,
#         'y': p.centroid[1] if len(p.centroid)==3 else p.centroid[0],
#         'x': p.centroid[2] if len(p.centroid)==3 else p.centroid[1],
#         'area': p.area,
#         'bbox': p.bbox}
#         for p in props
#     ]
#     df = pd.DataFrame(data)
#     return df

# def get_zstack_sdata(stack, zstack_masks=None):
#     # Create the z-stack image array
#     chan_arrays = {}
#     for ch_ind, chan_name in enumerate(stack['zstack_channels']):
#         chan_img = create_zstack_array(tif_path=stack['channel_tifs'][ch_ind]['chan_tif_path'], 
#                     fov_x_um=stack['zstack_size']['width'], 
#                     fov_y_um=stack['zstack_size']['height'], 
#                     fov_z_um=stack['zstack_size']['depth'])
#         chan_img = Image3DModel.parse(
#                 chan_img,
#                 dims=['c', 'z', 'y', 'x'],
#                 c_coords=chan_name,
#                 chunks='auto',
#             )
#         chan_arrays[chan_name] = chan_img

#     if zstack_masks is not None:
#         zstack_labels = {}
#         # Get labels for each channel
#         for mask_ind, masks in zstack_masks['channel_tifs'].items():
#             channel_name = zstack_masks['zstack_channels'][mask_ind]
#             zstack_label = create_zstack_array(tif_path=masks['chan_tif_path'], 
#                         fov_x_um=zstack_masks['zstack_size']['width'], 
#                         fov_y_um=zstack_masks['zstack_size']['height'], 
#                         fov_z_um=zstack_masks['zstack_size']['depth'],
#                         add_chan=False)

#             zstack_label = Labels3DModel.parse(
#                         zstack_label,
#                         dims=['z', 'y', 'x'],
#                         chunks='auto',
#                     )
#             zstack_labels[f"{channel_name}_labels"] = zstack_label
    
#         tables = {}
#         for label_name, labels_obj in zstack_labels.items():
#             chan_name = label_name.replace('_labels','')
#             label_type_id = f'{chan_name}_id'
#             chan_label_ids = get_element_instances(labels_obj).values
#             obs = pd.DataFrame(chan_label_ids, columns=[label_type_id])
#             cells_df = get_label_params(labels_obj, id_name=chan_name)
#             cells_df['region'] = label_name
#             obs = obs.merge(cells_df, left_on=label_type_id, right_on=chan_name+'_id', how='left')
#             table = ad.AnnData(obs=obs, obsm={'spatial': obs[['z','y','x']].values})
#             table = TableModel.parse(table, region=label_name, region_key='region', instance_key=label_type_id)
#             tables[f'{chan_name}_cells'] = table

#     # Assemble SpatialData
#     zstack_sdata = sd.SpatialData(
#         images={**chan_arrays},
#         labels={**zstack_labels} if zstack_masks is not None else {},
#         tables={**tables} if zstack_masks is not None else {}
#     )    

#     # Determine pixel sizes
#     zstack_chan = zstack_sdata[stack['zstack_channels'][0]] # Use first channel for pixel size reference
#     if zstack_chan.attrs['pixel_size_um_x'] == zstack_chan.attrs['pixel_size_um_y']:
#         pixel_size = zstack_chan.attrs['pixel_size_um_x']
#     if zstack_chan.attrs['fov_um_z']==zstack_chan.shape[1]:
#         z_step_size = 1

#     # Add micron coordinate system if not already present
#     if 'microns' not in zstack_sdata.coordinate_systems:
#         zstack_sdata = add_micron_coord_sys(zstack_sdata, pixel_size=pixel_size, z_step=z_step_size)
#     else:
#         print("Micron coordinate system already exists")
#     return zstack_sdata

# def get_alignment_spatial_elements(sdata, scale_from_level=2, channel_names=['dapi', 'boundary', 'rna', 'protein'], keep_coord_systems=['global', 'microns']):
#     # Technically should only need to replace morphology focus transforms, but doing for all elements just in case
#     # For elements, get at a specific scale level (if multi-scale) and set transform to global coordinate system
#     # Images
#     # Dapi z-stack
#     dapi_zstack_level = sdata['dapi_zstack'][f'scale{scale_from_level}'].image
#     el_transforms = {}
#     for cs in list(dapi_zstack_level.attrs['transform'].keys()):
#         if cs in keep_coord_systems:
#             el_transforms[cs] = get_transformation(dapi_zstack_level, to_coordinate_system=cs)
#     # dapi_zstack_global_tf = get_transformation(dapi_zstack_level, to_coordinate_system='global')
#     dapi_zstack = Image3DModel.parse(sdata['dapi_zstack'][f'scale{scale_from_level}'].image,
#                                                         dims=['c', 'z', 'y', 'x'],
#                                                         c_coords=['DAPI'],
#                                                         chunks='auto',
#                                                     )
#     for cs, tf in el_transforms.items():
#         set_transformation(dapi_zstack, tf, to_coordinate_system=cs)
#     # set_transformation(dapi_zstack, dapi_zstack_global_tf, to_coordinate_system='global')

#     # Morphology focus channels    
#     mf_chans_level = sdata['morphology_focus'][f'scale{scale_from_level}'].image
#     el_transforms = {}
#     for cs in list(mf_chans_level.attrs['transform'].keys()):
#         if cs in keep_coord_systems:
#             el_transforms[cs] = get_transformation(mf_chans_level, to_coordinate_system=cs)
#     # mf_img_global_tf = get_transformation(mf_chans_level, to_coordinate_system='global')
#     chans_arrays = {}
#     for chan_ind, chan in enumerate(channel_names):
#         chan_img = sdata['morphology_focus'][f'scale{scale_from_level}'].image[chan_ind]
#         chan_img = np.expand_dims(chan_img.data, axis=0)
#         chans_arrays[chan] = Image2DModel.parse(chan_img,
#                                                 dims=['c', 'y', 'x'],
#                                                 c_coords=chan,
#                                                 chunks='auto',
#                                             )
#         for cs, tf in el_transforms.items():    
#             set_transformation(chans_arrays[chan], tf, to_coordinate_system=cs)
#         # set_transformation(chans_arrays[chan], mf_img_global_tf, to_coordinate_system='global')

#     images = {'dapi_zstack': dapi_zstack, **chans_arrays}

#     # Labels
#     cell_labels_level = sdata['cell_labels'][f'scale{scale_from_level}'].image
#     for cs in list(cell_labels_level.attrs['transform'].keys()):
#         if cs in keep_coord_systems:
#             el_transforms[cs] = get_transformation(cell_labels_level, to_coordinate_system=cs)
#     # cell_labels_tf = get_transformation(cell_labels_level, to_coordinate_system='global')
#     cell_labels = Labels2DModel.parse(cell_labels_level, dims=['y', 'x'], chunks='auto')
#     for cs, tf in el_transforms.items():
#         set_transformation(cell_labels, tf, to_coordinate_system=cs)
#     # set_transformation(cell_labels, cell_labels_tf, to_coordinate_system='global')
#     nucleus_labels_level = sdata['nucleus_labels'][f'scale{scale_from_level}'].image
#     nucleus_labels = Labels2DModel.parse(nucleus_labels_level, dims=['y', 'x'], chunks='auto')
#     # set_transformation(nucleus_labels, cell_labels_tf, to_coordinate_system='global')
#     for cs, tf in el_transforms.items():
#         set_transformation(nucleus_labels, tf, to_coordinate_system=cs)

#     labels = {
#         'cell_labels': cell_labels,
#         'nucleus_labels': nucleus_labels,
#     }

#     return images, labels

# def get_alignment_shapes_tables(sdata, 
#                     transcripts_qv_thresh=20, 
#                     annotate_spatial_elements='cell_boundaries',
#                     cell_id_name='cell_id',
#                     mask_id_name='cell_labels',
#                     keep_coord_systems=['global', 'microns']):
#     # Make cell_id to cell_label mapping dictionary
#     cell_id_label_dict = dict(zip(sdata['table'].obs[cell_id_name].values, sdata['table'].obs[mask_id_name].values))
#     # Transcripts transforms
#     points_transforms = {}
#     for cs in list(sdata['transcripts'].attrs['transform'].keys()):
#         if cs in keep_coord_systems:
#             points_transforms[cs] = get_transformation(sdata['transcripts'], to_coordinate_system=cs)
    
#     transcripts = sdata['transcripts'].compute()
#     # Drop transcripts not included in counts
#     transcripts = transcripts[transcripts['qv']>=transcripts_qv_thresh]
#     # Add cell_labels to transcripts based on cell_id
#     transcripts[mask_id_name] = transcripts[cell_id_name].map(cell_id_label_dict).fillna(0).astype('int64')
#     # Annotate spatial elements (e.g., cell_boundaries) with cell_labels
#     sdata[annotate_spatial_elements][cell_id_name] = sdata[annotate_spatial_elements].index.values
#     sdata[annotate_spatial_elements][mask_id_name] = sdata[annotate_spatial_elements][cell_id_name].map(cell_id_label_dict).values
#     sdata[annotate_spatial_elements].set_index(mask_id_name, inplace=True, drop=False)
#     # Update annotation regions
#     table = sdata['table'].copy()
#     table.obs['region'] = annotate_spatial_elements
#     table.obs['region'] = pd.Categorical(table.obs['region'])
#     table.uns['spatialdata_attrs'].update({
#         'region_key': 'region',
#         'region': [annotate_spatial_elements],
#         'instance_key': mask_id_name
#     })

#     # Parse shapes
#     annotated_shape = ShapesModel.parse(sdata[annotate_spatial_elements])
#     shapes = sdata.shapes
#     shapes[annotate_spatial_elements] = annotated_shape
#     # Parse table
#     table = TableModel.parse(table)
#     # Parse transcripts
#     transcripts = PointsModel.parse(transcripts)
#     for cs, tf in points_transforms.items():
#         set_transformation(transcripts, tf, to_coordinate_system=cs)
    
#     return table, transcripts, shapes

# def generate_zstack(zstack_path, zstack_masks_path, zstack_size=None, zstack_ind=None, zstack_channels=None, save_folder=None):
#     # Make the dictionary for the available z-stacks
#     zstacks_dict = get_zstacks_dict(zstack_path)
#     zstacks_masks_dict = get_zstacks_dict(zstack_masks_path)
#     print(f'Number of z-stacks found: {len(zstacks_dict)}\n')
#     for stack_ind, stack_info in zstacks_dict.items():
#         print(f"Stack {stack_ind}: {stack_info['zstack_name']}")
#         zstack_width = stack_info['zstack_size']['width']
#         zstack_height = stack_info['zstack_size']['height']
#         zstack_depth = stack_info['zstack_size']['depth']
#         print(f"  Size: {zstack_width} W x {zstack_height} H x {zstack_depth} D")
#         print(f"  Channels: {stack_info['zstack_channels']}\n")

#     # Select the zstack that matches the criteria
#     if len(zstacks_dict) == 1:
#         zstack_ind = 0
#         zstack_size = None
#         zstack_channels = None
#     zstack_info = get_zstack(zstacks_dict, zstack_ind=zstack_ind, zstack_size=zstack_size, zstack_channels=zstack_channels)
#     zstack_size = zstack_info['zstack_size']
#     zstack_save_name = f"zstack_{zstack_size['width']}x{zstack_size['height']}x{zstack_size['depth']}.zarr"
#     zstack_save_path = save_folder / zstack_save_name
#     zstack_masks = get_zstack(zstacks_masks_dict, zstack_size=zstack_size)
#     zstack_sdata = get_zstack_sdata(zstack_info, zstack_masks=zstack_masks)
#     for table_name, table in zstack_sdata.tables.items():
#         table.uns = {'zstack_name': zstack_info['zstack_name']}
#     zstack_sdata.write(zstack_save_path)
#     print(f"Zstack saved at: {zstack_save_path}")
   
# def generate_section_alignment_data(section_n, 
#                                     paths, 
#                                     save_folder):
#     section_sdata_path = paths['sdata_path'] / f'section_{section_n}.zarr'
#     save_section_path = save_folder / f'xenium_section_{section_n}.zarr'

#     if save_section_path.exists():
#         print(f"Section alignment data already exists: {save_section_path}")
#         xenium_section = sd.read_zarr(save_section_path)
#     else:
#         section_sdata = sd.read_zarr(section_sdata_path)

#         # Add micron coordinate system
#         section_sdata = add_micron_coord_sys(section_sdata)

#         # Add mapped cell type columns if mapped data is available
#         mapped_h5ad_path = paths['data_root'] / f"{paths['xenium_dataset_name']}_mapped" / f'section_{section_n}.h5ad'
#         if mapped_h5ad_path.exists():
#             section_sdata = add_mapped_cells_cols(section_sdata, mapped_h5ad_path)
#         else:
#             print(f"Mapped h5ad file not found: {mapped_h5ad_path}")

#         # Reformat section data to only include elements needed for alignment
#         print("Generating alignment spatial data...")
#         alignment_images, alignment_labels = get_alignment_spatial_elements(section_sdata)
#         alignment_table, alignment_transcripts, alignment_shapes = get_alignment_shapes_tables(section_sdata)
#         xenium_section = sd.SpatialData(
#             images={**alignment_images},
#             labels={**alignment_labels},
#             tables={'table': alignment_table},
#             points={'transcripts': alignment_transcripts},
#             shapes={**alignment_shapes}
#         )

#         # Get transcript bounding boxes
#         cell_label_bboxes = get_transcripts_bboxes(xenium_section['transcripts'], id_col='cell_labels')
#         xenium_section['table'].obs['transcripts_bbox'] = xenium_section['table'].obs['cell_labels'].map(cell_label_bboxes)

#         # Save the xenium section data for alignment, then reload
#         xenium_section.write(save_section_path)
#         del section_sdata, xenium_section
#         xenium_section = sd.read_zarr(save_section_path)
#     return xenium_section


# def generate_channel_tifs(sdata,
#                           channels,
#                           save_folder,
#                           overwrite=False):
#     save_folder = Path(save_folder)
#     save_folder.mkdir(parents=True, exist_ok=True)

#     for chan_name in channels:
#         if chan_name not in sdata:
#             print(f"Skipping missing channel: {chan_name}")
#             continue

#         out_path = save_folder / f"{chan_name}.tif"
#         if out_path.exists() and not overwrite:
#             print(f"Exists (skip): {out_path}")
#             continue
        
#         da = sdata[chan_name]
        
#         # 1. Extract the resolution from the 'microns' transform sequence
#         microns_transform = get_transformation(da, to_coordinate_system='microns')
#         if hasattr(microns_transform, 'transformations'):
#             scale_factors = [t.scale for t in microns_transform.transformations if hasattr(t, 'scale')]
#         else:
#             scale_factors = [microns_transform.scale] if hasattr(microns_transform, 'scale') else []
        
#         # final_resolution is [y_scale, x_scale] in microns/pixel
#         final_resolution = np.prod(scale_factors, axis=0)
        
#         # 2. Convert to pixels per micron for TIFF metadata (1 / microns_per_pixel)
#         # tifffile expects (x_resolution, y_resolution)
#         res_yx = 1.0 / final_resolution
#         tif_res = (res_yx[1], res_yx[0]) 

#         if hasattr(da, "dims") and 'c' in da.dims and da.sizes.get('c', 0) == 1:
#             da = da.isel(c=0)
            
#         arr = da.values if hasattr(da, 'values') else np.asarray(da)
#         axes = ''.join(da.dims) if hasattr(da, 'dims') else \
#                     ('zyx' if arr.ndim == 3 else ('yx' if arr.ndim == 2 else 'c' + ''.join(map(str, range(arr.ndim)))))

#         # 3. Add 'unit' to metadata so ImageJ recognizes 'um'
#         meta = {
#             'axes': axes,
#             'unit': 'um'
#         }

#         tifffile.imwrite(
#             str(out_path),
#             arr.astype('uint16', copy=False), # Ensure consistent dtype for ImageJ
#             imagej=True,
#             resolution=tif_res,
#             metadata=meta
#         )
#         print(f"Wrote: {out_path} with resolution {final_resolution} um/px")

# def generate_annotated_masks(sdata, 
#                             label_key, 
#                             column_name, 
#                             categories, 
#                             save_folder,
#                             table_name='table',
#                             table_labels_key='cell_labels',
#                             overwrite=False):
#     save_folder = Path(save_folder)
#     save_folder.mkdir(parents=True, exist_ok=True)

#     out_path = save_folder / f"{'_'.join(categories)}_mask.tif"
#     if out_path.exists() and not overwrite:
#         print(f"Exists (skip): {out_path}")
#         return

#     # 1. Identify valid IDs from the table
#     table = sdata[table_name]
#     mask_indices = table.obs[column_name].isin(categories)
#     # Ensure this column corresponds to the integer values in your label array
#     valid_ids = table.obs[table_labels_key][mask_indices].values.astype(int)

#     # 2. Get the labels element
#     labels_el = sdata.labels[label_key]
    
#     # 3. Extract the resolution from the 'microns' transform sequence
#     microns_transform = get_transformation(labels_el, to_coordinate_system='microns')
#     scale_factors = [t.scale for t in microns_transform.transformations if hasattr(t, 'scale')]
    
#     # final_resolution is [y_scale, x_scale] in microns/pixel
#     final_resolution = np.prod(scale_factors, axis=0)
    
#     # Convert to pixels per micron for TIFF (tifffile expects x, y)
#     res_yx = 1.0 / final_resolution
#     tif_res = (res_yx[1], res_yx[0]) 

#     # 4. Resolve the numpy array
#     label_arr = labels_el.values
#     if label_arr.ndim == 3: # Handle (C, Y, X)
#         label_arr = label_arr[0]

#     # 5. Create the mask: keep original ID if it's in our list, else 0
#     filtered_mask = np.where(np.isin(label_arr, valid_ids), label_arr, 0).astype('uint16')
    
#     # 6. Save with ImageJ physical units
#     tifffile.imwrite(
#         str(out_path),
#         filtered_mask,
#         imagej=True,
#         resolution=tif_res,
#         metadata={'unit': 'um', 'axes': 'YX'}
#     )
#     print(f"Wrote: {out_path} with resolution {final_resolution} um/px")
