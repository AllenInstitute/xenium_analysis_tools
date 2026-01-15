from spatialdata.transformations import get_transformation, set_transformation, remove_transformation, Scale, Identity, Sequence
from spatialdata.models import Image2DModel, Image3DModel, Labels2DModel, TableModel
import dask.array as da
import numpy as np
import xarray as xr
import spatialdata as sd
from spatialdata import bounding_box_query

def compare_transforms(slide_sdata, cropped_sdata):
    element_names = []
    for attr_name in ['images', 'labels', 'shapes', 'points', 'tables']:
        elements = getattr(slide_sdata, attr_name, {})
        element_names.extend(elements.keys())
    for elem in element_names:
        if elem == 'table':
            continue
        print(f"\n\nElement: {elem}")
        if hasattr(slide_sdata[elem], 'scale0'):
            print(f"\nSlide Spatial Data ")
            for key,item in slide_sdata[elem].items():
                print(f"Scale key: {key}")
                print(f"\t{get_transformation(item.image)}")

            print("\nCropped Spatial Data ")
            for key,item in cropped_sdata[elem].items():
                print(f"Scale key: {key}")
                print(f"{get_transformation(item.image)}")
        else:
            print("Slide Spatial Data ")
            print(f"\t{get_transformation(slide_sdata[elem])}")

            print("\nCropped Spatial Data ")
            print(f"\t{get_transformation(cropped_sdata[elem])}")

def crop_filter_sdata(sdata, bbox, c_system='global', crop_transcripts_separately=True):
    """
    Crop SpatialData object to a bounding box.
    """
    from spatialdata.transformations import get_transformation
    if crop_transcripts_separately:
        # Get all non-transcript element names
        element_names = []
        for attr_name in ['images', 'labels', 'shapes', 'tables']:
            elements = getattr(sdata, attr_name, {})
            element_names.extend(elements.keys())
        
        # Crop standard elements using built-in query
        if element_names:
            cropped_sdata = sdata.subset(element_names=element_names).query.bounding_box(
                axes=('y', 'x'),
                min_coordinate=[bbox['y_min'], bbox['x_min']],
                max_coordinate=[bbox['y_max'], bbox['x_max']],
                target_coordinate_system=c_system
            )
        else:
            cropped_sdata = sd.SpatialData()
        
        # Transcripts
        if hasattr(sdata, 'points') and sdata.points and 'transcripts' in sdata.points:
            # Get transformation and apply scaling
            transform = get_transformation(sdata['transcripts'], to_coordinate_system=c_system)
            transcripts_df = sdata['transcripts'].copy()
            
            # Fix duplicate indices from multi-partition Dask DataFrames FIRST
            if hasattr(transcripts_df, 'npartitions') and transcripts_df.npartitions > 1:
                transcripts_df = transcripts_df.reset_index(drop=True)
            
            # Convert to pandas DataFrame to ensure no Dask operations in parsing
            if hasattr(transcripts_df, 'compute'):
                transcripts_df = transcripts_df.compute()
            
            # Apply transformation to coordinates
            if hasattr(transform, 'scale') and transform.scale is not None:
                axes_map = {ax: i for i, ax in enumerate(transform.axes)}
                x_scale = transform.scale[axes_map.get('x', 0)]
                y_scale = transform.scale[axes_map.get('y', 1)]
                
                x_global = transcripts_df['x'] * x_scale
                y_global = transcripts_df['y'] * y_scale
            else:
                x_global = transcripts_df['x']
                y_global = transcripts_df['y']
            
            # Filter transcripts within bounding box
            mask = (
                (y_global >= bbox['y_min']) & (y_global <= bbox['y_max']) &
                (x_global >= bbox['x_min']) & (x_global <= bbox['x_max'])
            )
            
            filtered_transcripts = transcripts_df[mask]

            if len(filtered_transcripts) > 0:
                # Ensure clean index after filtering
                filtered_transcripts = filtered_transcripts.reset_index(drop=True)
                cropped_sdata.points['transcripts'] = sd.models.PointsModel.parse(filtered_transcripts)
        else:
            cropped_sdata = sdata.query.bounding_box(
                axes=('y', 'x'),
                min_coordinate=[bbox['y_min'], bbox['x_min']],
                max_coordinate=[bbox['y_max'], bbox['x_max']],
                target_coordinate_system=c_system
            )
    return cropped_sdata

def crop_dapi_image(dapi_image, bbox):
    cropped_dapi_image = xr.DataTree()

    for scale_level in dapi_image.keys():
        # Add scaling transformation to non-scale0 levels
        if scale_level != 'scale0':
            scale_factors = np.array(dapi_image[f'scale0'].image.shape) / np.array(dapi_image[scale_level].image.shape)
            scale_transform = Scale(scale_factors, axes=dapi_image[scale_level].image.dims)
            sequence = Sequence([scale_transform, Identity()])
            set_transformation(dapi_image[scale_level].image, sequence, to_coordinate_system="global")
        # Crop the image at this scale level
        cropped_scale = bounding_box_query(
            dapi_image[scale_level].image,
            axes=('y','x'),
            min_coordinate=[bbox['y_min'], bbox['x_min']],
            max_coordinate=[bbox['y_max'], bbox['x_max']],
            target_coordinate_system='global',
        )
        # Re-parse to ensure correct model
        parsed_image = Image3DModel.parse(
                cropped_scale,
                dims=['c', 'z', 'y', 'x'],
                c_coords=['DAPI'],
                chunks='auto',
            )
        
        # Add to the new DataTree
        cropped_dapi_image[scale_level] = xr.Dataset({'image': parsed_image})
    return cropped_dapi_image

def extract_scale_transform(current_transform):
    """Extract scale transformation from a compound transformation."""
    if hasattr(current_transform, 'transformations'):
        # Look for Scale transform in sequence
        for transform in current_transform.transformations:
            if isinstance(transform, Scale):
                return transform
    elif isinstance(current_transform, Scale):
        return current_transform
    return None
    
def reset_spatial_elements_coords(element, model_type='image', transforms=None):
    """
    Function to reset coordinates on spatial data elements.
    """
    data_tree_obj = xr.DataTree()
    
    for scale_level in element.keys():
        level_image = element[scale_level].image
        
        if transforms[scale_level] is None:
            # Get current transformation and extract scale component
            current_transform = get_transformation(level_image, to_coordinate_system='global')
            scale_transform = extract_scale_transform(current_transform)
            
            # Remove all transformations
            remove_transformation(level_image, to_coordinate_system='global')
            
            # Set only the scale transform or identity
            if scale_transform:
                set_transformation(level_image, scale_transform, to_coordinate_system='global')
            else:
                set_transformation(level_image, Identity(), to_coordinate_system='global')
        else:
            # Remove all transformations
            remove_transformation(level_image, to_coordinate_system='global')
            set_transformation(level_image, transforms[scale_level], to_coordinate_system='global')

        # Parse according to model type
        if model_type == 'label':
            parsed_image = Labels2DModel.parse(
                level_image,
                dims=list(level_image.dims),
                chunks='auto',
            )
        else:  # image type
            # Determine if 2D or 3D based on dimensions
            spatial_dims = len([d for d in level_image.dims if d in ['y', 'x', 'z']])
            
            if spatial_dims == 2:
                parsed_image = Image2DModel.parse(
                    level_image,
                    dims=list(level_image.dims),
                    c_coords=level_image['c'].values,
                    chunks='auto',
                )
            else:
                parsed_image = Image3DModel.parse(
                    level_image,
                    dims=list(level_image.dims),
                    c_coords=level_image['c'].values,
                    chunks='auto',
                )
        
        data_tree_obj[scale_level] = xr.Dataset({'image': parsed_image})
    
    return data_tree_obj

def reset_transcript_coords(transcripts, bbox, transcripts_transform, to_c_sys='global'):
    # Handle Dask/Partitioning
    if hasattr(transcripts, 'npartitions') and transcripts.npartitions > 1:
        transcripts = transcripts.reset_index(drop=True)
    if hasattr(transcripts, 'compute'):
        transcripts = transcripts.compute()

    # Save original coordinates
    transcripts['y_slide'] = transcripts['y'].astype('float64') 
    transcripts['x_slide'] = transcripts['x'].astype('float64')

    # Convert bbox to transcript coordinate system
    scale_transform = extract_scale_transform(transcripts_transform)
    if scale_transform is not None:
        # Get scale factors
        axes_to_index = {ax: i for i, ax in enumerate(scale_transform.axes)}
        x_scale = scale_transform.scale[axes_to_index['x']]
        y_scale = scale_transform.scale[axes_to_index['y']]
        
        # Convert bbox (pixels) to transcript coordinate system
        bbox_scaled = {
            'x_min': bbox['x_min'] / x_scale,
            'y_min': bbox['y_min'] / y_scale
        }
    else:
        print("No scale transform found, using original bbox.")
        bbox_scaled = bbox
        scale_transform = Identity()

    # Translate Coordinates
    transcripts['y'] = (transcripts['y'] - bbox_scaled['y_min']).astype('float64')
    transcripts['x'] = (transcripts['x'] - bbox_scaled['x_min']).astype('float64')

    # Type casting cleanup
    if 'is_gene' in transcripts.columns:
        transcripts['is_gene'] = transcripts['is_gene'].astype('str')
    if 'transcript_id' in transcripts.columns:
        transcripts['transcript_id'] = transcripts['transcript_id'].astype('float64')
    if 'overlaps_nucleus' in transcripts.columns:
        transcripts['overlaps_nucleus'] = transcripts['overlaps_nucleus'].astype('float64')
    if 'codeword_index' in transcripts.columns:
        transcripts['codeword_index'] = transcripts['codeword_index'].astype('float64')

    # Parse
    parsed_transcripts = sd.models.PointsModel.parse(transcripts)

    # Set transform
    set_transformation(parsed_transcripts, transcripts_transform, to_coordinate_system=to_c_sys)

    return parsed_transcripts

def reset_shapes_coordinates(shapes_element, bbox, shapes_transform, to_c_sys='global'):
    # Convert bbox to shape coordinate system
    scale_transform = extract_scale_transform(shapes_transform)
    if scale_transform is not None:
        # Get scale factors
        axes_to_index = {ax: i for i, ax in enumerate(scale_transform.axes)}
        x_scale = scale_transform.scale[axes_to_index['x']]
        y_scale = scale_transform.scale[axes_to_index['y']]
        
        # Convert bbox (pixels) to shape coordinate system
        bbox_scaled = {
            'x_min': bbox['x_min'] / x_scale,
            'y_min': bbox['y_min'] / y_scale
        }
    else:
        print("No scale transform found, using original bbox.")
        bbox_scaled = bbox
        scale_transform = Identity()

    # Translate Coordinates
    shapes_element.geometry = shapes_element.geometry.translate(
        xoff=-bbox_scaled['x_min'], 
        yoff=-bbox_scaled['y_min']
    )

    parsed_shapes = sd.models.ShapesModel.parse(shapes_element)
    set_transformation(parsed_shapes, shapes_transform, to_coordinate_system=to_c_sys)
    return parsed_shapes

def reset_spatial_element_coords(cropped_el, full_el, model_type, func, element_transforms=None, to_c_sys='global'):
    element_transforms = {}
    for scale_level in full_el.keys():
        element_transforms[scale_level] = get_transformation(full_el[scale_level].image, to_coordinate_system=to_c_sys)
    element = func(
            cropped_el,
            model_type=model_type,
            transforms=element_transforms
    )
    return element

def reset_table_coordinates(table, table_transform, bbox):
    # Convert bbox to shape coordinate system
    scale_transform = extract_scale_transform(table_transform)
    if scale_transform is not None:
        # Get scale factors
        axes_to_index = {ax: i for i, ax in enumerate(scale_transform.axes)}
        x_scale = scale_transform.scale[axes_to_index['x']]
        y_scale = scale_transform.scale[axes_to_index['y']]
        
        # Convert bbox (pixels) to shape coordinate system
        bbox_scaled = {
            'x_min': bbox['x_min'] / x_scale,
            'y_min': bbox['y_min'] / y_scale
        }
    else:
        print("No scale transform found, using original bbox.")
        bbox_scaled = bbox
        scale_transform = Identity()

    # Update centroids to be relative to the cropped bounding box
    table.obsm['spatial'][:, 0] = table.obsm['spatial'][:, 0] - bbox_scaled['x_min']
    table.obsm['spatial'][:, 1] = table.obsm['spatial'][:, 1] - bbox_scaled['y_min']

    # Also update any other spatial coordinates in the table if they exist
    if 'x' in table.obs.columns:
        table.obs['x'] = table.obs['x'] - bbox_scaled['x_min']
    if 'y' in table.obs.columns:
        table.obs['y'] = table.obs['y'] - bbox_scaled['y_min']

    # Fix sections_bboxes to have string keys for zarr compatibility
    if 'sections_bboxes' in table.uns and table.uns['sections_bboxes'] is not None:
        sections_bboxes_str_keys = {}
        for key, value in table.uns['sections_bboxes'].items():
            sections_bboxes_str_keys[str(key)] = value
        table.uns['sections_bboxes'] = sections_bboxes_str_keys

    # Fix any other dictionary keys in uns that might have integer keys
    for key, value in table.uns.items():
        if isinstance(value, dict):
            # Check if any keys are integers and convert to strings
            if any(isinstance(k, int) for k in value.keys()):
                table.uns[key] = {str(k): v for k, v in value.items()}

    parsed_table = TableModel.parse(adata=table)
    return parsed_table

def reset_section_coordinates(cropped_sdata, cropped_dapi_zstack, full_dapi_zstack, slide_sdata, section):
    # Get the full scale section bbox
    fs_bbox = cropped_sdata['table'].uns['sections_bboxes'][str(section)]

    # Morphology Focus
    element_name = 'morphology_focus'
    morphology_focus = reset_spatial_element_coords(
        cropped_sdata[element_name],
        slide_sdata[element_name],
        model_type='image',
        func=reset_spatial_elements_coords
    )

    # DAPI Z-Stack
    element_name = 'dapi_zstack'
    dapi_zstack = reset_spatial_element_coords(
            cropped_dapi_zstack,
            full_dapi_zstack,
            model_type='image',
            func=reset_spatial_elements_coords
        )

    # Cell Labels
    element_name = 'cell_labels'
    cell_labels = reset_spatial_element_coords(
        cropped_sdata[element_name],
        slide_sdata[element_name],
        model_type='label',
        func=reset_spatial_elements_coords
    )

    # Nucleus Labels
    element_name = 'nucleus_labels'
    nucleus_labels = reset_spatial_element_coords(
        cropped_sdata[element_name],
        slide_sdata[element_name],
        model_type='label',
        func=reset_spatial_elements_coords
    )

    # Transcripts
    transcripts = reset_transcript_coords(
        cropped_sdata['transcripts'],
        bbox=fs_bbox,
        transcripts_transform=get_transformation(slide_sdata['transcripts'])
    )

    # Cell boundaries
    cell_boundaries = reset_shapes_coordinates(
        cropped_sdata['cell_boundaries'], 
        bbox=fs_bbox, 
        shapes_transform=get_transformation(slide_sdata['cell_boundaries'])
    )

    # Nucleus boundaries
    nucleus_boundaries = reset_shapes_coordinates(
        cropped_sdata['nucleus_boundaries'], 
        bbox=fs_bbox, 
        shapes_transform=get_transformation(slide_sdata['nucleus_boundaries'])
    )

    # Cell circles
    cell_circles = reset_shapes_coordinates(
        cropped_sdata['cell_circles'], 
        bbox=fs_bbox, 
        shapes_transform=get_transformation(slide_sdata['cell_circles'])
    )

    # Table
    table = reset_table_coordinates(
        table=cropped_sdata['table'],
        table_transform=get_transformation(slide_sdata['transcripts']), # Same scale as transcripts
        bbox=fs_bbox
    )

    # Assemble section SpatialData
    section_sdata = sd.SpatialData(
        images={'morphology_focus': morphology_focus, 'dapi_zstack': dapi_zstack},
        labels={'cell_labels': cell_labels, 'nucleus_labels': nucleus_labels},
        shapes={'cell_boundaries': cell_boundaries, 'nucleus_boundaries': nucleus_boundaries, 'cell_circles': cell_circles},
        points={'transcripts': transcripts},
        tables={'table': table},
    )
    
    return section_sdata