import json
import numpy as np
import pandas as pd
import spatialdata as sd
import dask
import dask.array as da
import zarr
from spatialdata.models import Image3DModel
import tifffile
import spatialdata_io
from sklearn.cluster import KMeans

def read_xenium_slide(xenium_bundle_path, params, dask_config=None):
    if dask_config is None:
        dask_config = {}

    with dask.config.set(dask_config):
        sdata = spatialdata_io.xenium(
            path=xenium_bundle_path,
            cells_as_circles=params.get('cells_as_circles', True),
            gex_only=params.get('gex_only', False),
            n_jobs=params.get('n_jobs', 1),
        )

    return sdata

def get_section_metadata(xenium_data_path):
    """Add section metadata to anndata"""
    metrics_summary_csv = xenium_data_path / 'metrics_summary.csv'
    section_metadata = pd.read_csv(metrics_summary_csv, nrows=1)
    section_metadata = section_metadata.iloc[0].replace({np.nan: None}).to_dict()
    
    experiment_xenium_json = xenium_data_path / 'experiment.xenium'
    with open(experiment_xenium_json, 'r') as f:
        experiment_metadata = json.load(f)
    
    section_metadata.update(experiment_metadata)
    section_metadata.update({'xenium_bundle_name': xenium_data_path.name})

    return section_metadata

def get_fov_metadata(xenium_data_path):
    """Add FOV metadata with section assignments"""
    morphology_fov_locations_file = xenium_data_path / 'aux_outputs' / 'morphology_fov_locations.json'
    with open(morphology_fov_locations_file, 'r') as f:
        fov_locations = json.load(f)

    fov_df = pd.DataFrame.from_dict(fov_locations['fov_locations'], orient='index')
    fov_df.reset_index(names='fov_name', inplace=True)
    fov_df['units'] = fov_locations.get('units', 'microns')
    return fov_df

def assign_fov_sections(df, section_order, print_counts=False):
    """
    Assign section IDs using K-Means clustering and pre-calculate 
    bounding box coordinates for downstream processing.
    """
    df = df.copy()
    
    # 1. Calculate boundaries (Required for get_fov_bboxes)
    df['x_min'] = df['x']
    df['x_max'] = df['x'] + df['width']
    df['y_min'] = df['y']
    df['y_max'] = df['y'] + df['height']

    # 2. Find centroids for Clustering
    df['cx'] = df['x'] + df['width'] / 2
    df['cy'] = df['y'] + df['height'] / 2
    
    # 3. Perform K-Means Clustering
    n_sections = len(section_order)
    coords = df[['cx', 'cy']].values
    
    # random_state=42 ensures reproducibility
    kmeans = KMeans(n_clusters=n_sections, random_state=42, n_init=10)
    df['cluster_label'] = kmeans.fit_predict(coords)
    
    # 4. Map Clusters to Spatial Order
    # Calculate the average position for each cluster to sort them
    cluster_stats = df.groupby('cluster_label').agg({
        'cy': 'mean', 
        'cx': 'mean'
    }).reset_index()
    
    # Determine dominant axis (Vertical vs Horizontal)
    y_range = df['cy'].max() - df['cy'].min()
    x_range = df['cx'].max() - df['cx'].min()
    
    if y_range > x_range:
        # Sort Top-to-Bottom
        cluster_stats = cluster_stats.sort_values('cy')
    else:
        # Sort Left-to-Right
        cluster_stats = cluster_stats.sort_values('cx')
        
    # Create mapping: Cluster Label -> Order Index -> Section Name
    label_to_order = {label: idx for idx, label in enumerate(cluster_stats['cluster_label'])}
    
    df['section'] = df['cluster_label'].map(label_to_order).map(lambda i: section_order[i])
    
    # Logging
    if print_counts:
        print(f"Assigned {n_sections} sections using K-Means.")
        for sec in section_order:
            count = (df['section'] == sec).sum()
            print(f"Section {sec}: {count} FOVs")
        
    # Return df with x_min/max etc., but drop temporary clustering columns
    return df.drop(columns=['cx', 'cy', 'cluster_label'])
    

def get_fov_bboxes(fov_metadata):
    """Vectorized calculation of FOV bounding boxes per section."""
    grouped = fov_metadata.groupby('section')
    pixel_sizes = grouped['pixel_size'].first()
    
    bboxes = grouped.agg({
        'x_min': 'min', 'x_max': 'max',
        'y_min': 'min', 'y_max': 'max'
    })
    
    bboxes_normalized = bboxes.div(pixel_sizes, axis=0)
    bboxes_normalized = bboxes_normalized.round().astype(int)
    return bboxes_normalized.to_dict(orient='index')

def get_ome_metadata(tif_path, level_n=0):
    import xml.etree.ElementTree as ET
    import tifffile
    with tifffile.TiffFile(tif_path, is_ome=True) as tif:
        ome_metadata = tif.ome_metadata
        root = ET.fromstring(ome_metadata)
        ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
        pixels_elem = root.find('.//ome:Image/ome:Pixels', ns)
        
        if hasattr(tif.series[0], 'levels'):
            page = tif.series[0].levels[level_n].pages[0]
        else:
            page = tif.pages[0]
            
        metadata = {
            'samples_per_pixel': page.tags.get('SamplesPerPixel').value,
            'PhysicalSizeX': float(pixels_elem.get('PhysicalSizeX', 0.2125)),
            'PhysicalSizeY': float(pixels_elem.get('PhysicalSizeY', 0.2125)),
            'PhysicalSizeZ': float(pixels_elem.get('PhysicalSizeZ', 3.0)),
        }
        if hasattr(tif.series[0], 'levels'):
            scale_factor = 2**level_n
            metadata['PhysicalSizeX'] *= scale_factor
            metadata['PhysicalSizeY'] *= scale_factor
            metadata['PhysicalSizeZ'] *= scale_factor
    return metadata

def get_dapi_zstack(ome_tif_path, sdata):
    import xarray as xr
    if not ome_tif_path.exists():
        raise FileNotFoundError(f"DAPI file not found: {ome_tif_path}")

    datatree_obj = xr.DataTree()
    tiff_store = tifffile.imread(ome_tif_path, aszarr=True)
    tiff_store.path = str(ome_tif_path)
    z_tiff = zarr.open(tiff_store, mode='r')

    for level_name in list(sdata['morphology_focus'].keys()):
        level_id = level_name.rsplit('scale')[1]
        dask_data = da.from_zarr(z_tiff[level_id])
        dask_data = dask_data[np.newaxis, :, :, :]

        parsed_image = Image3DModel.parse(
                dask_data, dims=['c', 'z', 'y', 'x'], c_coords=['DAPI'], chunks='auto',
        )
        parsed_image.attrs.update(get_ome_metadata(ome_tif_path, level_n=int(level_id)))
        parsed_image.attrs['scale_level'] = int(level_id)
        datatree_obj[f'scale{level_id}'] = xr.Dataset({'image': parsed_image})
    return datatree_obj

def process_metadata(sdata, xenium_bundle_path, slide_sections):
    anndata = sdata['table'].copy()
    section_metadata = get_section_metadata(xenium_bundle_path)
    section_metadata['sections_on_slide'] = slide_sections

    fov_metadata = get_fov_metadata(xenium_bundle_path)
    fov_metadata = assign_fov_sections(fov_metadata, sorted(slide_sections))
    fov_metadata['pixel_size'] = section_metadata.get('pixel_size', 0.2125)

    if len(slide_sections) > 1:
        section_bboxes = get_fov_bboxes(fov_metadata)
        section_bboxes_str_keys = {str(k): v for k, v in section_bboxes.items()}
        section_bboxes = section_bboxes_str_keys
    else:
        section_bboxes = None

    anndata.uns['section_metadata'] = section_metadata
    anndata.uns['sections_bboxes'] = section_bboxes    
    anndata.uns['fov_metadata'] = fov_metadata
    sdata['table'] = anndata

    return sdata