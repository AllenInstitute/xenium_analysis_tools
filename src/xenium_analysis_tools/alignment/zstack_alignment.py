import re
import tifffile
import xarray as xr
from spatialdata.models import Image3DModel, Labels3DModel, TableModel
from spatialdata.transformations import Identity, Scale, set_transformation
import anndata as ad
from pathlib import Path
import spatialdata as sd
import numpy as np
import pandas as pd

def create_zstack_da(tif_path, name, add_chan=True, dims=("z", "y", "x"), fov_um=(450.0, 400.0, 400.0)):
    data = tifffile.imread(tif_path)
    # Calculate scale: microns per pixel
    pixel_sizes = [fov / pix for fov, pix in zip(fov_um, data.shape)]
    coords = {d: np.arange(data.shape[i]) * pixel_sizes[i] for i, d in enumerate(dims)}
    
    if add_chan:
        data = np.expand_dims(data, axis=0)
        coords["c"] = [name]
        current_dims = ("c",) + dims
    else:
        current_dims = dims
    
    da = xr.DataArray(data, coords=coords, dims=current_dims, name=name)
    da.attrs |= {f"scale_{d}": s for d, s in zip(dims, pixel_sizes)}
    return da

def parse_stack_metadata(folder, lookup_chans=['gcamp', 'dextran']):
    """Extracts size and specific channel name from naming convention."""
    # Pattern: Matches '400x400x450' and captures the following word (e.g., GCaMP)
    pattern = r'(\d+)x(\d+)x(\d+)'
    match = re.search(pattern, folder.name)
    
    if match:
        width, height, depth = match.groups()
        size = {"width": int(width), "height": int(height), "depth": int(depth)}
        # Normalize channel name (e.g., GCaMP -> gcamp)
        detected_channels = [ch for ch in lookup_chans if ch in folder.name.lower()]
    else:
        size = {"width": None, "height": None, "depth": None}
        detected_channels = folder.name # Fallback
    
    tifs = {d.name.lower(): list(d.glob("*.tif"))[0] 
            for d in folder.iterdir() if d.is_dir() and "channel" in d.name.lower()}
    jsons = {re.sub(r'.*_(registration|roi_groups|scanimage).*', r'\1', f.stem): f 
             for f in folder.glob("*.json")}

    return {"size": size, "detected_channels": detected_channels, "tifs": tifs, "jsons": jsons, "name": folder.name}

def get_zstack_elements(stack_folder, masks_folder, return_tables=False, chan_mapping=None, add_size_suffix=True):
    if chan_mapping is None:
        chan_mapping = {
            'channel_0_ref_0': 'gcamp',
            'channel_1_ref_1': 'dextran'
        }

    # Get metadata for stacks
    stack_meta = parse_stack_metadata(stack_folder, lookup_chans=list(chan_mapping.values()))
    sz = stack_meta['size']
    fov = (sz['depth'], sz['height'], sz['width'])

    # Get stack channel images
    images = {}
    for chan, tif_path in stack_meta['tifs'].items():
        chan_name = chan_mapping.get(chan, chan)
        img_da = create_zstack_da(tif_path, chan_name, add_chan=True, fov_um=fov)
        img_da.attrs.update()
        images[chan_name] = Image3DModel.parse(img_da, 
                                                c_coords=[chan_name],
                                                chunks='auto')
    
    # If specified to keep names unique, add size suffix
    if add_size_suffix:
        size_suffix = f"{fov[0]}x{fov[1]}x{fov[2]}"
        images = {f"{name}_{size_suffix}": img for name, img in images.items()}

    # Use name of stack to get corresponding masks
    img_name = stack_meta['name'].split('_registered')[0]
    all_masks = list(masks_folder.iterdir())
    matched_masks_path = [m for m in all_masks if img_name in m.name]
    if matched_masks_path:
        matched_masks_path = matched_masks_path[0] if len(matched_masks_path) == 1 else None
    if matched_masks_path is None:
        print(f"No matching mask found for {img_name} in {masks_folder}")

    # Get mask metadata
    masks_meta = parse_stack_metadata(matched_masks_path, lookup_chans=list(chan_mapping.values()))

    # Get labels and tables for each mask channel
    labels = {}
    tables = {}
    for chan, tif_path in masks_meta['tifs'].items():
        chan_name = chan_mapping.get(chan, chan)
        labels_name = f"{chan_name}_labels"
        if add_size_suffix:
            labels_name = f"{labels_name}_{size_suffix}"
        mask_da = create_zstack_da(tif_path, labels_name, add_chan=False, fov_um=fov)
        labels[labels_name] = Labels3DModel.parse(mask_da, chunks='auto')  
        if return_tables:
        # Corresponding table
            unique_ids = np.unique(mask_da.values)
            unique_ids = unique_ids[unique_ids > 0]
            obs = pd.DataFrame(unique_ids, columns=[f"{chan_name}_id"], index=unique_ids.astype(str))
            obs['region'] = labels_name
            ann = ad.AnnData(obs=obs)
            table_name = f"{chan_name}_table"
            if add_size_suffix:
                table_name = f"{table_name}_{size_suffix}"
            tables[table_name] = TableModel.parse(ann, region=labels_name, region_key='region', instance_key=f"{chan_name}_id")
    
    return images, labels, tables

def get_zstacks_sdata(stacks_folder, masks_folder, return_tables=False, chan_mapping=None):
    if chan_mapping is None:
        chan_mapping = {
            'channel_0_ref_0': 'gcamp',
            'channel_1_ref_1': 'dextran'
        }
    all_stacks = list(stacks_folder.iterdir())
    combined_images = {}
    combined_labels = {}
    combined_tables = {}
    if len(all_stacks) > 1:
        add_size_suffix = True
    else:
        add_size_suffix = False
    for zstack_folder in all_stacks:
        images, labels, tables = get_zstack_elements(zstack_folder, masks_folder, return_tables=return_tables, chan_mapping=chan_mapping, add_size_suffix=add_size_suffix)            
        combined_images.update(images)
        combined_labels.update(labels)
        if tables:
            combined_tables.update(tables)
    
    # Combine into SpatialData         
    sdata = sd.SpatialData(images=combined_images, labels=combined_labels, tables=combined_tables)
    # Apply Transformations
    for el_type in ['images', 'labels']:
        for name, el in getattr(sdata, el_type).items():
            set_transformation(el, Identity(), "global")
            scale = Scale([el.attrs[f"scale_{d}"] for d in ['z', 'y', 'x']], axes=('z', 'y', 'x'))
            set_transformation(el, scale, "microns")
    return sdata