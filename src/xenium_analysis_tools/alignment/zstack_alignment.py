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

def get_zstack_sdata(zstack_path, zstack_masks_path, target_size, channel_names=['gcamp', 'dextran']):
    def get_matching_metas(root):
        matches = []
        for d in Path(root).iterdir():
            if not d.is_dir(): continue
            meta = _parse_stack_metadata(d) # Extracts size and tif paths
            if meta['size'] == target_size:
                matches.append(meta)
        return sorted(matches, key=lambda x: x['name']) # Sort ensures Channel 0 -> gcamp

    img_metas = get_matching_metas(zstack_path)
    mask_metas = get_matching_metas(zstack_masks_path)
    
    sz = img_metas[0]['size']
    fov = (sz['depth'], sz['height'], sz['width'])
    images, labels, tables = {}, {}, {}

    for i, img_meta in enumerate(img_metas):
        chan_name = channel_names[i] if i < len(channel_names) else f"channel_{i}"
        
        # Process Image
        img_tif = next(iter(img_meta['tifs'].values()))
        img_da = create_zstack_da(img_tif, chan_name, add_chan=True, fov_um=fov)
        images[chan_name] = Image3DModel.parse(img_da, chunks='auto')

        # Process Labels & Tables
        if i < len(mask_metas):
            mask_meta = mask_metas[i]
            mask_tif = next(iter(mask_meta['tifs'].values()))
            label_key = f"{chan_name}_labels"
            
            mask_da = create_zstack_da(mask_tif, label_key, add_chan=False, fov_um=fov)
            labels[label_key] = Labels3DModel.parse(mask_da, chunks='auto')
            
            # Table logic
            unique_ids = np.unique(mask_da.values)
            unique_ids = unique_ids[unique_ids > 0]
            obs = pd.DataFrame(unique_ids, columns=[f"{chan_name}_id"], index=unique_ids.astype(str))
            obs['region'] = label_key
            ann = ad.AnnData(obs=obs)
            tables[f"{chan_name}_cells"] = TableModel.parse(ann, region=label_key, region_key='region', instance_key=f"{chan_name}_id")

    sdata = sd.SpatialData(images=images, labels=labels, tables=tables)
    
    # Apply Transformations
    for el_type in ['images', 'labels']:
        for name, el in getattr(sdata, el_type).items():
            set_transformation(el, Identity(), "global")
            scale = Scale([el.attrs[f"scale_{d}"] for d in ['z', 'y', 'x']], axes=('z', 'y', 'x'))
            set_transformation(el, scale, "microns")
            
    return sdata

def _parse_stack_metadata(folder):
    """Extracts size and specific channel name from the Allen Institute folder naming convention."""
    # Pattern: Matches '400x400x450' and captures the following word (e.g., GCaMP)
    pattern = r'(\d+)x(\d+)x(\d+)-([^_]+)'
    match = re.search(pattern, folder.name)
    
    if match:
        width, height, depth, channel = match.groups()
        size = {"width": int(width), "height": int(height), "depth": int(depth)}
        # Normalize channel name (e.g., GCaMP -> gcamp)
        detected_channel = channel.lower() 
    else:
        size = {"width": None, "height": None, "depth": None}
        detected_channel = folder.name # Fallback
    
    tifs = {d.name.lower(): list(d.glob("*.tif"))[0] 
            for d in folder.iterdir() if d.is_dir() and "channel" in d.name.lower()}
    jsons = {re.sub(r'.*_(registration|roi_groups|scanimage).*', r'\1', f.stem): f 
             for f in folder.glob("*.json")}
    
    return {"size": size, "detected_channel": detected_channel, "tifs": tifs, "jsons": jsons, "name": folder.name}