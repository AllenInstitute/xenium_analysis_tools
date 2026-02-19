import re
import tifffile
import xarray as xr
from spatialdata.models import Image3DModel, Labels3DModel, TableModel
from spatialdata.transformations import Identity, Scale, set_transformation
from xenium_analysis_tools.utils.sd_utils import add_micron_coord_sys
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

def get_zstack_images(stack_folder, chan_mapping=None, chunk_size=(1, 256, 256, 256)):
    if chan_mapping is None:
        chan_mapping = {
             'channel_0_ref_0': 'gcamp',
             'channel_1_ref_1': 'dextran'
         }
    stack_meta = parse_stack_metadata(stack_folder, lookup_chans=list(chan_mapping.values()))
    sz = stack_meta['size']
    fov = (sz['depth'], sz['height'], sz['width'])

    img_das = []
    channel_names = []
    for i, (chan, tif_path) in enumerate(stack_meta['tifs'].items()):
        chan_name = chan_mapping.get(chan, chan)
        img_da = create_zstack_da(tif_path, chan_name, fov_um=fov, add_chan=False)
        if i == 0:
            metadata = img_da.attrs
        
        # Remove channel dimension if it exists (since we'll add it back)
        if 'c' in img_da.dims:
            img_da = img_da.squeeze('c')
        
        img_das.append(img_da)
        channel_names.append(chan_name)

    # Concatenate along new channel dimension
    img_chans_da = xr.concat(img_das, dim='c')

    # Assign proper channel coordinates
    img_chans_da = img_chans_da.assign_coords(c=channel_names)

    # Always make gcamp the first channel
    if 'gcamp' in channel_names:
        gcamp_idx = channel_names.index('gcamp')
        if gcamp_idx != 0:
            # Swap the gcamp channel to the first position
            img_chans_da = img_chans_da.transpose('c', 'z', 'y', 'x')
            img_chans_da = img_chans_da.reindex(c=['gcamp'] + [ch for ch in channel_names if ch != 'gcamp'])
            img_chans_da = img_chans_da.transpose('c', 'z', 'y', 'x')
            channel_names = ['gcamp'] + [ch for ch in channel_names if ch != 'gcamp']

    # Restore metadata
    img_chans_da.attrs = metadata
    img_chans_da = img_chans_da.rename('zstack')
    zstack_chans = Image3DModel.parse(img_chans_da,
                            c_coords=channel_names,
                            chunks=chunk_size,
                            scale_factors=[2])
    zstack_chans.attrs.update(img_chans_da.attrs)
    zstack_chans.attrs['fov_size'] = fov
    return zstack_chans


def get_zstack_labels(stack_folder, zstack_masks, chan_mapping=None, chunk_size=(256, 256, 256)):
    if chan_mapping is None:
        chan_mapping = {
             'channel_0_ref_0': 'gcamp',
             'channel_1_ref_1': 'dextran'
         }
    masks_folder = [mask_path for mask_path in list(zstack_masks.iterdir()) if mask_path.stem.split('_segmented')[0]==stack_folder.stem.split('_registered')[0]][0]
    stack_meta = parse_stack_metadata(masks_folder, lookup_chans=list(chan_mapping.values()))
    sz = stack_meta['size']
    fov = (sz['depth'], sz['height'], sz['width'])
    labels = {}
    for i, (chan, tif_path) in enumerate(stack_meta['tifs'].items()):
        chan_name = chan_mapping.get(chan, chan)
        labels_name = f"{chan_name}_labels"
        img_da = create_zstack_da(tif_path, chan_name, fov_um=fov, add_chan=False)
        chan_masks = Labels3DModel.parse(img_da, chunks=chunk_size, scale_factors=[2])
        chan_masks.attrs.update(img_da.attrs)
        chan_masks.attrs['fov_size'] = fov
        labels[labels_name] = chan_masks
    return labels

def generate_zstack_sdata(zstacks_path, zstack_masks_path, image_chunk_size=(1, 256, 256, 256), label_chunk_size=(256, 256, 256)):
    sdata_dict = {}
    if len(list(zstacks_path.iterdir())) > 1:
        add_size_suffix = True
        sdata_dict = {}
    else:
        add_size_suffix = False
    for stack_folder in zstacks_path.iterdir():
        zstack_images = get_zstack_images(stack_folder, chunk_size=image_chunk_size)
        zstack_labels = get_zstack_labels(stack_folder, zstack_masks=zstack_masks_path, chunk_size=label_chunk_size)
        sdata = sd.SpatialData(
            images={'zstack': zstack_images},
            labels={**zstack_labels}
        )
        if add_size_suffix:
            size_suffix = f"{zstack_images.attrs['fov_size'][0]}x{zstack_images.attrs['fov_size'][1]}x{zstack_images.attrs['fov_size'][2]}"
            sdata_name = f"zstack_{size_suffix}"
        else:
            sdata_name = "zstack"
        sdata = add_micron_coord_sys(sdata, pixel_size=[sdata['zstack'].attrs['scale_y'], sdata['zstack'].attrs['scale_x']], z_step=sdata['zstack'].attrs['scale_z'])
        sdata_dict[sdata_name] = sdata
    return sdata_dict