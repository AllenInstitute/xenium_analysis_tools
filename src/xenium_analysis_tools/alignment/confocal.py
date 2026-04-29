import dask.array as da
import tifffile
import pandas as pd
from pathlib import Path
from dask_image.imread import imread
from IPython.display import display
from tqdm.notebook import tqdm
import xarray as xr
import spatialdata as sd
from spatialdata.models import Image3DModel
from xenium_analysis_tools.utils.sd_utils import (
    add_micron_coord_sys,
    write_sdata_elements
)

def get_confocal_sdata(dataset_id,
                        confocal_path, 
                        raw_confocal_path=None,
                        img_names=['surface','deep'],
                        save_folder=None,
                        xy_pixel_size=0.31862745,
                        z_step=1,
                        chunks={'c': 1, 'z': 1, 'y': 512, 'x': 512},
                        num_workers=8):
    confocal_sdata = sd.SpatialData()
    with tqdm(img_names, desc="Processing images") as pbar:
        for img_name in pbar:
            pbar.set_description(f"Processing [{img_name}]:")
            img_path = confocal_path / f"{img_name}.tif"
            if raw_confocal_path is not None and raw_confocal_path.exists():
                confocal_metadata = get_confocal_metadata(
                                        dataset_id=dataset_id,
                                        raw_confocal_path=raw_confocal_path,
                                        note_name=img_name,
                                        processed_confocal=img_path,
                                    )
                xy_pixel_size = confocal_metadata['xy_pixel_size']
                z_step = confocal_metadata['zstep']
            darr = imread(str(img_path))  # lazy dask array, no data loaded
            fr_xarray = xr.DataArray(darr, dims=('z', 'y', 'x'))
            fr_xarray = fr_xarray.expand_dims(c=[0]).transpose('c', 'z', 'y', 'x')
            fr_xarray.attrs.update(confocal_metadata)
            img_el = Image3DModel.parse(fr_xarray,
                                        c_coords=['confocal'],
                                        scale_factors=[
                                            {'z': 1, 'y': 2, 'x': 2},
                                            {'z': 1, 'y': 2, 'x': 2},
                                            {'z': 1, 'y': 2, 'x': 2},
                                        ],
                                    )
            # Rechunk after parse — Image3DModel.parse overrides input chunks
            if chunks is not None:
                for scale_key in img_el.keys():
                    img_el[scale_key]['image'] = img_el[scale_key].image.chunk(chunks)
            confocal_sdata.images[img_name] = img_el
    confocal_sdata = add_micron_coord_sys(confocal_sdata, pixel_size=xy_pixel_size, z_step=z_step)
    if save_folder is not None:
        print(f"Saving confocal SpatialData to: {save_folder}")
        write_sdata_elements(confocal_sdata, save_folder, num_workers=num_workers)
    return confocal_sdata

def parse_imagerecord_blocks(text):
    blocks = []
    lines = text.splitlines()
    i = 0
    n = len(lines)

    while i < n:
        if lines[i].strip() == 'StartClass:':
            start = i
            i += 1
            while i < n and not lines[i].startswith('EndClass:'):
                i += 1
            if i < n and lines[i].startswith('EndClass:'):
                block_text = '\n'.join(lines[start:i + 1])
                try:
                    block = yaml.safe_load(block_text)
                    if isinstance(block, dict):
                        blocks.append(block)
                except Exception as e:
                    blocks.append({'parse_error': str(e), 'raw_block': block_text[:500]})
        i += 1

    return blocks

def get_confocal_metadata(dataset_id, raw_confocal_path, note_name='surface', processed_confocal=None):
    import yaml

    def parse_startclass_blocks(text):
        """Parse SlideBook-style YAML blocks delimited by StartClass:/EndClass:."""
        blocks = []
        lines = text.splitlines()
        i = 0
        n = len(lines)

        while i < n:
            if lines[i].strip() == 'StartClass:':
                start = i
                i += 1
                while i < n and not lines[i].startswith('EndClass:'):
                    i += 1
                if i < n and lines[i].startswith('EndClass:'):
                    block_text = '\n'.join(lines[start:i + 1])
                    try:
                        parsed = yaml.safe_load(block_text)
                        if isinstance(parsed, dict):
                            blocks.append(parsed.get('StartClass', {}))
                    except Exception:
                        pass
            i += 1
        return blocks

    cf_csv = pd.read_csv(raw_confocal_path / 'notes.csv')
    capture_name = cf_csv.loc[cf_csv['note'] == note_name, 'capture names'].values[0]
    cap_dir = raw_confocal_path / f'{dataset_id}.dir/{capture_name}.imgdir'

    # ImageRecord.yaml
    with open(cap_dir / 'ImageRecord.yaml', 'r') as f:
        image_blocks = parse_startclass_blocks(f.read())

    image_record = next((b for b in image_blocks if b.get('ClassName') == 'CImageRecord70'), None)
    lens_def = next((b for b in image_blocks if b.get('ClassName') == 'CLensDef70'), None)

    # ChannelRecord.yaml
    with open(cap_dir / 'ChannelRecord.yaml', 'r') as f:
        channel_blocks = parse_startclass_blocks(f.read())

    exposure_record = next((b for b in channel_blocks if b.get('ClassName') == 'CExposureRecord70'), None)

    # Native tile metadata (single capture frame)
    xy_pixel_size = lens_def.get('mMicronPerPixel') if lens_def else None
    tile_image_size = (image_record.get('mWidth'), image_record.get('mHeight')) if image_record else None
    z_planes = image_record.get('mNumPlanes') if image_record else None
    zstep = exposure_record.get('mInterplaneSpacing') if exposure_record else None

    # Optional stitched image metadata from processed TIFF / Zarr
    stitched_shape_zyx = None
    stitched_image_size = None
    if isinstance(processed_confocal, Path) and processed_confocal.suffix == '.tif':
        if processed_confocal.exists():
            with tifffile.TiffFile(processed_confocal) as tf:
                stitched_shape_zyx = tf.series[0].shape
            if len(stitched_shape_zyx) == 3:
                stitched_image_size = (int(stitched_shape_zyx[2]), int(stitched_shape_zyx[1]))
    elif isinstance(processed_confocal, Path) and processed_confocal.suffix == '.zarr':
        if processed_confocal.exists():
            stitched_shape_zyx = da.from_zarr(processed_confocal / '0').shape
            if len(stitched_shape_zyx) == 3:
                stitched_image_size = (int(stitched_shape_zyx[2]), int(stitched_shape_zyx[1]))
    else:
        print("Unsupported processed_confocal format")

    out = {
        'xy_pixel_size': xy_pixel_size,
        'image_size': tile_image_size,
        'z_planes': z_planes,
        'zstep': zstep,
        'tile_image_size': tile_image_size,
        'stitched_shape_zyx': stitched_shape_zyx,
        'stitched_image_size': stitched_image_size,
    }

    if xy_pixel_size is not None:
        out['tile_extent_um_xy'] = (
            tile_image_size[0] * xy_pixel_size if tile_image_size else None,
            tile_image_size[1] * xy_pixel_size if tile_image_size else None,
        )
        out['stitched_extent_um_xy'] = (
            stitched_image_size[0] * xy_pixel_size if stitched_image_size else None,
            stitched_image_size[1] * xy_pixel_size if stitched_image_size else None,
        )

    return out
