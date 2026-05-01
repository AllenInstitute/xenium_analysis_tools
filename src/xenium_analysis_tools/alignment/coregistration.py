import numpy as np
import pandas as pd
import spatialdata as sd
import tifffile
from IPython.display import display
from tqdm.notebook import tqdm

import gc
import json
from pathlib import Path

import dask
import dask.array as da
import concurrent.futures

import re
import struct

from spatialdata.transformations import get_transformation, Scale, Identity, Sequence

from xenium_analysis_tools.utils.sd_utils import (
    add_micron_coord_sys,
    _is_multiscale,
)

from spatialdata.transformations import (
    get_transformation,
    Scale,
    Identity,
    Sequence
)

from xenium_analysis_tools.alignment.format_for_napari import (
    add_mapped_cells_cols,
    filter_cells,
    filter_labels,
)

from spatialdata.models import TableModel

# BigWarp packed-int ARGB color values
COLOR_MAP = {
    'red':     -65536,
    'green':   -16711936,
    'blue':    -16776961,
    'white':   -1,
    'gray':    -1,
    'cyan':    -16711681,
    'magenta': -65281,
    'yellow':  -256,
}

class AlignmentImageGenerator:
    """Orchestrates zarr generation and TIFF extraction for BigWarp alignment.

    Parameters
    ----------
    dataset_id      : int
    paths           : dict  — output of get_dataset_paths()
    zstack_ms_level : int   — multiscale level for gcamp (0 = full res)
    xenium_ms_level : int   — multiscale level for Xenium sections (2 = 4x down)
    cf_ms_level     : int   — multiscale level for confocal (3 = 8x down)
    overwrite_*     : bool  — per-modality overwrite flags
    """

    def __init__(
        self,
        dataset_id,
        paths,
        zstack_ms_level: int = 0,
        xenium_ms_level: int = 2,
        cf_ms_level: int = 3,
        overwrite_sections: bool = False,
        overwrite_czstack: bool = False,
        overwrite_confocal: bool = False,
    ):
        self.dataset_id       = dataset_id
        self.paths            = paths
        self.zstack_ms_level  = zstack_ms_level
        self.xenium_ms_level  = xenium_ms_level
        self.cf_ms_level      = cf_ms_level
        self.overwrite_sections = overwrite_sections
        self.overwrite_czstack  = overwrite_czstack
        self.overwrite_confocal = overwrite_confocal

        self.cf_sdata      = None
        self.czstack_sdata = None

    # ── Zarr generation ───────────────────────────────────────────────────

    def load_confocal(self) -> sd.SpatialData:
        from xenium_analysis_tools.alignment.confocal import get_confocal_sdata
        save_path = self.paths['alignment_folder'] / 'confocal' / 'surface.zarr'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if not save_path.exists() or self.overwrite_confocal:
            self.cf_sdata = get_confocal_sdata(
                dataset_id=self.dataset_id,
                confocal_path=self.paths['confocal_path'],
                raw_confocal_path=self.paths['raw_confocal_path'],
                img_names=['surface'],
                save_folder=save_path,
            )
        else:
            self.cf_sdata = sd.read_zarr(save_path)
        return self.cf_sdata

    def load_czstack(self) -> sd.SpatialData:
        from xenium_analysis_tools.alignment.cortical_zstack import get_zstack_sdata
        save_path = self.paths['alignment_folder'] / 'czstack' / 'czstack.zarr'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if not save_path.exists() or self.overwrite_czstack:
            sdata = get_zstack_sdata(
                zstack_img_gcamp_path=self.paths['gcamp_image_path'],
                zstack_masks_gcamp_path=self.paths['gcamp_masks_path'],
            )
            xy_res = sdata['gcamp_table'].uns['segmentation_metadata'].get('xy_resolution') \
                     or sdata['gcamp'].attrs['scale_x']
            z_res  = sdata['gcamp'].attrs.get('scale_z', 1.0)
            sdata  = add_micron_coord_sys(sdata, pixel_size=xy_res, z_step=z_res)
            write_sdata_elements(sdata, save_path)
        self.czstack_sdata = sd.read_zarr(save_path)
        return self.czstack_sdata

    # ── TIFF extraction ───────────────────────────────────────────────────

    def extract_confocal_tif(self):
        out = self.paths['coregistration_folder'] / 'confocal.tif'
        if out.exists() and not self.overwrite_confocal:
            print(f"confocal.tif already exists, skipping")
            return
        cf = self.cf_sdata or self.load_confocal()
        tf = get_transformation(sd.get_pyramid_levels(cf['surface'], n=0), to_coordinate_system='microns')
        extract_bigwarp_images(sdata=cf, output_folder=self.paths['coregistration_folder'],
                               el_name='surface', multiscale_level=self.cf_ms_level,
                               z_step_um=get_dim_scale(tf, 'z'))

    def extract_czstack_tifs(self):
        coreg = self.paths['coregistration_folder']
        czs   = self.czstack_sdata or self.load_czstack()
        tf    = get_transformation(czs['gcamp'], to_coordinate_system='microns')
        z_step = get_dim_scale(tf, 'z')

        if not (coreg / 'gcamp.tif').exists() or self.overwrite_czstack:
            extract_bigwarp_images(sdata=czs, output_folder=coreg,
                                   el_name='gcamp', multiscale_level=self.zstack_ms_level,
                                   z_step_um=z_step)
        if not (coreg / 'gcamp_labels.tif').exists() or self.overwrite_czstack:
            extract_bigwarp_labels(labels_el=czs['gcamp_labels'], labels_name='gcamp_labels',
                                   output_folder=coreg, z_step_um=z_step)

    def extract_section_tifs(self):
        section_ns = sorted(
            int(p.stem.split('_')[1])
            for p in self.paths['sections_folder'].glob('section_*.zarr')
        )
        for s_n in tqdm(section_ns, desc='Sections', unit='section'):
            get_section_tifs(
                sections_folder=self.paths['sections_folder'],
                section=s_n,
                mapping_output=self.paths['mapping_output'],
                multiscale_level=self.xenium_ms_level,
                output_folder=self.paths['coregistration_folder'],
                overwrite_sections=self.overwrite_sections,
            )

    # ── Orchestration ─────────────────────────────────────────────────────

    def run(self):
        self.load_confocal()
        self.load_czstack()
        self.extract_confocal_tif()
        self.extract_czstack_tifs()
        self.extract_section_tifs()

def get_tif_um_px(tif_path):
    with tifffile.TiffFile(str(tif_path)) as tif:
        page = tif.pages[0]
        tag282 = page.tags.get(282)   # XResolution — rational (numer, denom)
        tag296 = page.tags.get(296)   # ResolutionUnit: 2=inch, 3=centimeter

        if tag282 is not None and tag282.value[0] != 0:
            numer, denom = tag282.value
            res_val = numer / denom          # pixels per unit
            res_unit = int(tag296.value) if tag296 else 2
            if res_unit == 3:                # px / cm  →  µm/px = 1e4 / (px/cm)
                px_um = 1e4 / res_val
            elif res_unit == 2:              # px / inch → µm/px = 25400 / (px/inch)
                px_um = 25400.0 / res_val
            else:
                raise ValueError(f"Unknown ResolutionUnit={res_unit} in {tif_path.name}")
        else:
            raise ValueError(f"No XResolution tag found in {tif_path.name}")
    return px_um, res_unit, res_val

def get_scale_pixel_size(el, microns_coord_sys_name='microns'):
    microns_tf = get_transformation(el, to_coordinate_system=microns_coord_sys_name)

    if isinstance(microns_tf, Scale):
        pixel_size_yx = [microns_tf.scale[microns_tf.axes.index(d)] for d in ['y', 'x']]
    elif isinstance(microns_tf, Identity):
        pixel_size_yx = [1.0, 1.0]
    elif isinstance(microns_tf, Sequence):
        py, px = 1.0, 1.0
        for tf in microns_tf.transformations:
            if isinstance(tf, Scale):
                py *= tf.scale[tf.axes.index('y')]
                px *= tf.scale[tf.axes.index('x')]
        pixel_size_yx = [py, px]
    else:
        pixel_size_yx = None
        print(f'  Warning: unhandled transform type {type(microns_tf)}, skipping calibration')
    return pixel_size_yx

def get_dim_scale(tf, dim='z'):
    if isinstance(tf, Scale) and dim in tf.axes:
        return tf.scale[list(tf.axes).index(dim)]
    if isinstance(tf, Sequence):
        scale_val = 1.0
        for t in tf.transformations:
            if isinstance(t, Scale) and dim in t.axes:
                scale_val *= t.scale[list(t.axes).index(dim)]
        return scale_val
    return None

def _rename_channel_coord(element_obj, channel_name_map=None):
    if channel_name_map is None:
        channel_name_map = {
            'DAPI': 'dapi',
            'ATP1A1/CD45/E-Cadherin': 'boundary',
            '18S': 'rna',
            'AlphaSMA/Vimentin': 'protein'
        }

    if not hasattr(element_obj, 'coords'):
        return element_obj
    if 'c' not in element_obj.coords:
        return element_obj

    old_names = [str(ch) for ch in element_obj.coords['c'].values]
    new_names = [channel_name_map.get(ch, ch) for ch in old_names]

    if old_names == new_names:
        return element_obj

    return element_obj.assign_coords(c=new_names)

def add_mapping_results(sdata, mapping_output_path, table_el='table', section_n=None):
    import anndata as ad
    from xenium_analysis_tools.map_xenium.format_mapping import (
        add_broad_types,
    )
    mapped_data = ad.read_h5ad(mapping_output_path)
    if section_n is not None:
        mapped_data = mapped_data[mapped_data.obs['section'] == section_n]
    mapped_data.obs['cell_id'] = mapped_data.obs['original_cell_id']
    mapped_data.obs.rename_axis('cell_section_id', inplace=True)
    mapped_data.var.set_index('gene_symbol', inplace=True, drop=False)
    sdata[table_el] = add_mapped_cells_cols(sdata[table_el], mapped_data, verbose=True)
    sdata[table_el] = add_broad_types(sdata[table_el])
    return sdata

def get_cell_labels(section_sdata, table_el='table', labels_el='cell_labels', multiscale_level=2, cell_filters=None):
    if 'microns' not in section_sdata.coordinate_systems:
        section_sdata = add_micron_coord_sys(section_sdata)

    if _is_multiscale(section_sdata[labels_el]):
        label_da = sd.get_pyramid_levels(section_sdata[labels_el], n=multiscale_level)
    else:
        label_da = section_sdata[labels_el]

    import dask.array as da
    label_da = label_da.copy()
    label_da.data = da.from_array(label_da.data.compute())

    table = section_sdata[table_el].copy()
    table.obs['region'] = pd.Categorical(['cell_labels'] * len(table), categories=['cell_labels'])
    table = TableModel.parse(table, region='cell_labels', region_key='region', instance_key='cell_labels', overwrite_metadata=True)

    cell_labels_sd = sd.SpatialData(tables={table_el: table}, labels={labels_el: label_da})
    if cell_filters is not None:
        cell_labels_sd = filter_cells(cell_labels_sd, cell_filters=cell_filters)
        cell_labels_sd = filter_labels(cell_labels_sd)

    return cell_labels_sd

def get_section_tifs(sections_folder, section, output_folder, mapping_output=None, multiscale_level=2, overwrite_sections=False,
                    expected_tifs = ['boundary', 'dapi', 'protein', 'rna', 'cell_labels', 'gaba_labels', 'glut_labels', 'nn_labels']):
    import warnings
    import io
    import contextlib
    section_output_folder = output_folder / f'section_{section}'
    section_output_folder.mkdir(parents=True, exist_ok=True)

    all_done = all((section_output_folder / f'{t}.tif').exists() for t in expected_tifs)
    if all_done and not overwrite_sections:
        tqdm.write(f"Section {section}: tifs already exist, skipping.")
        return

    tqdm.write(f"\n--- Section {section} ---")
    section_sdata = sd.read_zarr(sections_folder / f'section_{section}.zarr')

    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")

        if mapping_output is not None and mapping_output.exists():
            section_sdata = add_mapping_results(section_sdata, mapping_output, section_n=section)

        # Images
        if not (section_output_folder / 'morphology_focus.tif').exists() or overwrite_sections:
            extract_bigwarp_images(section_sdata,
                                   output_folder=output_folder,
                                   section_n=section)

        # Cell labels
        label_tasks = [('cell_labels', section_sdata['cell_labels'], 'cell_labels', {})]

        # Cell-type specific labels (only if mapping results available)
        if 'broad_class_name' in section_sdata['table'].obs.columns:
            for cell_type, label_name in [('Glut', 'glut_labels'), ('GABA', 'gaba_labels'), ('NN', 'nn_labels')]:
                filtered = get_cell_labels(
                    section_sdata,
                    cell_filters=[{'column': 'broad_class_name', 'operator': '==', 'value': cell_type}]
                )['cell_labels']
                label_tasks.append((label_name, filtered, label_name, {}))

    missing_labels = [t for t in label_tasks
                      if not (section_output_folder / f'{t[0]}.tif').exists() or overwrite_sections]

    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        for _, labels_el, label_name, kwargs in tqdm(missing_labels, desc=f"[section_{section}] labels", unit="label", leave=False):
            extract_bigwarp_labels(labels_el, label_name,
                                   section_n=section,
                                   output_folder=output_folder,
                                   **kwargs)

def _iter_slices_batched(dask_arr, batch_size=16):
    """Yield (iz, page) by computing `batch_size` Z-slices at once with dask.compute."""
    n_z = dask_arr.shape[0]
    dask_arr = dask_arr.rechunk({0: 1, 1: dask_arr.shape[-2], 2: dask_arr.shape[-1]})
    for start in range(0, n_z, batch_size):
        end = min(start + batch_size, n_z)
        pages = dask.compute(*[dask_arr[i] for i in range(start, end)],
                             scheduler='threads')
        for j, page in enumerate(pages):
            yield start + j, page


def extract_bigwarp_images(sdata,
                            output_folder,
                            section_n=None,
                            el_name='morphology_focus',
                            subset_channels='all',
                            multiscale_level=2,
                            dtype='uint16',
                            normalize=False,
                            z_step_um=None,
                            resunit='cm',
                            microns_coord_sys_name='microns',
                            compute_batch_size=16,
                            return_sdata=True):
    label = f'section_{section_n}' if section_n is not None else el_name
    print(f"[{label}] Starting extraction  element='{el_name}'  level={multiscale_level}  dtype={dtype}  normalize={normalize}  batch={compute_batch_size}")

    if section_n is not None:
        save_bigwarp_folder = output_folder / f'section_{section_n}'
    else:
        save_bigwarp_folder = output_folder
    save_bigwarp_folder.mkdir(exist_ok=True, parents=True)

    if not isinstance(sdata, sd.SpatialData):
        print(f"[{label}] Reading zarr from disk...")
        sdata = sd.read_zarr(sdata)
    if microns_coord_sys_name not in sdata.coordinate_systems:
        sdata = add_micron_coord_sys(sdata)
    if _is_multiscale(sdata[el_name]):
        mf_element = sd.get_pyramid_levels(sdata[el_name], n=multiscale_level)
    else:
        mf_element = sdata[el_name]
    mf_element = _rename_channel_coord(mf_element)
    if subset_channels == 'all':
        subset_channels = mf_element.coords['c'].values

    shape_str = 'x'.join(str(s) for s in mf_element.shape)
    print(f"[{label}] Array shape: {shape_str}  channels: {list(subset_channels)}")

    for ch in tqdm(subset_channels, desc=f'[{label}] channels', unit='ch', leave=True):
        out_path = save_bigwarp_folder / f'{ch}.tif'
        ch_el = mf_element.sel(c=ch)
        dims = ch_el.dims
        microns_tf = get_transformation(ch_el, to_coordinate_system=microns_coord_sys_name)

        if isinstance(microns_tf, Scale):
            pixel_size_yx = [microns_tf.scale[microns_tf.axes.index(d)] for d in ['y', 'x']]
        elif isinstance(microns_tf, Identity):
            pixel_size_yx = [1.0, 1.0]
        elif isinstance(microns_tf, Sequence):
            py, px = 1.0, 1.0
            for tf in microns_tf.transformations:
                if isinstance(tf, Scale):
                    py *= tf.scale[tf.axes.index('y')]
                    px *= tf.scale[tf.axes.index('x')]
            pixel_size_yx = [py, px]
        else:
            pixel_size_yx = None
            tqdm.write(f'  [{label}/{ch}] Warning: unhandled transform type {type(microns_tf)}, skipping calibration')

        if pixel_size_yx is not None:
            tqdm.write(f'  [{label}/{ch}] pixel size (y,x): {pixel_size_yx[0]:.4f} x {pixel_size_yx[1]:.4f} µm  shape: {"x".join(str(s) for s in ch_el.shape)}')

        dask_arr = ch_el.data
        if np.issubdtype(np.dtype(dtype), np.integer) and normalize:
            tqdm.write(f'  [{label}/{ch}] Computing min/max for normalization...')
            lo = float(da.nanmin(dask_arr).compute())
            hi = float(da.nanmax(dask_arr).compute())
            tqdm.write(f'  [{label}/{ch}] Normalizing: lo={lo:.1f}  hi={hi:.1f}')
            imax = np.iinfo(dtype).max
            def _norm_block(block, lo=lo, hi=hi, imax=imax, dtype=dtype):
                block = block.astype(np.float64)
                if hi > lo:
                    block = (block - lo) / (hi - lo) * imax
                block = np.nan_to_num(block, nan=0.0)
                return np.clip(block, 0, imax).astype(dtype)
            dask_arr = dask_arr.map_blocks(_norm_block, dtype=dtype)
        else:
            dask_arr = dask_arr.astype(dtype)

        is_3d = dask_arr.ndim >= 3
        n_z = dask_arr.shape[0] if is_3d else None

        ij_meta = {}
        if is_3d:
            # Explicitly declare Z-stack dimensions so ImageJ does not treat
            # each slice-by-slice write() call as an additional channel.
            ij_meta['slices']     = n_z
            ij_meta['channels']   = 1
            ij_meta['hyperstack'] = True
            ij_meta['mode']       = 'grayscale'
        else:
            ij_meta['axes'] = ''.join(d.upper() for d in dims)
        if z_step_um is not None:
            ij_meta['spacing'] = z_step_um
        if pixel_size_yx is not None:
            ij_meta['unit'] = 'um'

        page_kwargs = dict(metadata=ij_meta)
        if pixel_size_yx is not None:
            py, px = pixel_size_yx
            resolution_um = (1.0 / px, 1.0 / py)
            resolution_cm = (1e4 / px, 1e4 / py)
            page_kwargs['resolution'] = resolution_um if resunit == 'um' else resolution_cm
            page_kwargs['resolutionunit'] = tifffile.RESUNIT.MICROMETER if resunit == 'um' else tifffile.RESUNIT.CENTIMETER

        with tifffile.TiffWriter(str(out_path), imagej=True) as tif:
            if is_3d:
                tqdm.write(f'  [{label}/{ch}] Writing {n_z} Z-slices (batch={compute_batch_size}) → {out_path.name}')
                with tqdm(total=n_z, desc=f'    z-slices', unit='z', leave=False) as zbar:
                    for iz, page in _iter_slices_batched(dask_arr, batch_size=compute_batch_size):
                        tif.write(page, contiguous=True, **(page_kwargs if iz == 0 else {}))
                        zbar.update(1)
            else:
                tile_h = min(512, dask_arr.shape[0])
                tile_w = min(512, dask_arr.shape[1])
                rechunked = dask_arr.rechunk((tile_h, tile_w))
                n_tiles = rechunked.numblocks[0] * rechunked.numblocks[1]
                tqdm.write(f'  [{label}/{ch}] Writing 2D tiled TIFF ({n_tiles} tiles {tile_h}x{tile_w}) → {out_path.name}')

                all_blocks = [(cy, cx)
                              for cy in range(rechunked.numblocks[0])
                              for cx in range(rechunked.numblocks[1])]

                def _iter_tiles_batched():
                    for i in range(0, len(all_blocks), compute_batch_size):
                        batch = all_blocks[i:i + compute_batch_size]
                        pages = dask.compute(*[rechunked.blocks[cy, cx] for cy, cx in batch],
                                             scheduler='threads')
                        yield from pages

                tif.write(
                    _iter_tiles_batched(),
                    shape=dask_arr.shape,
                    dtype=dask_arr.dtype,
                    tile=(tile_h, tile_w),
                    **page_kwargs,
                )

        tqdm.write(f'  [{label}/{ch}] ✓ Saved {out_path.name}  ({out_path.stat().st_size / 1e6:.1f} MB)')

    print(f"[{label}] Done. {len(subset_channels)} channel(s) written to {save_bigwarp_folder}")
    if return_sdata:
        return sdata
    
def extract_bigwarp_labels(
    labels_el,
    labels_name,
    output_folder,
    multiscale_level=2,
    section_n=None,
    microns_coord_sys_name='microns',
    dtype='uint8',
    binary=True,
    z_step_um=None,
    resunit='cm'
):
    if section_n is None:
        save_bigwarp_folder = output_folder
    else:
        save_bigwarp_folder = output_folder / f'section_{section_n}'
    out_path = save_bigwarp_folder / f'{labels_name}.tif'

    # Get the labels DataArray — handle SpatialData objects and raw DataArrays
    if isinstance(labels_el, sd.SpatialData):
        el_name = list(labels_el.labels.keys())[0]
        el = labels_el.labels[el_name]
    else:
        el = labels_el
    if _is_multiscale(el):
        el = sd.get_pyramid_levels(el, n=multiscale_level)

    dims = el.dims
    microns_tf = get_transformation(el, to_coordinate_system=microns_coord_sys_name)

    if isinstance(microns_tf, Scale):
        pixel_size_yx = [microns_tf.scale[microns_tf.axes.index(d)] for d in ['y', 'x']]
    elif isinstance(microns_tf, Identity):
        pixel_size_yx = [1.0, 1.0]
    elif isinstance(microns_tf, Sequence):
        py, px = 1.0, 1.0
        for tf in microns_tf.transformations:
            if isinstance(tf, Scale):
                py *= tf.scale[tf.axes.index('y')]
                px *= tf.scale[tf.axes.index('x')]
        pixel_size_yx = [py, px]
    else:
        pixel_size_yx = None
        print(f'  Warning: unhandled transform type {type(microns_tf)}, skipping calibration')

    dask_arr = el.data
    is_3d = dask_arr.ndim >= 3

    if binary:
        dask_arr = dask_arr.map_blocks(
            lambda b: np.where(b > 0, np.iinfo('uint16').max, 0).astype('uint16'),
            dtype='uint16',
        )
    else:
        dask_arr = dask_arr.astype(dtype, copy=False)

    ij_meta = {}
    if is_3d:
        # Explicitly declare Z-stack dimensions so ImageJ does not treat
        # each slice-by-slice write() call as an additional channel.
        ij_meta['slices']     = dask_arr.shape[0]
        ij_meta['channels']   = 1
        ij_meta['hyperstack'] = True
        ij_meta['mode']       = 'grayscale'
    else:
        ij_meta['axes'] = ''.join(d.upper() for d in dims)
    if z_step_um is not None:
        ij_meta['spacing'] = z_step_um
    if pixel_size_yx is not None:
        ij_meta['unit'] = 'um'

    page_kwargs = dict(metadata=ij_meta)
    if pixel_size_yx is not None:
        py, px = pixel_size_yx
        resolution_cm = (1e4 / px, 1e4 / py)
        resolution_um = (1.0 / px, 1.0 / py)
        page_kwargs['resolution'] = resolution_um if resunit == 'um' else resolution_cm
        page_kwargs['resolutionunit'] = tifffile.RESUNIT.MICROMETER if resunit == 'um' else tifffile.RESUNIT.CENTIMETER

    save_bigwarp_folder.mkdir(exist_ok=True, parents=True)
    with tifffile.TiffWriter(str(out_path), imagej=True) as tif:
        if is_3d:
            n_z = dask_arr.shape[0]
            print(f'  [{labels_name}] Writing {n_z} Z-slices → {out_path.name}')
            for iz, page in _iter_slices_batched(dask_arr, batch_size=16):
                tif.write(page, contiguous=True, **(page_kwargs if iz == 0 else {}))
        else:
            arr = dask_arr.compute()
            tif.write(arr, **page_kwargs)

    print(f'  Wrote: {out_path.name}  shape={dask_arr.shape}  binary={binary}')

def save_composite_ome_tiff(arr, out_path, channel_names, pixel_size_yx=None, z_step_um=None):
    """Save a (C, Z, Y, X) or (C, Y, X) array as an OME-TIFF with proper channel metadata.

    photometric is intentionally omitted for multi-channel data so tifffile
    picks the correct value per channel rather than forcing RGB.
    """
    axes = 'CZYX' if arr.ndim == 4 else 'CYX'
    ome_metadata = {
        'axes': axes,
        'Channel': {'Name': channel_names},
    }
    if pixel_size_yx is not None:
        ome_metadata['PhysicalSizeX']     = pixel_size_yx[1]
        ome_metadata['PhysicalSizeY']     = pixel_size_yx[0]
        ome_metadata['PhysicalSizeXUnit'] = 'µm'
        ome_metadata['PhysicalSizeYUnit'] = 'µm'
    if z_step_um is not None:
        ome_metadata['PhysicalSizeZ']     = z_step_um
        ome_metadata['PhysicalSizeZUnit'] = 'µm'

    tifffile.imwrite(str(out_path), arr, metadata=ome_metadata)
    print(f'Wrote OME-TIFF: {out_path.name}  shape={arr.shape}  axes={axes}')

def build_bigwarp_project(
    out_json,
    moving_channel_config,
    fixed_channel_config,
    n_dims=3,
    transform_type='Affine',
):
    """Build a BigWarp project JSON that loads each channel from its own TIFF.

    Loading via project JSON (rather than Fiji Merge Channels) preserves the Z
    dimension correctly for 3D stacks — BigWarp reads each TIFF independently.

    Parameters
    ----------
    out_json             : Path
    moving_channel_config: list[dict]  — each dict has keys: file, name, color, min, max
    fixed_channel_config : list[dict]  — same
    n_dims               : int         — 3 for 3D↔3D (Stage 1), 2 for 3D→2D (Stage 2)
    transform_type       : str         — 'Affine' or 'Thin Plate Spline'
    """
    n_moving  = len(moving_channel_config)
    n_fixed   = len(fixed_channel_config)
    n_total   = n_moving + n_fixed
    all_ch    = moving_channel_config + fixed_channel_config

    sources = {}
    for i, ch in enumerate(moving_channel_config):
        sources[str(i)] = {
            'uri':      str(ch['file']),
            'name':     ch['name'],
            'isMoving': True,
        }
    for i, ch in enumerate(fixed_channel_config):
        sources[str(n_moving + i)] = {
            'uri':      str(ch['file']),
            'name':     ch['name'],
            'isMoving': False,
        }

    converter_setups = {
        str(i): {
            'min':     float(ch.get('min', 0.0)),
            'max':     float(ch.get('max', 65535.0)),
            'color':   COLOR_MAP.get(ch['color'], -1),
            'groupId': i,
        }
        for i, ch in enumerate(all_ch)
    }

    minmax_groups = {
        str(i): {
            'fullRangeMin': -2147483648.0,
            'fullRangeMax':  2147483647.0,
            'rangeMin':      0.0,
            'rangeMax':      65535.0,
            'currentMin': float(ch.get('min', 0.0)),
            'currentMax': float(ch.get('max', 65535.0)),
        }
        for i, ch in enumerate(all_ch)
    }

    moving_ids     = list(range(n_moving))
    fixed_ids      = list(range(n_moving, n_total))
    viewer_sources = [True] * n_total

    def viewer(current_group):
        return {
            'Sources': viewer_sources,
            'SourceGroups': [
                {'active': True, 'name': 'moving images', 'id': moving_ids},
                {'active': True, 'name': 'target images', 'id': fixed_ids},
            ],
            'DisplayMode':     'sg',
            'Interpolation':   'nearestneighbor',
            'CurrentSource':   0,
            'CurrentGroup':    current_group,
            'CurrentTimePoint': 0,
        }

    project = {
        'Sources': sources,
        'ViewerP': viewer(0),
        'ViewerQ': viewer(1),
        'SetupAssignments': {
            'ConverterSetups': converter_setups,
            'MinMaxGroups':    minmax_groups,
        },
        'Bookmarks': {'bookmarks': {}},
        'Transform': {
            'type': transform_type,
            'landmarks': {
                'type':          'BigWarpLandmarks',
                'numDimensions': n_dims,
                'movingPoints':  [],
                'fixedPoints':   [],
                'active':        [],
                'names':         [],
            },
        },
    }

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(project, f, indent=4)
    print(f'Wrote BigWarp project JSON: {out_json.name}  '
          f'(n_dims={n_dims}, transform={transform_type})')
    return out_json
    
def save_ij_macro(macro_path, project_json, tif_paths=None):
    """Write a Fiji macro that launches BigWarp via a project JSON.

    When a project JSON is provided, BigWarp opens all source TIFFs itself
    using BDV's lazy/virtual stack reader. Do NOT pre-open the TIFFs with
    open() before calling BigWarp — this forces Fiji to load the entire stack
    into heap memory at once, causing OutOfMemoryError on large volumes and
    a downstream TransformedSource->WarpedSource ClassCastException.

    tif_paths is accepted but ignored when project_json is provided,
    kept only for API compatibility.

    Parameters
    ----------
    macro_path   : Path
    project_json : Path  — BigWarp project JSON (contains all source URIs)
    tif_paths    : ignored when project_json provided
    """
    macro_path   = Path(macro_path)
    project_json = Path(project_json)

    # BigWarp reads source TIFFs lazily from the URIs in the project JSON.
    # No open() calls needed or wanted.
    macro_text = f'run("Big Warp", "project=[{project_json}]");\n'

    macro_path.parent.mkdir(parents=True, exist_ok=True)
    with open(macro_path, 'w') as f:
        f.write(macro_text)
    print(f'Saved macro: {macro_path.name}')
    print(f'  -> BigWarp will load sources from: {project_json.name}')
    return macro_path


def _landmarks_from_csv(csv_path):
    """Parse a BigWarp landmarks CSV into the standard list-of-dicts format.

    Expected CSV format (no header, 8 columns for 3D / 6 for 2D):
        "name", "active", moving_x, moving_y[, moving_z], fixed_x, fixed_y[, fixed_z]
    """
    landmarks = []
    with open(csv_path, newline='') as fh:
        import csv
        for row in csv.reader(fh):
            row = [c.strip() for c in row]
            if len(row) < 6:
                continue
            name   = row[0].strip('"')
            active = row[1].strip('"').lower() == 'true'
            if len(row) >= 8:
                moving = [float(row[2]), float(row[3]), float(row[4])]
                fixed  = [float(row[5]), float(row[6]), float(row[7])]
            else:  # 2D: name, active, mx, my, fx, fy
                moving = [float(row[2]), float(row[3])]
                fixed  = [float(row[4]), float(row[5])]
            landmarks.append({'name': name, 'active': active, 'moving': moving, 'fixed': fixed})
    return landmarks


def _get_tif_shape(tif_path):
    """Return (n_z, n_y, n_x) or (n_y, n_x) from a TIFF file."""
    with tifffile.TiffFile(str(tif_path)) as tif:
        if tif.series:
            return tif.series[0].shape
        n_pages = len(tif.pages)
        h, w    = tif.pages[0].shape[:2]
        return (n_pages, h, w) if n_pages > 1 else (h, w)


def _detect_landmark_pixel_size(coords, tif_path):
    """Detect whether landmark coordinates are in physical (µm) or pixel space.

    BigWarp exports landmark CSVs in the world coordinate space of the loaded
    image. When a TIFF has XResolution/ResolutionUnit metadata embedded,
    BigWarp world space == physical (micron) space. If the TIFF has no
    calibration the coordinates are in pixels.

    When landmarks from one session (physical space) are injected into a new
    project, the coordinates must be converted back to pixels so Fiji
    displays them on the correct voxel.

    Heuristic
    ---------
    If ``max(|coord|)`` is far larger than the image's pixel extent but
    consistent with the image's micron extent, the coords are in µm and the
    returned scale factor is ``px_um`` (divide coords by this to get pixels).
    Otherwise returns 1.0 (no scaling needed).

    Parameters
    ----------
    coords   : flat iterable of coordinate values (moving OR fixed — not mixed)
    tif_path : Path to the corresponding TIFF

    Returns
    -------
    float — px_um if coords appear physical, 1.0 if already in pixel space
    """
    tif_path = Path(tif_path)
    if not tif_path.exists():
        return 1.0

    try:
        px_um, _, _ = get_tif_um_px(tif_path)
    except (ValueError, KeyError):
        return 1.0

    if px_um is None or px_um <= 0:
        return 1.0

    try:
        shape = _get_tif_shape(tif_path)
    except Exception:
        return 1.0

    coords = [c for c in coords if c is not None]
    if not coords:
        return 1.0

    max_coord   = max(abs(c) for c in coords)
    # Largest spatial extent in pixels (ignore z — it may be tiny)
    max_px      = max(shape[-1], shape[-2])
    max_um      = max_px * px_um

    ratio_to_px = max_coord / max_px if max_px > 0 else 0
    ratio_to_um = max_coord / max_um if max_um > 0 else 0

    # If coords are way outside pixel bounds but fit within micron bounds,
    # they are in physical space.
    if ratio_to_px > 4.0 and 0.0 < ratio_to_um < 4.0:
        print(f'  Auto-detected physical-space landmarks '
              f'(max coord={max_coord:.1f}, image={max_px}px / {max_um:.1f}µm). '
              f'  Scaling by 1/{px_um:.4f} to convert to pixels.')
        return px_um

    return 1.0


def _scale_coords(pts, scale):
    """Divide each point's coordinates by scale (in-place safe, returns new list)."""
    if scale == 1.0:
        return pts
    return [[c / scale for c in pt] for pt in pts]


def inject_landmarks(project_json_path, landmarks,
                     moving_tif=None, fixed_tif=None,
                     auto_scale=True):
    """Inject pre-existing BigWarp landmarks into a project JSON.

    ``landmarks`` can be either:
      - a list of dicts, each with keys 'name', 'active', 'moving', 'fixed'
        (where 'moving' and 'fixed' are [x, y, z] lists), or
      - a path (str or Path) to a BigWarp landmarks CSV with the format:
            "name","active",moving_x,moving_y,moving_z,fixed_x,fixed_y,fixed_z

    BigWarp requires points as an array-of-arrays, NOT a flat list:
        CORRECT: movingPoints = [[x0,y0,z0], [x1,y1,z1], ...]
        WRONG:   movingPoints = [x0,y0,z0, x1,y1,z1, ...]  <- causes
                     'Not a JSON Array' IllegalStateException on load.

    Coordinate-space auto-scaling
    ------------------------------
    BigWarp saves CSVs in *world* coordinates. When a TIFF has pixel-size
    metadata embedded, world space == micron space. Injecting micron-space
    landmarks into a fresh project causes them to appear far outside the
    image bounds in Fiji.

    If ``auto_scale=True`` (default) and ``moving_tif`` / ``fixed_tif`` are
    provided, the function detects this mismatch and divides each coordinate
    set by the TIFF's µm/px value so they land on the correct voxel.

    Parameters
    ----------
    project_json_path : Path
    landmarks         : list[dict] or CSV path
    moving_tif        : Path, optional — TIFF for the moving image; used for
                        auto-detection of physical-space moving coords
    fixed_tif         : Path, optional — TIFF for the fixed image; same
    auto_scale        : bool — enable physical→pixel auto-scaling (default True)
    """
    import os
    if isinstance(landmarks, (str, os.PathLike)):
        lm_path = Path(landmarks)
        if not lm_path.exists():
            print(f'  Landmarks file not found, skipping: {lm_path}')
            return
        landmarks = _landmarks_from_csv(lm_path)

    if not landmarks:
        print('  No landmarks to inject.')
        return

    moving_pts = [lm['moving'] for lm in landmarks]
    fixed_pts  = [lm['fixed']  for lm in landmarks]

    if auto_scale:
        if moving_tif is not None:
            moving_flat = [c for pt in moving_pts for c in pt]
            scale_m = _detect_landmark_pixel_size(moving_flat, moving_tif)
            moving_pts = _scale_coords(moving_pts, scale_m)

        if fixed_tif is not None:
            fixed_flat = [c for pt in fixed_pts for c in pt]
            scale_f = _detect_landmark_pixel_size(fixed_flat, fixed_tif)
            fixed_pts = _scale_coords(fixed_pts, scale_f)

    project_json_path = Path(project_json_path)
    with open(project_json_path) as f:
        proj = json.load(f)

    proj['Transform']['landmarks'].update({
        'movingPoints': moving_pts,
        'fixedPoints':  fixed_pts,
        'active':       [lm['active'] for lm in landmarks],
        'names':        [lm['name']   for lm in landmarks],
    })

    with open(project_json_path, 'w') as f:
        json.dump(proj, f, indent=4)
    print(f'Injected {len(landmarks)} landmarks into {project_json_path.name}')
    print(f'  Format check — movingPoints[0]: {proj["Transform"]["landmarks"]["movingPoints"][0]}')






def fix_imagej_zstack_tif(tif_path, pixel_size_um=None, z_step_um=None):
    """Audit and fix ImageJ TIFF metadata for a single-channel Z-stack.

    Checks and corrects in the ImageDescription tag:
      1. channels=N  ->  slices=N, channels=1   (main Z-dimension bug from
         extract_bigwarp_images writing slices one-by-one without explicit dims)
      2. unit field present and = 'um'
      3. spacing field present                  (if z_step_um provided)
      4. hyperstack = true
      5. mode = grayscale

    And in the raw TIFF IFD tags:
      6. ResolutionUnit = CENTIMETER (3)        (if pixel_size_um provided)
      7. XResolution / YResolution = correct    (if pixel_size_um provided)

    Patches entirely in-place (binary seeks only — no pixel data read or written).
    Also updates the IFD count field for the ImageDescription tag so the full
    patched string is read back correctly by tifffile and Fiji.

    Safe to run multiple times: detects already-correct files and skips them.

    Parameters
    ----------
    tif_path      : Path  — TIFF to fix
    pixel_size_um : float — XY pixel size in microns (fixes resolution tags if wrong)
    z_step_um     : float — Z step size in microns (adds spacing= field if missing)

    Raises
    ------
    RuntimeError if the patched description string exceeds the space allocated
    by tifffile in the original file (extremely unlikely; tifffile over-allocates).
    """
    tif_path = Path(tif_path)

    with tifffile.TiffFile(str(tif_path)) as tif:
        if not tif.is_imagej:
            print(f'  SKIP {tif_path.name}: not an ImageJ TIFF')
            return False

        page0  = tif.pages[0]
        meta   = tif.imagej_metadata or {}
        n_ch   = meta.get('channels', 1)
        n_sl   = meta.get('slices',   1)

        tag270 = page0.tags.get(270)   # ImageDescription
        tag282 = page0.tags.get(282)   # XResolution
        tag283 = page0.tags.get(283)   # YResolution
        tag296 = page0.tags.get(296)   # ResolutionUnit

        if tag270 is None:
            print(f'  SKIP {tif_path.name}: no ImageDescription tag')
            return False

        desc_orig      = tag270.value
        desc_val_off   = tag270.valueoffset   # byte offset of the string data
        desc_count_off = tag270.offset + 4    # count field in the 12-byte IFD entry

        # Available space = gap between desc bytes and next data block
        later     = [t.valueoffset for t in page0.tags.values() if t.valueoffset > desc_val_off]
        available = (min(later) - desc_val_off) if later else None

        res_x         = tag282.value if tag282 else None
        res_x_off     = tag282.valueoffset if tag282 else None
        res_y_off     = tag283.valueoffset if tag283 else None
        res_u_off     = tag296.valueoffset if tag296 else None
        res_unit_int  = int(tag296.value) if tag296 else None

    # ── Determine what needs fixing ──────────────────────────────────────────
    issues = []

    needs_dim_fix = (n_ch > 1 and n_sl == 1)
    n_z = n_ch if needs_dim_fix else n_sl
    if needs_dim_fix:
        issues.append(f'channels={n_ch}->1, slices={n_z}')

    needs_res_fix  = False
    correct_res_cm = None
    if pixel_size_um is not None:
        correct_res_cm  = 1e4 / pixel_size_um
        current_res_val = (res_x[0] / res_x[1]) if (res_x and res_x[1]) else 1.0
        if res_unit_int != 3 or abs(current_res_val - correct_res_cm) > 1.0:
            needs_res_fix = True
            issues.append(f'resolution (unit={res_unit_int}->3, val->{correct_res_cm:.1f} px/cm)')

    # ── Build corrected ImageDescription string ──────────────────────────────
    desc_new = desc_orig

    if needs_dim_fix:
        desc_new = re.sub(r'channels=\d+', 'channels=1', desc_new)
        if 'slices=' in desc_new:
            desc_new = re.sub(r'slices=\d+', f'slices={n_z}', desc_new)
        else:
            desc_new = re.sub(r'(images=\d+)', rf'\1\nslices={n_z}', desc_new)

    if 'unit=' not in desc_new:
        desc_new += '\nunit=um'
        issues.append('unit=um added')
    elif not re.search(r'unit=um\b', desc_new):
        desc_new = re.sub(r'unit=\S+', 'unit=um', desc_new)
        issues.append('unit->um')

    if z_step_um is not None and 'spacing=' not in desc_new:
        desc_new += f'\nspacing={z_step_um}'
        issues.append(f'spacing={z_step_um}')

    if 'hyperstack=' not in desc_new:
        desc_new += '\nhyperstack=true'
        issues.append('hyperstack=true added')
    elif not re.search(r'hyperstack=true\b', desc_new):
        desc_new = re.sub(r'hyperstack=\S+', 'hyperstack=true', desc_new)
        issues.append('hyperstack->true')

    if 'mode=' not in desc_new:
        desc_new += '\nmode=grayscale'
        issues.append('mode=grayscale added')
    elif not re.search(r'mode=gra?yscale\b', desc_new):
        desc_new = re.sub(r'mode=\S+', 'mode=grayscale', desc_new)
        issues.append('mode->grayscale')

    if not issues:
        print(f'  OK   {tif_path.name}: all metadata correct')
        return False

    print(f'  FIX  {tif_path.name}: {";".join(issues)}')

    # ── Encode and check space ───────────────────────────────────────────────
    new_b = desc_new.encode('latin-1') + b'\x00'

    if available is not None and len(new_b) > available:
        raise RuntimeError(
            f'{tif_path.name}: patched ImageDescription ({len(new_b)} B) exceeds '
            f'allocated space ({available} B). Rewrite the file instead.'
        )

    # Pad to fill the full allocated region cleanly
    fill     = available if available else len(new_b)
    new_b_padded = new_b.ljust(fill, b'\x00')

    with open(str(tif_path), 'r+b') as fh:
        # 1. Write patched ImageDescription bytes
        fh.seek(desc_val_off)
        fh.write(new_b_padded)

        # 2. Update the IFD count field (4-byte LE LONG) so tifffile reads
        #    the full new string, not just the original character count
        fh.seek(desc_count_off)
        fh.write(struct.pack('<I', len(new_b)))

        # 3. Patch ResolutionUnit and Resolution rational tags if needed
        if needs_res_fix and correct_res_cm is not None:
            fh.seek(res_u_off)
            fh.write(struct.pack('<H', 3))          # ResolutionUnit = CENTIMETER
            numer     = round(correct_res_cm * 10000)
            res_bytes = struct.pack('<II', numer, 10000)
            fh.seek(res_x_off); fh.write(res_bytes) # XResolution
            fh.seek(res_y_off); fh.write(res_bytes) # YResolution

    # ── Verify round-trip ────────────────────────────────────────────────────
    with tifffile.TiffFile(str(tif_path)) as tif:
        m    = tif.imagej_metadata
        ax   = tif.series[0].axes
        r282 = tif.pages[0].tags.get(282)
        r296 = tif.pages[0].tags.get(296)
        px_rb = None
        if r282 and r282.value[0]:
            px_rb = (r282.value[1] / r282.value[0]) * 1e4
        print(f'       axes={ax}  slices={m.get("slices")}  channels={m.get("channels")}  '
              f'unit={m.get("unit")}  spacing={m.get("spacing")}  '
              f'hyperstack={m.get("hyperstack")}  mode={m.get("mode")}')
        if px_rb:
            print(f'       resolution: {px_rb:.5f} um/px  resunit={int(r296.value) if r296 else "?"}')
    return True