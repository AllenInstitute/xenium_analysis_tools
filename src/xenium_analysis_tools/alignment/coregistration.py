import numpy as np
import pandas as pd
import spatialdata as sd
import tifffile
from IPython.display import display
from tqdm.notebook import tqdm

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

    # Compute to numpy so filter_labels' in-place assignment works
    # (dask arrays silently ignore item assignment)
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

def extract_bigwarp_images(sdata, 
                                       bigwarp_projects_folder,
                                        section_n=None, 
                                        el_name = 'morphology_focus',
                                        subset_channels = 'all',
                                        multiscale_level = 2,
                                        dtype='uint16',
                                        normalize=False,
                                        z_step_um=None,
                                        resunit = 'cm',
                                        microns_coord_sys_name = 'microns',
                                        save_name=None,
                                        return_sdata=True):
    if section_n is not None:
        save_bigwarp_folder = bigwarp_projects_folder / f'section_{section_n}'
        save_bigwarp_folder.mkdir(exist_ok=True, parents=True)
        out_path = None
    else:
        save_bigwarp_folder = bigwarp_projects_folder
        save_bigwarp_folder.mkdir(exist_ok=True, parents=True)
    if not isinstance(sdata, sd.SpatialData):
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

    for ch in tqdm(subset_channels, desc=f'Extracting channels'):
        out_path = save_bigwarp_folder / f'{ch}.tif'
        ch_el = mf_element.sel(c=ch)
        dims = ch_el.dims
        microns_tf = get_transformation(ch_el, to_coordinate_system=microns_coord_sys_name)

        # Get scale factors for Y and X dimensions
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
            print(f'  Warning: unhandled transform type {type(microns_tf)} for channel {ch}, skipping calibration')

        arr = ch_el.data.compute()
        if np.issubdtype(np.dtype(dtype), np.integer) and normalize:
            finite = arr[np.isfinite(arr)]
            lo, hi = finite.min(), finite.max()
            if hi > lo:
                arr = (arr.astype(np.float64) - lo) / (hi - lo) * np.iinfo(dtype).max
            arr = np.nan_to_num(arr, nan=0.0)
            arr = np.clip(arr, 0, np.iinfo(dtype).max)
        arr = arr.astype(dtype, copy=False)

        ij_meta = {'axes': ''.join(d.upper() for d in dims)}
        if 'z' in dims and z_step_um is not None:
            ij_meta['spacing'] = z_step_um
        if pixel_size_yx is not None:
            ij_meta['unit'] = 'um'

        # Resolution tag: pixels per micron (for XY)
        kwargs = dict(imagej=True, metadata=ij_meta)
        if pixel_size_yx is not None:
            py, px = pixel_size_yx
            resolution_um = (1.0 / px, 1.0 / py)
            resolution_cm = (1e4 / px, 1e4 / py)
            kwargs['resolution'] = resolution_um if resunit == 'um' else resolution_cm
            kwargs['resolutionunit'] = tifffile.RESUNIT.MICROMETER if resunit == 'um' else tifffile.RESUNIT.CENTIMETER

        tifffile.imwrite(str(out_path), arr, **kwargs)

    if return_sdata:
        return sdata
    
def extract_bigwarp_labels(
    labels_el,
    labels_name,
    bigwarp_projects_folder,
    section_n=None,
    microns_coord_sys_name='microns',
    dtype='uint8',
    binary=True,
    z_step_um=None,
    resunit='cm'
):
    if section_n is None:
        save_bigwarp_folder = bigwarp_projects_folder
    else:
        save_bigwarp_folder = bigwarp_projects_folder / f'section_{section_n}'
    out_path = save_bigwarp_folder / f'{labels_name}.tif'

    # Get the labels DataArray — handle SpatialData objects and raw DataArrays
    if isinstance(labels_el, sd.SpatialData):
        el_name = list(labels_el.labels.keys())[0]
        el = labels_el.labels[el_name]
    else:
        el = labels_el
    if _is_multiscale(el):
        el = sd.get_pyramid_levels(el, n=2)

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

    arr = el.data.compute()
    if binary:
        arr = np.where(arr > 0, np.iinfo('uint16').max, 0).astype('uint16')
    else:
        arr = arr.astype(dtype, copy=False)

    ij_meta = {'axes': ''.join(d.upper() for d in dims)}
    if 'z' in dims and z_step_um is not None:
        ij_meta['spacing'] = z_step_um
    if pixel_size_yx is not None:
        ij_meta['unit'] = 'um'

    kwargs = dict(imagej=True, metadata=ij_meta)
    if pixel_size_yx is not None:
        py, px = pixel_size_yx
        resolution_cm = (1e4 / px, 1e4 / py)
        resolution_um = (1.0 / px, 1.0 / py)
        kwargs['resolution'] = resolution_um if resunit == 'um' else resolution_cm
        kwargs['resolutionunit'] = tifffile.RESUNIT.MICROMETER if resunit == 'um' else tifffile.RESUNIT.CENTIMETER

    save_bigwarp_folder.mkdir(exist_ok=True, parents=True)
    tifffile.imwrite(str(out_path), arr, **kwargs)
    print(f'  Wrote: {out_path.name}  shape={arr.shape}  binary={binary}')
