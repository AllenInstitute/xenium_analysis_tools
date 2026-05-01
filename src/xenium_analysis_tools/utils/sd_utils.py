
from spatialdata.transformations import Scale, Identity, set_transformation, get_transformation
from spatialdata.models import Image3DModel
from dask.callbacks import Callback
import spatialdata as sd
import xarray as xr
import numpy as np
import pandas as pd
import json
from pathlib import Path
from geopandas import GeoDataFrame
from spatialdata._io._utils import _resolve_zarr_store
from tqdm.notebook import tqdm as tqdm_nb
import time
import dask
import zarr
import xml.etree.ElementTree as ET
import tifffile

from xenium_analysis_tools.utils.env import detect_env, get_datasets_json_path

def get_dataset_paths(
    dataset_id,
    alignment_folder_parent='scratch',
    data_root=None,
    scratch_root=None,
    results_root=None,
    code_root=None,
    datasets_json_path=None,
    confocal_surface_name='surface',
    create_folders=False,
    confocal_path=None,
    raw_confocal_path=None,
    gcamp_image_path=None,
    gcamp_masks_path=None,
    dextran_image_path=None,
    dextran_masks_path=None,
    sections_folder=None,
    mapping_output=None,
):
    """Return resolved dataset paths from xenium_datasets.json.

    Root paths default to detect_env() values (Code Ocean aware).
    Any individual path override always wins over the JSON-derived value.

    Parameters
    ----------
    dataset_id         : str | int  — dataset key in the JSON
    data_root          : Path, optional — override data root
    scratch_root       : Path, optional — override scratch root
    results_root       : Path, optional — override results root
    code_root          : Path, optional — override code root
    datasets_json_path : Path, optional — explicit path to xenium_datasets.json;
                         falls back to code_root/xenium_datasets.json, then the
                         bundled package copy
    confocal_surface_name : str — name of the confocal surface zarr (default 'surface')
    create_folders     : bool — create alignment/coregistration folders if missing
    confocal_path / raw_confocal_path / gcamp_image_path / gcamp_masks_path /
    dextran_image_path / dextran_masks_path / sections_folder / mapping_output
                       : Path, optional — override any specific resolved path
    """
    # ── Resolve root paths ────────────────────────────────────────────────
    env = detect_env()
    data_root    = Path(data_root)    if data_root    is not None else env['data_root']
    scratch_root = Path(scratch_root) if scratch_root is not None else env['scratch_root']
    results_root = Path(results_root) if results_root is not None else env['results_root']
    code_root    = Path(code_root)    if code_root    is not None else env['code_root']

    # ── Locate xenium_datasets.json ───────────────────────────────────────
    if datasets_json_path is not None:
        datasets_json_path = Path(datasets_json_path)
    else:
        _candidate = code_root / 'xenium_datasets.json'
        datasets_json_path = _candidate if _candidate.exists() else get_datasets_json_path()

    with open(datasets_json_path) as f:
        datasets = json.load(f)

    dataset_id = str(dataset_id)
    if dataset_id not in datasets:
        raise KeyError(f"Dataset ID '{dataset_id}' not found in {datasets_json_path}")

    cfg          = datasets[dataset_id]
    dataset_info = cfg.get('dataset_info', {})
    dataset_paths_cfg = cfg.get('paths', {})

    # ── Helpers ───────────────────────────────────────────────────────────
    def _nested_get(dct, *keys, default=None):
        cur = dct
        for key in keys:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(key)
            if cur is None:
                return default
        return cur

    def _rp(base, rel):
        if base is None or rel in (None, ''):
            return None
        rel = Path(rel)
        return rel if rel.is_absolute() else base / rel

    # ── Derive paths from JSON ────────────────────────────────────────────
    xenium_dataset_name = dataset_info.get('xenium_name')

    _confocal_cfg  = dataset_paths_cfg.get('confocal', {})
    _raw_cf        = _rp(data_root, _confocal_cfg.get('raw_folder'))
    _cf_processed  = _rp(data_root, _confocal_cfg.get('processed_folder'))
    _cf            = (_cf_processed / f'{confocal_surface_name}.zarr') if _cf_processed else None

    _czstack_cfg   = dataset_paths_cfg.get('cortical_zstack', {})
    _zstack_path   = _rp(data_root, _czstack_cfg.get('image_folder'))
    _zstack_masks  = _rp(data_root, _czstack_cfg.get('masks_folder'))
    _gcamp_img     = _rp(_zstack_path,  _nested_get(_czstack_cfg, 'gcamp',   'img_tif_path'))
    _gcamp_masks   = _rp(_zstack_masks, _nested_get(_czstack_cfg, 'gcamp',   'masks_tif_path'))
    _dextran_img   = _rp(_zstack_path,  _nested_get(_czstack_cfg, 'dextran', 'img_tif_path'))
    _dextran_masks = _rp(_zstack_masks, _nested_get(_czstack_cfg, 'dextran', 'masks_tif_path'))

    _xenium_cfg    = dataset_paths_cfg.get('xenium', {})
    _sdata_path    = _rp(data_root, _xenium_cfg.get('processed_data'))
    _mapping       = _rp(data_root, _xenium_cfg.get('mapping'))
    if xenium_dataset_name is None:
        xenium_dataset_name = _xenium_cfg.get('name')

    # ── Apply per-path overrides ─────────────────────────────────────────
    confocal_path      = Path(confocal_path)      if confocal_path      is not None else _cf
    raw_confocal_path  = Path(raw_confocal_path)  if raw_confocal_path  is not None else _raw_cf
    gcamp_image_path   = Path(gcamp_image_path)   if gcamp_image_path   is not None else _gcamp_img
    gcamp_masks_path   = Path(gcamp_masks_path)   if gcamp_masks_path   is not None else _gcamp_masks
    dextran_image_path = Path(dextran_image_path) if dextran_image_path is not None else _dextran_img
    dextran_masks_path = Path(dextran_masks_path) if dextran_masks_path is not None else _dextran_masks
    sections_folder    = Path(sections_folder)    if sections_folder    is not None else _sdata_path
    mapping_output     = Path(mapping_output)     if mapping_output     is not None else _mapping

    # ── Section list ──────────────────────────────────────────────────────
    section_ns = []
    if sections_folder is not None and sections_folder.exists():
        for sp in sections_folder.glob('section_*.zarr'):
            parts = sp.stem.split('_')
            if len(parts) > 1 and parts[1].isdigit():
                section_ns.append(int(parts[1]))
        section_ns = sorted(section_ns)

    # ── Alignment folders ─────────────────────────────────────────────────
    if alignment_folder_parent == 'scratch':
        alignment_parent = scratch_root
    elif alignment_folder_parent == 'result':
        alignment_parent = results_root
    else:
        alignment_parent = Path.cwd()

    alignment_folder       = alignment_parent / f'xenium_{dataset_id}_alignment'
    coregistration_folder  = alignment_folder / 'coregistration'
    if create_folders:
        alignment_folder.mkdir(exist_ok=True, parents=True)
        coregistration_folder.mkdir(exist_ok=True, parents=True)

    return {
        'dataset_id':            dataset_id,
        'dataset_info':          dataset_info,
        'data_root':             data_root,
        'scratch_root':          scratch_root,
        'results_root':          results_root,
        'code_root':             code_root,
        'xenium_dataset_name':   xenium_dataset_name,
        'sdata_path':            sections_folder,
        'sections_folder':       sections_folder,
        'section_ns':            section_ns,
        'mapping_output':        mapping_output,
        'confocal_path':         confocal_path,
        'raw_confocal_path':     raw_confocal_path,
        'zstack_path':           _zstack_path,
        'zstack_masks':          _zstack_masks,
        'zstack_img_gcamp_path': gcamp_image_path,
        'zstack_masks_gcamp_path': gcamp_masks_path,
        'zstack_img_dextran_path': dextran_image_path,
        'zstack_masks_dextran_path': dextran_masks_path,
        # convenience aliases
        'gcamp_image_path':      gcamp_image_path,
        'gcamp_masks_path':      gcamp_masks_path,
        'dextran_image_path':    dextran_image_path,
        'dextran_masks_path':    dextran_masks_path,
        'alignment_folder':      alignment_folder,
        'coregistration_folder': coregistration_folder,
    }

def add_micron_coord_sys(sdata, pixel_size=None, z_step=None):
    # Define the pixel scaling factor
    if pixel_size is None and 'table' in sdata:
        pixel_size = sdata['table'].uns['section_metadata']['pixel_size']
    if z_step is None and 'table' in sdata:
        z_step = sdata['table'].uns['section_metadata']['z_step_size']
    else:
        z_step = 1.0

    if isinstance(pixel_size, (int, float)):
        pixel_size = [pixel_size, pixel_size]
        
    # 2D Images (channel, y, x)
    scale_yx = Scale(pixel_size, axes=("y", "x"))

    # For 3D Z-Stacks (c, z, y, x)
    scale_czyx = Scale(
        [z_step] + pixel_size, 
        axes=("z", "y", "x")
    )

    identity = Identity()
    # --- Images ---
    for image_name in sdata.images:
        dims = sdata[image_name].dims if not isinstance(sdata[image_name], xr.core.datatree.DataTree) else sdata[image_name]['scale0'].dims
        if 'z' in dims:
            set_transformation(
                sdata.images[image_name], 
                scale_czyx, 
                to_coordinate_system="microns"
            )
        else:
            set_transformation(
                sdata.images[image_name], 
                scale_yx, 
                to_coordinate_system="microns"
            )

    # Labels
    for label_name in sdata.labels:
        set_transformation(
            sdata.labels[label_name], 
            scale_yx, 
            to_coordinate_system="microns"
        )

    # Shapes
    for shape_name in sdata.shapes:
        set_transformation(
            sdata.shapes[shape_name], 
            identity, 
            to_coordinate_system="microns"
        )
    # Points
    for point_name in sdata.points:
        set_transformation(
            sdata.points[point_name], 
            identity, 
            to_coordinate_system="microns"
        )
    return sdata

def get_ome_metadata(tif_path, level_n=0):
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

def _is_multiscale(element):
    return (
        hasattr(element, 'keys')
        and callable(element.keys)
        and not isinstance(element, GeoDataFrame)
    )

def rename_chans(sdata, el, channel_name_map=None):
    if channel_name_map is None:
        channel_name_map = {
            'DAPI': 'dapi',
            'ATP1A1/CD45/E-Cadherin': 'boundary',
            '18S': 'rna',
            'AlphaSMA/Vimentin': 'protein'
        }

    def _rename_channel_coord(element_obj):
        if not hasattr(element_obj, 'coords'):
            return element_obj
        if 'c' not in element_obj.coords:
            return element_obj

        old_names = [str(ch) for ch in element_obj.coords['c'].values]
        new_names = [channel_name_map.get(ch, ch) for ch in old_names]

        if old_names == new_names:
            return element_obj

        if len(set(new_names)) != len(new_names):
            raise ValueError(
                f"Renaming channels for '{el}' would create duplicate names: {new_names}"
            )

        return element_obj.assign_coords(c=new_names)

    element = sdata[el]

    if _is_multiscale(element):
        for scale_key in list(element.keys()):
            scale_obj = element[scale_key]
            if hasattr(scale_obj, 'image'):
                scale_obj['image'] = _rename_channel_coord(scale_obj['image'])
            else:
                element[scale_key] = _rename_channel_coord(scale_obj)
    else:
        sdata[el] = _rename_channel_coord(element)

    return sdata

def extract_scale_transform(current_transform):
    """Extract a Scale transform from a single transform or Sequence."""
    if hasattr(current_transform, 'transformations'):
        for t in current_transform.transformations:
            if isinstance(t, Scale):
                return t
    elif isinstance(current_transform, Scale):
        return current_transform
    return None

def _count_element_chunks(element):
    """Count total dask tasks across all scales of an element.

    Dask's zarr writer generates ~3 tasks per data chunk
    (compute → encode/compress → write), so we multiply the raw chunk
    count by 3 to get a realistic task-count estimate for the progress bar.
    """
    import numpy as np

    def _get_darr(obj):
        """Extract the underlying dask array from various element/node types."""
        # DataTree node: variables live in .ds (xarray Dataset)
        if hasattr(obj, 'ds') and obj.ds is not None:
            for var in obj.ds.data_vars:
                da = obj.ds[var]
                if hasattr(da, 'data') and hasattr(da.data, 'numblocks'):
                    return da.data
                if hasattr(da, 'chunks'):
                    return da
        # xarray DataArray with .data
        if hasattr(obj, 'data') and hasattr(obj.data, 'numblocks'):
            return obj.data
        # xarray DataArray with .chunks directly
        if hasattr(obj, 'chunks') and obj.chunks:
            return obj
        return None

    total = 0
    if _is_multiscale(element):
        for scale_key in element.keys():
            scale_obj = element[scale_key]
            darr = _get_darr(scale_obj)
            if darr is None:
                continue
            if hasattr(darr, 'numblocks'):
                total += int(np.prod(darr.numblocks))
            elif hasattr(darr, 'chunks'):
                total += int(np.prod([len(c) for c in darr.chunks]))
    else:
        darr = _get_darr(element)
        if darr is not None:
            if hasattr(darr, 'numblocks'):
                total += int(np.prod(darr.numblocks))
            elif hasattr(darr, 'chunks'):
                total += int(np.prod([len(c) for c in darr.chunks]))
    # Each chunk produces ~3 dask tasks (compute, encode, write to zarr)
    return max(total * 3, 1)


class _TqdmDaskCallback(Callback):
    """Dask callback that increments a tqdm bar on each completed task."""
    def __init__(self, pbar):
        self._pbar = pbar

    def _posttask(self, key, result, dsk, state, worker_id):
        self._pbar.update(1)


def write_sdata_elements(sdata, sdata_path, overwrite=False, num_workers=4):
    """
    Write a SpatialData object element-by-element with a progress bar.
    
    If overwrite=False, skips elements that already exist on disk.
    If overwrite=True, rewrites all elements.
    If writing fails, the partially-written element is deleted to avoid corrupted zarr.
    """
    import shutil

    sdata_path = Path(sdata_path)
    all_elements = list(sdata.gen_elements())  # [(etype, name, element), ...]

    # --- Step 1: create/open zarr store and write root metadata ---
    store = _resolve_zarr_store(sdata_path)
    if sdata_path.exists():
        zarr_group = zarr.open_group(store=store, mode='a')
    else:
        zarr_group = zarr.create_group(store=store, overwrite=True)
    
    sdata.write_attrs(zarr_group=zarr_group)
    store.close()
    sdata.path = sdata_path

    # --- Step 2: determine which elements to write ---
    etype_to_folder = {
        'images': 'images',
        'labels': 'labels', 
        'points': 'points',
        'shapes': 'shapes',
        'tables': 'tables',
    }

    def _element_exists(sdata_path, etype, name):
        folder = etype_to_folder.get(etype, etype)
        return (sdata_path / folder / name).exists()

    def _delete_element(sdata_path, etype, name):
        folder = etype_to_folder.get(etype, etype)
        el_path = sdata_path / folder / name
        if el_path.exists():
            shutil.rmtree(el_path)
            tqdm_nb.write(f"    🗑  Deleted incomplete element at {el_path}")

    to_write = []
    skipped = []
    for etype, name, el in all_elements:
        if not overwrite and _element_exists(sdata_path, etype, name):
            skipped.append((etype, name))
        else:
            to_write.append((etype, name, el))

    if skipped:
        print(f"Skipping {len(skipped)} already-written elements:")
        for etype, name in skipped:
            print(f"  [{etype}] {name} (already exists)")

    if not to_write:
        print("All elements already written. Nothing to do.")
        return

    t0 = time.time()
    failed = []

    # --- Step 3: write elements with a per-element chunk-level progress bar ---
    with dask.config.set(scheduler='threads', num_workers=num_workers):
        for etype, name, el in to_write:
            n_chunks = _count_element_chunks(el)
            t1 = time.time()
            with tqdm_nb(
                total=n_chunks,
                unit='task',
                desc=f"[{etype}] {name}",
                bar_format='{desc} {bar} {n_fmt}/{total_fmt} tasks [{elapsed}<{remaining}, {rate_fmt}]',
            ) as chunk_pbar:
                try:
                    with _TqdmDaskCallback(chunk_pbar):
                        sdata.write_element(name, overwrite=overwrite)
                    chunk_pbar.set_postfix_str(f"done in {time.time()-t1:.1f}s", refresh=True)
                except Exception as e:
                    failed.append((etype, name, str(e)))
                    _delete_element(sdata_path, etype, name)

        # --- Step 4: consolidate metadata ---
        sdata.write_consolidated_metadata()

    if failed:
        print(f"⚠  {len(failed)} element(s) failed:")
        for etype, name, err in failed:
            print(f"  [{etype}] {name}: {err}")

    print(f"Done: {len(to_write) - len(failed)}/{len(to_write)} elements in {(time.time()-t0)/60:.1f} min")

def get_microns_scales(sdata, element_name):
    el = sdata[element_name]
    if _is_multiscale(el):
        img = sd.get_pyramid_levels(el, n=0)
        img = img.image if hasattr(img, 'image') else img
    else:
        img = el.image if hasattr(el, 'image') else el

    # Get transforms from the actual image DataArray, not the DataTree
    el_transforms = get_transformation(img, get_all=True)
    microns_tf = el_transforms.get('microns', None)
    if microns_tf is None:
        ps = sdata['table'].uns['section_metadata']['pixel_size']
        microns_tf = Scale([ps, ps], axes=('x', 'y'))
        set_transformation(img, microns_tf, to_coordinate_system='microns')
    if len(microns_tf.scale) >= 2:
        x_y_axes = ('x', 'y')
        x_y_tf = [microns_tf.axes.index(axis) for axis in x_y_axes if axis in microns_tf.axes]
        microns_tf = Scale([microns_tf.scale[i] for i in x_y_tf], x_y_axes)
    return microns_tf


def get_channel_name(image, chan, print_chan_names_only=False):
    channel_aliases = {'DAPI': ['dapi','nuclear'], 
                    'ATP1A1/CD45/E-Cadherin': ['boundary'],
                    '18S': ['rna', 'RNA'],
                    'AlphaSMA/Vimentin': ['protein']
    }
    if print_chan_names_only:
        chan_names = sd.models.get_channel_names(image)
        print('Available channel names:')
        for name in chan_names:
            print(f' - {name}')
        return None
    for chan_label, aliases in channel_aliases.items():
        for alias in aliases:
            if alias.lower() in chan.lower():
                return chan_label
    return chan

def get_element_bytes(el):
    try:
        if hasattr(el, 'data') and hasattr(el.data, 'nbytes'):
            return el.data.nbytes
        elif hasattr(el, 'nbytes') and not callable(el.nbytes):
            return el.nbytes
        elif hasattr(el, 'compute'):  # Dask DataFrame (points)
            return el.compute().memory_usage(deep=True).sum()
        elif hasattr(el, '__sizeof__'):  # AnnData
            return el.__sizeof__()
    except Exception:
        pass
    return 0

def print_sdata_size_summary(sdata):
    # --- Size summary ---
    print("\n=== Combined SpatialData size summary ===")
    total_bytes = 0
    for element_type, container in [('images', sdata.images),
                                     ('labels', sdata.labels),
                                     ('points', sdata.points),
                                     ('tables', sdata.tables)]:
        # Group by prefix (e.g. 'dapi_zstack', 'boundary', 'cell_labels')
        groups = {}
        for name, el in container.items():
            el_bytes = get_element_bytes(el)
            # Extract prefix: 'dapi_zstack-3' → 'dapi_zstack', 'gcamp' → 'gcamp'
            prefix = name.rsplit('-', 1)[0] if '-' in name and name.rsplit('-', 1)[-1].isdigit() else name
            if prefix not in groups:
                groups[prefix] = {'bytes': 0, 'count': 0}
            groups[prefix]['bytes'] += el_bytes
            groups[prefix]['count'] += 1
            total_bytes += el_bytes

        print(f"\n  [{element_type}]")
        for prefix, info in groups.items():
            n = info['count']
            gb = info['bytes'] / 1e9
            label = f"({n} sections)" if n > 1 else ""
            print(f"    {prefix} {label}: {gb:.2f} GB")

    print(f"\n  Total (uncompressed, in-memory): {total_bytes / 1e9:.2f} GB")
    print(f"  On-disk (zarr, ~3-5x compression): ~{total_bytes / 1e9 / 4:.2f}–{total_bytes / 1e9 / 3:.2f} GB estimated")
    print("=========================================\n")


def get_spatial_elements(sdata):
    spatial_elements = []
    spatial_elements.extend(sdata.images.keys())
    spatial_elements.extend(sdata.labels.keys())
    spatial_elements.extend(sdata.points.keys())
    spatial_elements.extend(sdata.shapes.keys())
    return spatial_elements


def rename_coordinate_systems_manual(sdata, rename_dict):
    from geopandas import GeoDataFrame

    def _rename_tfs(tfs):
        return {rename_dict.get(k, k): v for k, v in tfs.items()}

    def _is_multiscale_element(el):
        keys_attr = getattr(el, "keys", None)
        if not callable(keys_attr):
            return False
        try:
            ks = list(el.keys())
            if len(ks) == 0:
                return False
            # multiscale nodes usually have .image at each scale
            first = el[ks[0]]
            return hasattr(first, "image")
        except Exception:
            return False

    for store in [sdata.images, sdata.labels, sdata.points, sdata.shapes]:
        for el_name in list(store.keys()):
            el = store[el_name]
            try:
                if _is_multiscale_element(el):
                    for scale in el.keys():
                        node = el[scale]
                        img = node.image if hasattr(node, "image") else node
                        img.attrs["transform"] = _rename_tfs(
                            dict(img.attrs.get("transform", {}))
                        )
                else:
                    # points/shapes/geodataframe/single-scale elements
                    if hasattr(el, "attrs"):
                        el.attrs["transform"] = _rename_tfs(
                            dict(el.attrs.get("transform", {}))
                        )
            except Exception as e:
                print(f"  Warning: could not rename transforms for {el_name}: {e}")

    return sdata

def rename_elements_section(sdata, section_n, rename_tables=True):
    for el in list(sdata.images.keys()):
        section_el = sdata[el]
        del sdata[el]
        sdata.images[f"{el}_{section_n}"] = section_el

    for el in list(sdata.labels.keys()):
        section_el = sdata[el]
        del sdata[el]
        sdata.labels[f"{el}_{section_n}"] = section_el

    for el in list(sdata.points.keys()):
        section_el = sdata[el]
        del sdata[el]
        sdata.points[f"{el}_{section_n}"] = section_el
    
    for el in list(sdata.shapes.keys()):
        section_el = sdata[el]
        del sdata[el]
        sdata.shapes[f"{el}_{section_n}"] = section_el
    if rename_tables:
        for el in list(sdata.tables.keys()):
            section_el = sdata[el]
            del sdata[el]
            sdata.tables[f"{el}_{section_n}"] = section_el
    return sdata

def get_transcripts_bboxes(transcripts, id_col='cell_labels'):
    transcripts = transcripts.compute() if hasattr(transcripts, 'compute') else transcripts
    # If no transcripts, return empty dict quickly
    cell_label_bboxes = {}
    if transcripts.shape[0] == 0:
        cell_label_bboxes = {}
    else:
        # Aggregate min/max per cell label for z, y, x
        grouped = transcripts.groupby(id_col)[['z', 'y', 'x']].agg(['min', 'max'])

        import numpy as np
        for cell_label, row in grouped.iterrows():
            # Skip background / unmapped label if present
            if cell_label == 0:
                continue
            z_min = int(np.floor(row[('z', 'min')]))
            y_min = int(np.floor(row[('y', 'min')]))
            x_min = int(np.floor(row[('x', 'min')]))
            z_max = int(np.ceil(row[('z', 'max')]))
            y_max = int(np.ceil(row[('y', 'max')]))
            x_max = int(np.ceil(row[('x', 'max')]))
            cell_label_bboxes[cell_label] = (z_min, y_min, x_min, z_max, y_max, x_max)
    return cell_label_bboxes

def get_single_scale(sdata, keep_scale=2, zstack_scale=0):
    single_scale_sdata = sd.SpatialData()
    for el_name in sdata.images.keys():
        if el_name in ['zstack', 'gcamp', 'dextran']:
            single_scale_sdata.images[el_name] = sd.get_pyramid_levels(sdata[el_name], n=zstack_scale)
        else:
            single_scale_sdata.images[el_name] = sd.get_pyramid_levels(sdata[el_name], n=keep_scale)
    for el_name in sdata.labels.keys():
        if el_name in ['zstack_label', 'gcamp_labels', 'dextran_labels']:
            single_scale_sdata.labels[el_name] = sd.get_pyramid_levels(sdata[el_name], n=zstack_scale)
        else:
            single_scale_sdata.labels[el_name] = sd.get_pyramid_levels(sdata[el_name], n=keep_scale)
    for el_name in sdata.points.keys():
        single_scale_sdata.points[el_name] = sdata.points[el_name]
    for el_name in sdata.tables.keys():
        single_scale_sdata.tables[el_name] = sdata.tables[el_name]
    for el_name in sdata.shapes.keys():
        single_scale_sdata.shapes[el_name] = sdata.shapes[el_name]
    return single_scale_sdata

def drop_sdata_elements(sdata, drop_elements=['nucleus_labels', 'cell_boundaries', 'cell_circles', 'nucleus_boundaries']):
    for el_name in drop_elements:
        if el_name in sdata.labels:
            del sdata.labels[el_name]
        if el_name in sdata.images:
            del sdata.images[el_name]
        if el_name in sdata.shapes:
            del sdata.shapes[el_name]
        if el_name in sdata.points:
            del sdata.points[el_name]
    return sdata

def separate_channels(sdata, element='morphology_focus', section_n=None, drop_source=True):
    channel_name_map = {
        'DAPI': 'dapi',
        'ATP1A1/CD45/E-Cadherin': 'boundary',
        '18S': 'rna',
        'AlphaSMA/Vimentin': 'protein'
    }

    el = sdata.images[element]

    # Get channel names from scale0
    if hasattr(el, 'keys'):
        scale_levels = list(el.keys())
    else:
        scale_levels = None

    if scale_levels:
        c_coords = el[scale_levels[0]].image.coords['c'].values
    else:
        c_coords = el.coords['c'].values

    for ch in c_coords:
        ch_name = channel_name_map.get(str(ch), str(ch))

        if scale_levels:
            scale_dict = {}

            for scale_level in scale_levels:
                img = el[str(scale_level)].image          # (c, z, y, x) or (c, y, x)
                el_tf = get_transformation(img, get_all=True)

                chan_img = img.sel(c=ch)                  # (z, y, x) or (y, x)
                chan_img = chan_img.expand_dims('c', axis=0)
                chan_img = chan_img.assign_coords(c=[ch_name])

                # Determine dims based on actual shape
                if chan_img.ndim == 4:
                    parse_dims = ('c', 'z', 'y', 'x')
                else:
                    parse_dims = ('c', 'y', 'x')
                    use_model = Image3DModel if 'z' in img.dims else Image3DModel

                parsed = Image3DModel.parse(
                    chan_img.values,
                    dims=parse_dims,
                    c_coords=[ch_name],
                    scale_factors=None,
                    transformations=el_tf
                ) if chan_img.ndim == 4 else __import__(
                    'spatialdata.models', fromlist=['Image2DModel']
                ).Image2DModel.parse(
                    chan_img.values,
                    dims=parse_dims,
                    c_coords=[ch_name],
                    scale_factors=None,
                    transformations=el_tf
                )
                scale_dict[str(scale_level)] = parsed

            new_dt = xr.DataTree.from_dict({
                scale: xr.Dataset({'image': arr})
                for scale, arr in scale_dict.items()
            })
        else:
            el_tf = get_transformation(el, get_all=True)
            chan_img = el.sel(c=ch)
            if 'c' not in chan_img.dims:
                chan_img = chan_img.expand_dims('c', axis=0)
            chan_img = chan_img.assign_coords(c=[ch_name])

            # Determine if this is 3D (c, z, y, x) or 2D (c, y, x)
            if chan_img.ndim == 4:
                parse_dims = ('c', 'z', 'y', 'x')
                new_dt = Image3DModel.parse(
                    chan_img.values,
                    dims=parse_dims,
                    c_coords=[ch_name],
                    transformations=el_tf
                )
            else:
                from spatialdata.models import Image2DModel
                parse_dims = ('c', 'y', 'x')
                new_dt = Image2DModel.parse(
                    chan_img.values,
                    dims=parse_dims,
                    c_coords=[ch_name],
                    transformations=el_tf
                )

        if ch_name in sdata.images:
            del sdata.images[ch_name]
        if section_n is not None:
            ch_name = f"{ch_name}_{section_n}"
        sdata.images[ch_name] = new_dt

    if drop_source and element in sdata.images:
        del sdata.images[element]
    return sdata