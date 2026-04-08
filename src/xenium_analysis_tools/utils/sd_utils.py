
from spatialdata.transformations import Scale, Translation, Sequence, Identity, set_transformation, get_transformation
from spatialdata.models import Image3DModel
import spatialdata as sd
import xarray as xr
import numpy as np
import pandas as pd
import json
from pathlib import Path
from geopandas import GeoDataFrame

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

def get_dataset_paths(dataset_id, 
                            data_root=Path('/root/capsule/data'),
                            scratch_root=Path('/root/capsule/scratch'),
                            results_root=Path('/root/capsule/results'),
                            code_root=Path('/root/capsule/code')):
    datasets_naming_dict_path = code_root / 'datasets_names_dict.json'
    with open(datasets_naming_dict_path) as f:
        datasets_naming_dict = json.load(f)
    dataset_id = str(dataset_id)  # Ensure string format
    dataset_config = datasets_naming_dict[dataset_id]
    
    paths = {
        "data_root": data_root,
        "scratch_root": scratch_root,
        "results_root": results_root,
        "xenium_dataset_name": dataset_config.get("xenium_name", None),
        "sdata_path": data_root / f'{dataset_config["xenium_name"]}_processed' if dataset_config.get("xenium_name") else None,
        "confocal_path": data_root / dataset_config["confocal_name"] if dataset_config.get("confocal_name") else None,
        "raw_confocal_path": data_root / dataset_config["raw_confocal_name"] if dataset_config.get("raw_confocal_name") else None,
        "zstack_path": data_root / dataset_config["zstack_name"] if dataset_config.get("zstack_name") else None,
        "zstack_masks": data_root / dataset_config["zstack_masks_name"] if dataset_config.get("zstack_masks_name") else None,
    }
    
    return paths

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