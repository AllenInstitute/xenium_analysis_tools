
from spatialdata.transformations import Scale, Translation, Sequence, Identity, set_transformation
import xarray as xr
import numpy as np
import pandas as pd
import json
from pathlib import Path

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
        "sdata_path": data_root / f'{dataset_config.get("xenium_name", None)}_processed',
        "confocal_path": data_root / dataset_config.get("confocal_name", None),
        "raw_confocal_path": data_root / dataset_config.get("raw_confocal_name", None),
        "zstack_path": data_root / dataset_config.get("zstack_name", None),
        "zstack_masks": data_root / dataset_config.get("zstack_masks_name", None),
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

    # 2D Images (channel, y, x)
    # c = 1.0 (channels are discrete)
    scale_yx = Scale([pixel_size, pixel_size], axes=("y", "x"))
    scale_cyx = Scale([pixel_size, pixel_size], axes=("y", "x"))

    # For 3D Z-Stacks (c, z, y, x)
    # c = 1.0 (channels are discrete)
    # z = 3.0 (microns per plane)
    # y, x = 0.2125 (microns per pixel)
    scale_czyx = Scale(
        [z_step, pixel_size, pixel_size], 
        axes=("z", "y", "x")
    )

    # Identity transform for elements already in microns
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
                scale_cyx, 
                to_coordinate_system="microns"
            )

    # --- Labels ---
    # Both labels are (y, x)
    for label_name in sdata.labels:
        set_transformation(
            sdata.labels[label_name], 
            scale_yx, 
            to_coordinate_system="microns"
        )

    # --- Shapes & Points ---
    # Already in microns
    for shape_name in sdata.shapes:
        set_transformation(
            sdata.shapes[shape_name], 
            identity, 
            to_coordinate_system="microns"
        )

    for point_name in sdata.points:
        set_transformation(
            sdata.points[point_name], 
            identity, 
            to_coordinate_system="microns"
        )
    return sdata

def add_mapped_cells_cols(sdata, mapped_h5ad_path):
    import scanpy as sc
    mapped_h5ad = sc.read_h5ad(mapped_h5ad_path)
    mapping_obs_cols = np.setdiff1d(mapped_h5ad.obs.columns, sdata['table'].obs.columns)
    if len(mapping_obs_cols) == 0:
        print("No new columns to add from mapped data")
    else:
        print(f"Adding {len(mapping_obs_cols)} columns from mapped data: {mapping_obs_cols}")
        sdata['table'].obs = sdata['table'].obs.merge(
            mapped_h5ad.obs[mapping_obs_cols],
            left_index=True,
            right_index=True,
            how='outer'
        )
    mapping_vars_cols = np.setdiff1d(mapped_h5ad.var.columns, sdata['table'].var.columns)
    if len(mapping_vars_cols) == 0:
        print("No new columns to add from mapped data")
    else:
        print(f"Adding {len(mapping_vars_cols)} columns from mapped data: {mapping_vars_cols}")
        sdata['table'].var = sdata['table'].var.merge(
            mapped_h5ad.var[mapping_vars_cols],
            left_index=True,
            right_index=True,
            how='outer'
        )
    return sdata

def add_type_id_columns(sdata, col_name, table_name='table'):
    if col_name in sdata[table_name].obs.columns:
        col_id = col_name.replace('name', 'id')
        sdata[table_name].obs[col_id] = sdata[table_name].obs[col_name].str.split(' ').str[0].astype('int')
        print(f"Added {col_id} column")
    else:
        print(f"{col_name} column not found in {table_name}.obs")
    return sdata

def add_grouped_types_columns(sdata,
                           new_col,
                           type_mappings=None,
                           table_name='table',
                           null_value='other'):
    default_mappings = {
        'broad_class': {
            'class_name': [
                {'op': 'contains', 'value': 'GABA', 'assign': 'GABAergic'},
                {'op': 'contains', 'value': 'Glut', 'assign': 'Glutamatergic'},
            ],
            'class_id': [
                {'op': 'gte', 'value': 29, 'assign': 'Non-neuronal'}
            ]
        }
    }

    if type_mappings is None:
        type_mappings = default_mappings.get(new_col, {})

    norm_mappings = {}
    for crit_col, rules in type_mappings.items():
        norm = []
        if isinstance(rules, dict):
            for assign, crit in rules.items():
                if isinstance(crit, (list, tuple)) and len(crit) >= 2:
                    op, val = crit[0], crit[1]
                else:
                    op, val = 'contains', crit
                norm.append({'op': op, 'value': val, 'assign': assign})
        elif isinstance(rules, (list, tuple)):
            for r in rules:
                if isinstance(r, dict) and {'op', 'value', 'assign'}.issubset(r.keys()):
                    norm.append(r)
                elif isinstance(r, (list, tuple)) and len(r) >= 3:
                    op, val, assign = r[0], r[1], r[2]
                    norm.append({'op': op, 'value': val, 'assign': assign})
                else:
                    # skip unknown rule format
                    continue
        norm_mappings[crit_col] = norm

    # Initialize column
    print(f"Adding '{new_col}' to {table_name}.obs")
    sdata[table_name].obs[new_col] = null_value

    for crit_col, rules in norm_mappings.items():
        if crit_col not in sdata[table_name].obs.columns:
            # skip missing criteria columns
            continue
        series = sdata[table_name].obs[crit_col]

        for rule in rules:
            op = rule['op']
            val = rule['value']
            assign = rule['assign']

            mask = pd.Series(False, index=series.index)

            try:
                if op == 'contains':
                    mask = series.astype(str).str.contains(str(val), na=False)
                elif op == 'in':
                    if isinstance(val, (list, tuple, set)):
                        mask = series.isin(val)
                    else:
                        mask = series == val
                elif op == 'eq':
                    mask = series == val
                elif op == 'neq':
                    mask = series != val
                elif op in ('gte', 'lte', 'gt', 'lt'):
                    num = pd.to_numeric(series, errors='coerce')
                    cmp_val = float(val)
                    if op == 'gte':
                        mask = num >= cmp_val
                    elif op == 'lte':
                        mask = num <= cmp_val
                    elif op == 'gt':
                        mask = num > cmp_val
                    elif op == 'lt':
                        mask = num < cmp_val
                elif op == 'regex':
                    mask = series.astype(str).str.match(str(val))
                else:
                    # unknown op -> skip
                    continue
            except Exception:
                # on any evaluation error, skip this rule
                continue

            sdata[table_name].obs.loc[mask, new_col] = assign

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