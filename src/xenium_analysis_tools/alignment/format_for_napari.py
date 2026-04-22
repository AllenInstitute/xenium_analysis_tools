from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd
import spatialdata as sd
from spatialdata.models import Image3DModel, Labels3DModel
from spatialdata.transformations import get_transformation
import matplotlib.pyplot as plt
import dask.dataframe as dd
import xarray as xr
from xenium_analysis_tools.alignment.align_sections import _get_lifted_element_transforms
import dask.dataframe as dd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def filter_labels(sdata, label_elements='cell_labels', table='table', key_col='cell_labels'):
    # Get all label elements that match the specified prefix
    labels = [lbl for lbl in sdata.labels.keys() if lbl.startswith(label_elements)]
    
    # Get the valid IDs from the table
    valid_ids = set(sdata.tables[table].obs[key_col].unique())
    
    for el in labels:
        # Get the label data for the current element
        label_data = sdata.labels[el].data
        
        # Create a mask for invalid labels (not in valid_ids)
        invalid_mask = ~np.isin(label_data, list(valid_ids))
        
        # Set invalid labels to 0 (remove them)
        label_data[invalid_mask] = 0
        
        # Update the label data in the spatial data object
        sdata.labels[el].data = label_data
    
    return sdata

def get_fov_sdata(
    sdata,
    gcamp_shape_el='gcamp',
    x_buffer=0.0,
    y_buffer=0.0,
    target_coordinate_system='czstack_microns',
):
    if gcamp_shape_el not in sdata.images:
        raise KeyError(
            f"Reference image '{gcamp_shape_el}' not found in sdata.images. "
            f"Available: {list(sdata.images.keys())}"
        )

    ref_img = sdata.images[gcamp_shape_el]
    n_z, n_y, n_x = _get_zyx_shape(ref_img)

    # Full affine from image index (x,y,z) -> target world (x,y,z)
    tf = get_transformation(ref_img, to_coordinate_system=target_coordinate_system)
    mat = np.array(
        tf.to_affine_matrix(input_axes=('x', 'y', 'z'), output_axes=('x', 'y', 'z')),
        dtype=np.float64,
    )

    # 8 corners of image index volume (center coordinates)
    xs = np.array([0.0, float(n_x - 1)])
    ys = np.array([0.0, float(n_y - 1)])
    zs = np.array([0.0, float(n_z - 1)])
    corners_xyz = np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=np.float64)
    corners_h = np.concatenate([corners_xyz, np.ones((corners_xyz.shape[0], 1), dtype=np.float64)], axis=1)
    world_xyz = (corners_h @ mat.T)[:, :3]

    x_min, y_min, z_min = world_xyz.min(axis=0)
    x_max, y_max, z_max = world_xyz.max(axis=0)

    # Apply user buffers (in target coordinate units)
    x_min -= float(x_buffer)
    x_max += float(x_buffer)
    y_min -= float(y_buffer)
    y_max += float(y_buffer)

    min_coords = [float(z_min), float(y_min), float(x_min)]  # for axes ['z','y','x']
    max_coords = [float(z_max), float(y_max), float(x_max)]

    # Filter points lazily in the SAME target coordinate system box
    filtered_points = {}
    for el, pts in sdata.points.items():
        if el.startswith('transcripts'):
            tf_pts = sd.transform(pts, to_coordinate_system=target_coordinate_system)
            keep = (
                (tf_pts['z'] >= min_coords[0]) & (tf_pts['z'] <= max_coords[0]) &
                (tf_pts['y'] >= min_coords[1]) & (tf_pts['y'] <= max_coords[1]) &
                (tf_pts['x'] >= min_coords[2]) & (tf_pts['x'] <= max_coords[2])
            )
            filtered_points[el] = pts[keep]
        else:
            filtered_points[el] = pts

    def _is_2d_element(el):
        """Return True if the element has no 'z' spatial dimension."""
        if hasattr(el, 'dims'):
            return 'z' not in el.dims
        if hasattr(el, 'keys'):   # DataTree / multiscale
            try:
                lvl = next(iter(el.keys()))
                node = el[lvl]
                dims = node.dims if hasattr(node, 'dims') else getattr(node, 'image', node).dims
                return 'z' not in dims
            except Exception:
                pass
        return False

    images_2d = {k: v for k, v in sdata.images.items() if _is_2d_element(v)}
    labels_2d = {k: v for k, v in sdata.labels.items() if _is_2d_element(v)}

    # Query non-point elements with bounding box, then attach filtered points.
    points_backup = {k: sdata.points[k] for k in list(sdata.points.keys())}
    try:
        for el in list(sdata.points.keys()):
            del sdata.points[el]
        for el in images_2d:
            del sdata.images[el]
        for el in labels_2d:
            del sdata.labels[el]

        fov_data = sdata.query.bounding_box(
            axes=['z', 'y', 'x'],
            min_coordinate=min_coords,
            max_coordinate=max_coords,
            target_coordinate_system=target_coordinate_system,
            filter_table=False,
        )
    finally:
        sdata.points = points_backup
        sdata.images.update(images_2d)
        sdata.labels.update(labels_2d)

    # Re-attach 2D section elements and filtered points.
    # 2D elements are already precisely registered in czstack_microns space via
    # their lifting transform, so no additional spatial filtering is needed.
    fov_data.images.update(images_2d)
    fov_data.labels.update(labels_2d)
    fov_data.points = filtered_points
    return fov_data

def filter_transcripts(sdata,
                        genes_to_show='all',
                        filter_is_gene=True,
                        filter_assigned_to_cell=True,
                        min_qv=20,
                        filter_transcripts_to_cells=True,
                        sections=[],
                        add_prefix='',
                        source_el=None,
                        return_only=False):
    """
    Filter transcript point elements in sdata.

    If add_prefix is empty, the original elements are updated in place.
    If add_prefix is set, the original elements are never modified; filtered
    copies are written under f"{add_prefix}_{source_el_name}".

    Parameters
    ----------
    source_el : str or None
        When add_prefix is set, filter only this specific element instead of
        iterating all transcripts* elements. The source element is never
        modified. When None, all transcripts* elements are processed.
    """
    if add_prefix and source_el is not None:
        candidates = [source_el]
    else:
        candidates = [k for k in sdata.points.keys() if k.startswith('transcripts')]

    if return_only:
        tx_els = {}

    for tx_el in candidates:
        if tx_el not in sdata.points:
            print(f"Warning: {tx_el!r} not found in sdata.points, skipping")
            continue
        print(f'Filtering {tx_el}...')

        tx = sdata.points[tx_el]

        if sections:
            tx_sec = np.unique(tx['section'].compute())[0]
            if tx_sec not in sections:
                print(f"Skipping {tx_el} (section {tx_sec} not in filter list)")
                continue

        # 1. Filter by gene name
        if genes_to_show != 'all' and 'feature_name' in tx.columns:
            tx = tx[tx['feature_name'].isin(genes_to_show)]

        # 2. Filter by quality value (qv score)
        if min_qv is not None and 'qv' in tx.columns:
            tx = tx[tx['qv'] >= min_qv]

        # 3. Filter to is_gene only (handles both bool and string column types)
        if filter_is_gene and 'is_gene' in tx.columns:
            sample_val = get_sample_val(tx['is_gene'])
            if isinstance(sample_val, (bool, np.bool_)):
                tx = tx[tx['is_gene'] == True]
            elif isinstance(sample_val, str):
                tx = tx[tx['is_gene'].str.lower() == 'true']

        # 4. Filter to transcripts assigned to a cell
        if filter_assigned_to_cell and 'cell_id' in tx.columns:
            sample_val = get_sample_val(tx['cell_id'])
            if isinstance(sample_val, str):
                tx = tx[tx['cell_id'] != 'UNASSIGNED']
            else:
                tx = tx[tx['cell_id'] != -1]

        # 5. Filter transcripts to only cells that passed the cell-level filters
        if filter_transcripts_to_cells:
            s_n = np.unique(tx['section'])[0]
            cell_table = sdata['table'].copy()
            cell_table = cell_table[cell_table.obs['section'] == s_n]
            tx = tx[tx['cell_id'].isin(cell_table.obs['cell_id'])]

        dest_el = f"{add_prefix}_{tx_el}" if add_prefix else tx_el
        print(f"{tx_el} → {dest_el!r}: filters applied (lazy — will compute on render)")

        if return_only:
            tx_els[dest_el] = tx
        else:
            sdata.points[dest_el] = tx

    if return_only:
        return tx_els
    return sdata

def is_dask(df):
    return isinstance(df, dd.DataFrame)

def get_sample_val(series):
    """Safely get first value from either a Dask or Pandas Series."""
    if isinstance(series, dd.Series):
        return series.head(1).iloc[0]
    return series.iloc[0]

def filter_cells(sdata, el='table', cell_filters=[]):
    import operator as op_module

    # Map operator strings to functions
    OPS = {
        'isin':  lambda col, val: col.isin(val),
        '>=':    lambda col, val: col >= val,
        '<=':    lambda col, val: col <= val,
        '>':     lambda col, val: col > val,
        '<':     lambda col, val: col < val,
        '==':    lambda col, val: col == val,
        '!=':    lambda col, val: col != val,
    }

    kept_cell_ids = None  # will be populated if cell filters are applied

    if cell_filters:
        tbl = sdata.tables[el]
        obs = tbl.obs.copy()
        
        # Build combined boolean mask
        mask = np.ones(len(obs), dtype=bool)
        for f in cell_filters:
            col   = f['column']
            oper  = f['operator']
            value = f['value']
            
            if col not in obs.columns:
                print(f"Warning: column '{col}' not found in table.obs — skipping filter.")
                continue
            
            filter_mask = OPS[oper](obs[col], value)
            mask &= filter_mask.values
            print(f"Filter '{col} {oper} {value}': {mask.sum()} / {len(obs)} cells pass")
        
        filtered_tbl = tbl[mask].copy()
        
        # Sync spatialdata region metadata
        if 'spatialdata_attrs' in filtered_tbl.uns:
            attrs = filtered_tbl.uns['spatialdata_attrs']
            region_key = attrs.get('region_key', 'region')
            instance_key = attrs.get('instance_key', 'cell_labels')
            if region_key in filtered_tbl.obs:
                if hasattr(filtered_tbl.obs[region_key], 'cat'):
                    filtered_tbl.obs[region_key] = filtered_tbl.obs[region_key].cat.remove_unused_categories()
                attrs['region'] = filtered_tbl.obs[region_key].unique().tolist()
        
        sdata.tables[el] = filtered_tbl
        print(f"\n{mask.sum()} / {len(obs)} cells kept after all filters.")
    return sdata

def drop_2ds(sdata):
    sdata_els = list(sdata.images.keys()) + list(sdata.labels.keys())
    for el in sdata_els:
        if hasattr(sdata[el], 'keys'):
            el_dt = sd.get_pyramid_levels(sdata[el],n=0)
            del sdata[el]
            sdata[el] = el_dt
        if 'z' not in sdata[el].dims:
            print(f"Removing 2D element: {el}")
            del sdata[el]
    return sdata


def add_mapped_cells_cols(adata, mapped_adata,
                          adata_cell_id_col='cell_id',
                          mapped_cell_id_col='cell_id',
                          verbose=False):
    import scanpy as sc
    if isinstance(mapped_adata, str) or isinstance(mapped_adata, Path):
        mapped_adata = sc.read(mapped_adata)

    # Work on a copy of obs so we don't mutate the caller's object
    mapped_obs = mapped_adata.obs.copy()

    # Identify CDM_ columns, then strip the prefix
    cdm_cols = mapped_obs.columns[mapped_obs.columns.str.startswith('CDM_')]
    col_rename = {c: c.replace('CDM_', '') for c in cdm_cols}
    mapped_obs = mapped_obs.rename(columns=col_rename)

    # Derive broad_class / class_id before merging
    if 'class_name' in mapped_obs.columns:
        mapped_obs['class_id'] = (
            mapped_obs['class_name'].str.split(' ').str[0].astype(int)
        )
        conditions = [
            mapped_obs['class_name'].str.contains('GABA'),
            mapped_obs['class_name'].str.contains('Glut'),
            mapped_obs['class_id'] >= 29,
        ]
        mapped_obs['broad_class_name'] = np.select(
            conditions, ['GABA', 'Glut', 'NN'], default='Other'
        )

    # Columns to bring over (only those not already in adata.obs)
    derived_cols = [c for c in ['class_id', 'broad_class_name'] if c in mapped_obs.columns]
    renamed_cols = list(col_rename.values())
    mapping_obs_cols = [c for c in renamed_cols + derived_cols
                        if c not in adata.obs.columns]

    if not mapping_obs_cols:
        if verbose:
            print("No new obs columns to add from mapped data")
        return adata

    if verbose:
        print(f"Adding {len(mapping_obs_cols)} obs columns: {mapping_obs_cols}")

    # ── Determine join keys ──────────────────────────────────────────────────
    has_mapped_id  = mapped_cell_id_col in mapped_obs.columns
    has_adata_id   = adata_cell_id_col  in adata.obs.columns
    has_section    = 'section' in mapped_obs.columns and 'section' in adata.obs.columns
    multi_section  = has_section and mapped_obs['section'].nunique() > 1

    if has_mapped_id and has_adata_id:
        if multi_section:
            join_keys_right = [mapped_cell_id_col, 'section']
            join_keys_left  = [adata_cell_id_col,  'section']
        else:
            join_keys_right = [mapped_cell_id_col]
            join_keys_left  = [adata_cell_id_col]

        # Build right-side DataFrame with only the columns we need
        right_cols = list(dict.fromkeys(mapping_obs_cols + join_keys_right))  # preserve order, no dupes
        right = (
            mapped_obs[right_cols]
            .drop_duplicates(subset=join_keys_right)
            .copy()
        )

        # Align section dtype to prevent silent type-mismatch misses
        if 'section' in join_keys_right:
            right['section'] = right['section'].astype(adata.obs['section'].dtype)

        # Rename right join key to match left if they differ
        if mapped_cell_id_col != adata_cell_id_col:
            right = right.rename(columns={mapped_cell_id_col: adata_cell_id_col})
            join_keys_right = [adata_cell_id_col if k == mapped_cell_id_col else k
                               for k in join_keys_right]

        orig_index = adata.obs.index
        merged = adata.obs.merge(right, left_on=join_keys_left,
                                 right_on=join_keys_right, how='left')
        merged.index = orig_index
        adata.obs = merged

        n_matched = merged[mapping_obs_cols[0]].notna().sum()
        if verbose:
            print(f"  Joined on {join_keys_left}: "
                  f"{n_matched} / {len(adata.obs)} cells matched")

        # Diagnostic: show sample values to help debug zero-match cases
        if n_matched == 0:
            print("  WARNING: 0 cells matched. Sample join-key values:")
            print(f"    adata.obs['{adata_cell_id_col}'].head()  = "
                  f"{adata.obs[adata_cell_id_col].head().tolist()}")
            print(f"    right['{adata_cell_id_col}'].head()      = "
                  f"{right[adata_cell_id_col].head().tolist()}")
            if 'section' in join_keys_left:
                print(f"    adata 'section' dtype  : {adata.obs['section'].dtype}")
                print(f"    mapped 'section' dtype : {mapped_obs['section'].dtype}")

    else:
        # Index-based fallback
        if verbose:
            print("  Falling back to index-based merge")
        orig_index = adata.obs.index
        merged = adata.obs.merge(
            mapped_obs[mapping_obs_cols],
            left_index=True, right_index=True, how='left',
        )
        merged.index = orig_index
        adata.obs = merged

    # ── var columns (index-based) ────────────────────────────────────────────
    mapping_vars_cols = [c for c in mapped_adata.var.columns
                         if c not in adata.var.columns]
    if not mapping_vars_cols:
        if verbose:
            print("No new var columns to add from mapped data")
    else:
        if verbose:
            print(f"Adding {len(mapping_vars_cols)} var columns: {mapping_vars_cols}")
        adata.var = adata.var.merge(
            mapped_adata.var[mapping_vars_cols],
            left_index=True, right_index=True, how='left',
        )

    return adata


def make_element_3d(
    sdata,
    element_2d,
    reference_3d,
    use_model,
    keep_2d=False,
    lift_mode='centered_plane',
    lift_n_z=None,
):
    """
    Lift a 2D element into 3D with an explicit z-placement convention.

    If keep_2d=True, the original 2D element is preserved under its original name and the
    lifted 3D element is written as "{element_2d}_3d".

    lift_mode:
      - 'reference_slab': repeat across the full z support of `reference_3d`
      - 'centered_plane': place the 2D element on a single plane at the section center
      - 'centered_slab': repeat across `lift_n_z` planes centered within the section
    """
    if lift_mode not in {'reference_slab', 'centered_plane', 'centered_slab'}:
        raise ValueError("lift_mode must be 'reference_slab', 'centered_plane', or 'centered_slab'")

    raw_2d = sdata[element_2d]
    raw_ref = sdata[reference_3d]

    # Support both DataTree (multiscale) and plain DataArray (single-scale)
    is_multiscale_2d = not isinstance(raw_2d, xr.DataArray) and hasattr(raw_2d, 'keys')
    is_multiscale_ref = not isinstance(raw_ref, xr.DataArray) and hasattr(raw_ref, 'keys')

    scale_levels = list(raw_2d.keys()) if is_multiscale_2d else ['scale0']
    first_ref_level = list(raw_ref.keys())[0] if is_multiscale_ref else None

    scale_dict = xr.DataTree()
    for scale_level in scale_levels:
        # Get 2D element DataArray
        if is_multiscale_2d:
            if 'z' in raw_2d[scale_level].dims:
                return sdata
            element = raw_2d[scale_level].image
        else:
            if 'z' in raw_2d.dims:
                return sdata
            element = raw_2d

        # Get reference image and z-depth; match scale level or fall back to first
        if is_multiscale_ref:
            ref_level = scale_level if scale_level in raw_ref.children else first_ref_level
            ref_image = raw_ref[ref_level].image
            shape_3d = raw_ref[ref_level]['z'].shape[0]
        else:
            ref_image = raw_ref
            shape_3d = int(raw_ref.sizes['z'])

        if lift_mode == 'reference_slab':
            target_n_z = int(shape_3d)
        elif lift_mode == 'centered_plane':
            target_n_z = 1
        else:
            if lift_n_z is None:
                raise ValueError("lift_n_z must be provided when lift_mode='centered_slab'")
            target_n_z = int(lift_n_z)

        if target_n_z < 1 or target_n_z > int(shape_3d):
            raise ValueError(f"target_n_z must be in [1, {shape_3d}], got {target_n_z}")

        z_start = (shape_3d - target_n_z) / 2.0
        transforms_3d = _get_lifted_element_transforms(
            ref_image,
            z_start=z_start,
        )
        
        el_3d = xr.concat([element] * target_n_z, dim='z')
        
        if 'c' in el_3d.dims:
            el_3d = el_3d.transpose('c', 'z', 'y', 'x')
        else:
            el_3d = el_3d.transpose('z', 'y', 'x')
        
        parsed = use_model.parse(el_3d, c_coords=list(el_3d.coords['c'].values) if 'c' in el_3d.coords else None)
        parsed.attrs['transform'] = transforms_3d
        parsed.attrs['z_lift_mode'] = lift_mode
        parsed.attrs['z_lift_start_index'] = float(z_start)
        parsed.attrs['z_lift_n_planes'] = int(target_n_z)
        parsed.attrs['z_ref_n_planes'] = int(shape_3d)
        ds_key = 'image' if use_model == Image3DModel else 'labels'
        scale_dict[scale_level] = xr.Dataset({ds_key: parsed})
    
    if use_model == Image3DModel:
        if keep_2d:
            sdata.images[f'{element_2d}_3d'] = scale_dict
        else:
            del sdata.images[element_2d]
            sdata.images[element_2d] = scale_dict
    else:
        if keep_2d:
            sdata.labels[f'{element_2d}_3d'] = scale_dict
        else:
            del sdata.labels[element_2d]
            sdata.labels[element_2d] = scale_dict
    
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
            ch_name = f"{ch_name}-{section_n}"
        sdata.images[ch_name] = new_dt

    if drop_source and element in sdata.images:
        del sdata.images[element]
    return sdata

def get_plot_sdata(sdata, 
                   sections_to_plot='all',
                   elements_to_plot='all',
                   include_zstack=True,
                   split_morphology=False,
                   lift_to_3d=False,
                   lift_n_z=11,
                   sections_um=20.0):
    plot_sd = sd.SpatialData()
    
    # Standardize sections list to strings
    if sections_to_plot == 'all':
        if 'table' in sdata.tables and 'section' in sdata['table'].obs:
            sections = [str(s) for s in np.unique(sdata['table'].obs['section'])]
        else:
            sections = 'all'
    else:
        sections = [str(s) for s in sections_to_plot]

    # Helper function to check valid elements without exact string collisions
    def should_include(el_name):
        parts = el_name.rsplit('-', 1)
        has_section_suffix = len(parts) == 2 and parts[1].isdigit()
        
        base_name = parts[0] if has_section_suffix else el_name
        sec_id = parts[1] if has_section_suffix else None
        
        if elements_to_plot != 'all':
            if base_name not in elements_to_plot and el_name not in elements_to_plot:
                return False
                
        if sections != 'all' and has_section_suffix:
            if sec_id not in sections:
                return False
                
        if sections != 'all' and not has_section_suffix and elements_to_plot != 'all' and el_name not in elements_to_plot:
            return False

        return True

    # Iterate and add valid elements
    for img in sdata.images:
        if should_include(img):
            plot_sd.images[img] = sdata.images[img]
            
    for lbl in sdata.labels:
        if should_include(lbl):
            plot_sd.labels[lbl] = sdata.labels[lbl]
            
    for pt in sdata.points:
        if should_include(pt):
            plot_sd.points[pt] = sdata.points[pt]
            
    for tbl in sdata.tables:
        if elements_to_plot == 'all' or tbl in elements_to_plot:
            table_data = sdata.tables[tbl]
            # Safely filter AnnData tables by sections if 'section' column exists
            if sections != 'all' and 'section' in table_data.obs:
                filtered_table = table_data[table_data.obs['section'].astype(str).isin(sections)].copy()
                
                # Fix validation mismatch by syncing regions in metadata
                if 'spatialdata_attrs' in filtered_table.uns:
                    attrs = filtered_table.uns['spatialdata_attrs']
                    region_key = attrs.get('region_key', 'region')
                    
                    if region_key in filtered_table.obs:
                        # Drop unused categorical unused categories if they exist
                        if hasattr(filtered_table.obs[region_key], 'cat'):
                            filtered_table.obs[region_key] = filtered_table.obs[region_key].cat.remove_unused_categories()
                        
                        # Update expected regions list
                        attrs['region'] = filtered_table.obs[region_key].unique().tolist()
                
                plot_sd.tables[tbl] = filtered_table
            else:
                plot_sd.tables[tbl] = table_data

    # Add Zstack elements if requested
    if include_zstack:
        for ch in ['gcamp', 'dextran']:            
            if ch not in plot_sd.images and ch in sdata.images:
                plot_sd.images[ch] = sdata.images[ch]

    # Optionally expand morphology channels / lift to 3D for napari
    if split_morphology or lift_to_3d:
        sec_ns = None if sections == 'all' else [int(s) for s in sections]

        # When lifting to 3D, dapi_zstack-N is required as a z-placement reference for each
        # section's lifted channels. Temporarily include any missing ones from the source
        # sdata so expand_for_napari can use them, then remove them from the result.
        lift_refs_added = []
        if lift_to_3d:
            _lift_sec_ns = sec_ns if sec_ns is not None else [
                int(k.rsplit('-', 1)[1])
                for k in sdata.images
                if k.startswith('dapi_zstack-') and k.rsplit('-', 1)[1].isdigit()
            ]
            for s in _lift_sec_ns:
                ref_key = f'dapi_zstack-{s}'
                if ref_key in sdata.images and ref_key not in plot_sd.images:
                    plot_sd.images[ref_key] = sdata.images[ref_key]
                    lift_refs_added.append(ref_key)

        plot_sd = expand_for_napari(
            plot_sd,
            sections_um=sections_um,
            split_morphology=split_morphology,
            lift_to_3d=lift_to_3d,
            lift_n_z=lift_n_z,
            sections=sec_ns,
        )

        # Remove lift-only reference elements that weren't originally requested
        for ref_key in lift_refs_added:
            if ref_key in plot_sd.images:
                del plot_sd.images[ref_key]
                
    return plot_sd


def is_dask(df):
    return isinstance(df, dd.DataFrame)


def get_sample_val(series):
    """Safely get first value from either a Dask or Pandas Series."""
    if isinstance(series, dd.Series):
        return series.head(1).iloc[0]
    else:
        return series.iloc[0]


def _get_zyx_shape(image_element):
    """Return (n_z, n_y, n_x) from an image element with shape (z,y,x) or (c,z,y,x)."""
    da = image_element
    if hasattr(da, 'data'):
        arr = da.data
    else:
        arr = da

    if arr.ndim == 4:  # expected (c, z, y, x)
        return int(arr.shape[1]), int(arr.shape[2]), int(arr.shape[3])
    if arr.ndim == 3:  # expected (z, y, x)
        return int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
    raise ValueError(f"Unsupported image ndim={arr.ndim}. Expected 3D or 4D image.")


def expand_for_napari(
    sdata,
    sections_um=20.0,
    split_morphology=True,
    lift_to_3d=True,
    lift_n_z=11,
    sections=None,
):
    """
    Prepare a *copy* of sdata for napari visualization by optionally:

    1. Splitting `morphology_focus-N` into individual channel images
       (dapi-N, boundary-N, rna-N, protein-N).
    2. Lifting those 2D channel images (or morphology_focus itself) to 3D
       so they span the full section thickness along z, matching dapi_zstack-N.

    Parameters
    ----------
    sdata : SpatialData
        The combined data object (read from combined_data.zarr).
    sections_um : float
        Known section thickness in µm (used for z-centering of lifted images).
    split_morphology : bool
        If True, split morphology_focus into per-channel images.
    lift_to_3d : bool
        If True, lift the per-channel (or morphology_focus) images to a 3D
        slab spanning sections_um. Requires split_morphology=True to lift
        individual channels; if split_morphology=False, lifts morphology_focus.
    lift_n_z : int
        Number of z-planes for the lifted slab (should match dapi_zstack n_z).
    sections : list of int/str or None
        Which sections to expand. None = all sections found in the data.

    Returns
    -------
    SpatialData
        A shallow-copy SpatialData with added napari-ready elements.
        Original elements are untouched; new elements are added alongside them.
    """
    import copy

    # Build a shallow-copy so we don't mutate the original
    plot_sdata = sd.SpatialData(
        images=dict(sdata.images),
        labels=dict(sdata.labels),
        points=dict(sdata.points),
        tables=dict(sdata.tables),
    )

    # Determine which sections to expand
    if sections is None:
        mf_keys = [k for k in plot_sdata.images if k.startswith('morphology_focus-')]
        sections = [int(k.rsplit('-', 1)[1]) for k in mf_keys]

    for s_n in sections:
        mf_key = f'morphology_focus-{s_n}'
        ref_key = f'dapi_zstack-{s_n}'
        if mf_key not in plot_sdata.images:
            continue

        if split_morphology:
            # Split channels from morphology_focus with section suffix to avoid overwriting across sections
            plot_sdata = separate_channels(plot_sdata, element=mf_key, section_n=s_n, drop_source=False)

            # Determine the newly created channel keys
            channel_name_map = {
                'DAPI': 'dapi',
                'ATP1A1/CD45/E-Cadherin': 'boundary',
                '18S': 'rna',
                'AlphaSMA/Vimentin': 'protein',
            }
            el = sdata.images[mf_key]
            if hasattr(el, 'keys'):
                c_coords = el[list(el.keys())[0]].image.coords['c'].values
            else:
                c_coords = el.coords['c'].values
            channel_keys = [f"{channel_name_map.get(str(ch), str(ch))}" for ch in c_coords]

            if lift_to_3d and ref_key in plot_sdata.images:
                for ch_key in channel_keys:
                    full_ch_key = f'{ch_key}-{s_n}'
                    if full_ch_key not in plot_sdata.images:
                        continue
                    plot_sdata = make_element_3d(
                        plot_sdata,
                        full_ch_key,
                        ref_key,
                        Image3DModel,
                        keep_2d=False,
                        lift_mode='centered_slab',
                        lift_n_z=lift_n_z,
                    )
        elif lift_to_3d and ref_key in plot_sdata.images:
            # Lift morphology_focus itself without splitting
            plot_sdata = make_element_3d(
                plot_sdata,
                mf_key,
                ref_key,
                Image3DModel,
                keep_2d=False,
                lift_mode='centered_slab',
                lift_n_z=lift_n_z,
            )

        # Lift label elements (cell_labels-N, gcamp_labels-N, etc.) to 3D
        if lift_to_3d and ref_key in plot_sdata.images:
            for lbl_key in list(plot_sdata.labels.keys()):
                lbl_parts = lbl_key.rsplit('-', 1)
                if len(lbl_parts) == 2 and lbl_parts[1] == str(s_n):
                    plot_sdata = make_element_3d(
                        plot_sdata,
                        lbl_key,
                        ref_key,
                        Labels3DModel,
                        keep_2d=False,
                        lift_mode='centered_slab',
                        lift_n_z=lift_n_z,
                    )

    return plot_sdata

def set_solid_label_color(sdata, table_key, color, col_name='label_color_group'):
    table = sdata[table_key]
    table.obs[col_name] = pd.Categorical(['all'] * table.n_obs)
    table.uns[f'{col_name}_colors'] = np.array([color])

def apply_layer_style(layer, layer_styles):
    import napari
    def _find_style(layer_name):
        """Return params for the longest matching key, or None."""
        matches = [k for k in layer_styles if k in layer_name]
        return layer_styles[max(matches, key=len)] if matches else None
    
    params = _find_style(layer.name)
    if params is None:
        return

    THUMBNAIL_PROPS = {'colormap', 'contrast_limits', 'blending', 'opacity'}
    TYPE_GATES = {
        'colormap':   napari.layers.Image,
        'contour':    napari.layers.Labels,
        'face_color': napari.layers.Points,
    }
    # Keys handled elsewhere — skip silently
    SKIP = {'label_color', 'face_color_column'}

    thumbnail_updates = {k: v for k, v in params.items()
                         if k in THUMBNAIL_PROPS and k not in SKIP}
    other_updates     = {k: v for k, v in params.items()
                         if k not in THUMBNAIL_PROPS and k not in SKIP}

    try:
        for prop, val in thumbnail_updates.items():
            gate = TYPE_GATES.get(prop)
            if gate and not isinstance(layer, gate):
                continue
            if hasattr(layer, prop):
                setattr(layer, prop, val)
    except RuntimeError:
        pass  # 3D layers fail _update_thumbnail; settings apply on render

    for prop, val in other_updates.items():
        gate = TYPE_GATES.get(prop)
        if gate and not isinstance(layer, gate):
            continue
        if hasattr(layer, prop):
            try:
                setattr(layer, prop, val)
            except Exception:
                pass

def add_napari_colormaps(
    sdata,
    column_colors,
    table_key='table',
    points_elements=None,
    points_color_col_suffix='_color',
):
    """
    Register color palettes for categorical columns so that napari-spatialdata
    can use them automatically.

    Table obs columns
    -----------------
    Writes ``table.uns['{column}_colors']`` — the standard AnnData/scanpy
    palette convention that napari-spatialdata reads when you select that column
    for annotation in the GUI.  The list is aligned to
    ``table.obs[column].cat.categories``.

    Points elements (transcripts)
    -----------------------------
    Writes a ``'{column}{suffix}'`` column (default ``'{column}_color'``) into
    the Dask DataFrame with a hex string per row.  In napari you can then choose
    this color column to directly color individual transcripts by gene/feature.

    Parameters
    ----------
    sdata : SpatialData
    column_colors : dict
        ``{column_name: color_spec}`` where *color_spec* is one of:

        * **dict** ``{category_value: color}`` — explicit per-value colors
          (hex strings, RGB tuples, or matplotlib named colors).
        * **str** — a matplotlib colormap name; colors are auto-assigned to
          sorted unique category values.
        * **list** — colors in the same order as the sorted unique categories
          (or ``table.obs[col].cat.categories`` for table columns).

    table_key : str
        Key of the AnnData table to update.  Default ``'table'``.
    points_elements : list[str] | None
        Points element names to add color columns to.
        ``None`` → every element whose name starts with ``'transcripts'``.
    points_color_col_suffix : str
        Suffix appended to the column name when writing into points DataFrames.
        Default ``'_color'``, so ``'feature_name'`` → ``'feature_name_color'``.

    Returns
    -------
    SpatialData
        The same ``sdata`` object (modified in-place).

    Examples
    --------
    >>> add_napari_colormaps(
    ...     sdata,
    ...     column_colors={
    ...         # Cell-type column in table — explicit colors
    ...         'subclass_name': {'L2/3 IT': '#e41a1c', 'Pvalb': '#377eb8', 'L5 IT': '#4daf4a'},
    ...         # Another column — auto-assign from a matplotlib cmap
    ...         'broad_class': 'tab10',
    ...     },
    ...     table_key='table',
    ... )
    >>> add_napari_colormaps(
    ...     sdata,
    ...     column_colors={
    ...         'feature_name': {'EGFP': '#00ff00', 'Snap25': '#ff6600', 'Gad1': '#0066ff'},
    ...     },
    ...     table_key=None,          # skip table
    ...     points_elements=['transcripts-6'],
    ... )
    """

    def _to_hex(color):
        if isinstance(color, str) and color.startswith('#'):
            return color
        return matplotlib.colors.to_hex(color)

    def _resolve_color_map(color_spec, categories):
        """Return {str(category): hex_color} for the given category list."""
        categories = [str(c) for c in categories]
        n = len(categories)

        if isinstance(color_spec, dict):
            return {str(k): _to_hex(v) for k, v in color_spec.items()}

        if isinstance(color_spec, str):
            cmap = plt.get_cmap(color_spec, max(n, 1))
            colors = [cmap(i) for i in range(n)]
        elif isinstance(color_spec, (list, tuple)):
            colors = list(color_spec)
            if len(colors) < n:
                raise ValueError(
                    f"Color list has {len(colors)} entries but there are {n} categories."
                )
        else:
            raise TypeError(
                f"color_spec must be a dict, a matplotlib cmap name (str), or a list; "
                f"got {type(color_spec).__name__}"
            )
        return {cat: _to_hex(c) for cat, c in zip(categories, colors)}

    # ── Table obs columns ────────────────────────────────────────────────────
    if table_key is not None and table_key in sdata.tables:
        tbl = sdata.tables[table_key]
        for col, color_spec in column_colors.items():
            if col not in tbl.obs.columns:
                print(f"  [colormaps] Column '{col}' not in table.obs — skipping")
                continue

            # Ensure categorical so .cat.categories is available
            if not hasattr(tbl.obs[col], 'cat'):
                tbl.obs[col] = tbl.obs[col].astype('category')

            categories = tbl.obs[col].cat.categories
            color_map  = _resolve_color_map(color_spec, categories)

            # AnnData/scanpy convention: list aligned to .cat.categories
            tbl.uns[f'{col}_colors'] = [
                color_map.get(str(cat), '#aaaaaa') for cat in categories
            ]
            print(
                f"  [colormaps] table.uns['{col}_colors'] → "
                f"{len(categories)} categories"
            )

    # ── Points / transcript elements ─────────────────────────────────────────
    if points_elements is None:
        points_elements = [k for k in sdata.points if k.startswith('transcripts')]

    for pt_key in points_elements:
        if pt_key not in sdata.points:
            print(f"  [colormaps] Points element '{pt_key}' not found — skipping")
            continue

        pts = sdata.points[pt_key]
        for col, color_spec in column_colors.items():
            if col not in pts.columns:
                continue

            # Discover unique values (one pass over the lazy frame)
            unique_vals = sorted(
                str(v) for v in pts[col].drop_duplicates().compute().tolist()
            )
            color_map   = _resolve_color_map(color_spec, unique_vals)
            default_hex = '#aaaaaa'

            color_col = col + points_color_col_suffix
            pts[color_col] = pts[col].astype(str).map(
                color_map,
                meta=(color_col, 'str'),
            ).fillna(default_hex)

            sdata.points[pt_key] = pts
            print(
                f"  [colormaps] points['{pt_key}']['{color_col}'] → "
                f"{len(color_map)} gene colors"
            )

    return sdata

def make_column_colormap(
    source,
    column_name,
    colors=None,
    colormap_name='tab20',
    default_color='#808080',
    add_to_uns=False,
):
    import anndata as ad

    # --- Extract the series ---
    if isinstance(source, ad.AnnData):
        series = source.obs[column_name]
    elif isinstance(source, pd.Series):
        series = source
    else:
        # dask or pandas DataFrame
        series = source[column_name]

    # --- Get unique categories ---
    if hasattr(series, 'compute'):
        # dask: use cat.as_known() if categorical, else compute unique
        if hasattr(series, 'cat'):
            categories = series.cat.as_known().cat.categories.tolist()
        else:
            categories = sorted(series.unique().compute().dropna().tolist())
    else:
        if hasattr(series, 'cat'):
            categories = series.cat.categories.tolist()
        else:
            categories = sorted(series.dropna().unique().tolist())

    # --- Build color mapping ---
    cmap = plt.get_cmap(colormap_name, len(categories))
    auto_colors = {cat: mcolors.to_hex(cmap(i)) for i, cat in enumerate(categories)}

    # Override with any explicitly provided colors
    color_map = {cat: colors.get(cat, auto_colors.get(cat, default_color))
                 for cat in categories}
    if colors:
        color_map.update({k: v for k, v in colors.items() if k in color_map})

    # --- Optionally write back to AnnData uns ---
    if add_to_uns:
        if not isinstance(source, ad.AnnData):
            raise ValueError("add_to_uns requires an AnnData source")
        if not hasattr(series, 'cat'):
            source.obs[column_name] = pd.Categorical(source.obs[column_name])
        source.uns[f'{column_name}_colors'] = [
            color_map[cat] for cat in source.obs[column_name].cat.categories
        ]

    return color_map
    
def recolor_tx_layer(viewer, el_name, sdata, color_col, colors_dict=None, cmap='tab20'):
    tx_layer = viewer.layers[el_name]
    tx_colors = make_column_colormap(
        sdata[el_name],
        color_col,
        colors=colors_dict,
        colormap_name=cmap,
    )
    tx_data = sdata[el_name][color_col].compute()
    tx_fns = tx_data.values
    tx_layer.properties = {
        **tx_layer.properties,
        color_col: tx_fns,
    }
    tx_layer.face_color_cycle = tx_colors
    tx_layer.face_color = color_col


def add_cell_matcher(
    viewer,
    gcamp_layer_prefix='gcamp_labels',
    xenium_layer_prefix='cell_labels',
    output_path='cell_matches.csv',
    coord_system='czstack_microns',
    sdata=None,
    gcamp_table_key='gcamp_table',
):
    """
    Add an interactive cell-matching dock widget to an existing napari viewer.

    Hover over a cell in any layer and press a key to record its label ID.
    When both a GCaMP and a Xenium cell are selected, press Enter to save the
    pair. Matches are appended to a CSV after every confirmation.

    Controls
    --------
    G      — record the GCaMP label (gcamp_labels-*) under the cursor
    X      — record the Xenium cell label (cell_labels-*) under the cursor
    Enter  — confirm the pending pair and save to CSV
    Escape — clear the current pending selection

    Parameters
    ----------
    viewer : napari.Viewer
    gcamp_layer_prefix : str
        Prefix used to find the GCaMP labels layer (e.g. 'gcamp_labels').
    xenium_layer_prefix : str
        Prefix used to find the Xenium cell labels layer (e.g. 'cell_labels').
    output_path : str or Path
        CSV file to write/append matches to.  Created on first save; existing
        rows are loaded on startup so you can resume across sessions.

    Returns
    -------
    widget : QWidget
        The dock widget (already added to the viewer).
    matches : list of (int, int)
        Live list of confirmed (gcamp_id, xenium_cell_id) pairs.
    """
    from pathlib import Path as _Path
    import pandas as _pd
    from qtpy.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel,
        QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QShortcut,
        QAbstractItemView,
    )
    from qtpy.QtGui import QKeySequence
    from qtpy.QtCore import Qt

    output_path = _Path(output_path)

    # Each match is (gcamp_id, xenium_cell_id, xenium_layer,
    #                gcamp_z, gcamp_y, gcamp_x, xenium_z, xenium_y, xenium_x)
    _COORD_COLS = ['gcamp_z', 'gcamp_y', 'gcamp_x', 'xenium_z', 'xenium_y', 'xenium_x']
    if output_path.exists():
        _existing = _pd.read_csv(output_path)
        for col in _COORD_COLS:
            if col not in _existing.columns:
                _existing[col] = None
        def _maybe_float(v):
            try:
                return None if _pd.isna(v) else float(v)
            except Exception:
                return None
        matches = [
            (int(r.gcamp_id), int(r.xenium_cell_id), str(r.xenium_layer),
             _maybe_float(r.gcamp_z), _maybe_float(r.gcamp_y), _maybe_float(r.gcamp_x),
             _maybe_float(r.xenium_z), _maybe_float(r.xenium_y), _maybe_float(r.xenium_x))
            for r in _existing.itertuples(index=False)
        ]
    else:
        matches = []

    state = {
        'gcamp_id': None,
        'gcamp_pos': None,   # (z, y, x) world coords at pick time
        'xenium_hits': [],   # list of (xenium_cell_id, xenium_layer) — usually 1, >1 if sections overlap
        'xenium_pos': None,  # (z, y, x) world coords at pick time
        'armed': None,       # None | 'gcamp' | 'xenium'
    }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    import napari as _napari

    def _label_centroid_world(layer, label_val):
        import numpy as _np
        data = layer.data
        if hasattr(data, 'compute'):
            data = data.compute()
        data = _np.asarray(data)
        coords = _np.argwhere(data == int(label_val))
        if len(coords) == 0:
            return None
        centroid_idx = coords.mean(axis=0)
        return tuple(layer.data_to_world(centroid_idx))

    def _gcamp_label_at_pos(pos):
        for layer in viewer.layers:
            if isinstance(layer, _napari.layers.Labels) and layer.name.startswith(gcamp_layer_prefix):
                val = layer.get_value(pos, world=True)
                return val, layer, None
        return None, None, f"No Labels layer with prefix '{gcamp_layer_prefix}' found."

    def _xenium_label_at_pos(pos):
        """Returns (hits, hit_layers, warning) where hits is a list of (cell_id, layer_name).
        If multiple layers overlap, all are returned with a warning string."""
        hits = []
        hit_layers = []
        found_any = False
        for layer in viewer.layers:
            if isinstance(layer, _napari.layers.Labels) and layer.name.startswith(xenium_layer_prefix):
                found_any = True
                val = layer.get_value(pos, world=True)
                if val is not None and val != 0:
                    hits.append((int(val), layer.name))
                    hit_layers.append(layer)
        if not found_any:
            return [], [], f"No Labels layer with prefix '{xenium_layer_prefix}' found."
        if not hits:
            return [], [], "Clicked background in all Xenium layers. Try again."
        if len(hits) > 1:
            names = ', '.join(f"{n} (id={v})" for v, n in hits)
            return hits, hit_layers, f"Warning: overlapping sections — will save {len(hits)} rows: {names}"
        return hits, hit_layers, None

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    widget = QWidget()
    layout = QVBoxLayout()
    widget.setLayout(layout)

    pick_row = QHBoxLayout()
    pick_gcamp_btn = QPushButton("Pick GCaMP cell")
    pick_xenium_btn = QPushButton("Pick Xenium cell")
    pick_row.addWidget(pick_gcamp_btn)
    pick_row.addWidget(pick_xenium_btn)
    layout.addLayout(pick_row)

    instructions = QLabel(
        "Click a pick button, then click the cell in the viewer.<br>"
        "<b>Enter</b> = confirm pair &nbsp;&nbsp; <b>Esc</b> = cancel pick / clear"
    )
    instructions.setWordWrap(True)
    layout.addWidget(instructions)

    pending_label = QLabel("Pending: GCaMP=—  |  Xenium=—")
    pending_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #888;")
    layout.addWidget(pending_label)

    dupe_label = QLabel("")
    dupe_label.setStyleSheet("font-size: 12px; color: #3af; font-style: italic;")
    dupe_label.setWordWrap(True)
    layout.addWidget(dupe_label)

    table = QTableWidget(0, 3)
    table.setHorizontalHeaderLabels(['GCaMP ID', 'Xenium Cell ID', 'Xenium Layer'])
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    table.setEditTriggers(QTableWidget.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setSelectionMode(QAbstractItemView.ExtendedSelection)
    table.setMaximumHeight(220)
    layout.addWidget(table)

    btn_row = QHBoxLayout()
    confirm_btn    = QPushButton("Confirm  (Enter)")
    clear_btn      = QPushButton("Clear  (Esc)")
    undo_btn       = QPushButton("Undo last")
    delete_sel_btn = QPushButton("Delete selected rows")
    for b in (confirm_btn, clear_btn, undo_btn, delete_sel_btn):
        btn_row.addWidget(b)
    layout.addLayout(btn_row)

    update_completed_btn = QPushButton("Update completed cells")
    layout.addWidget(update_completed_btn)

    status = QLabel("")
    status.setWordWrap(True)
    layout.addWidget(status)

    count_label = QLabel()
    layout.addWidget(count_label)

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------
    _ARMED_GCAMP_STYLE  = "font-size: 12px; padding: 5px; background: #1a6a8a; color: white; font-weight: bold;"
    _ARMED_XENIUM_STYLE = "font-size: 12px; padding: 5px; background: #6a3a8a; color: white; font-weight: bold;"
    _UNARMED_STYLE      = "font-size: 12px; padding: 5px;"

    def _refresh_armed():
        armed = state['armed']
        pick_gcamp_btn.setStyleSheet(_ARMED_GCAMP_STYLE  if armed == 'gcamp'  else _UNARMED_STYLE)
        pick_xenium_btn.setStyleSheet(_ARMED_XENIUM_STYLE if armed == 'xenium' else _UNARMED_STYLE)

    def _refresh_table():
        table.setSortingEnabled(False)
        table.setRowCount(len(matches))
        for row, m in enumerate(matches):
            g, x, lyr = m[0], m[1], m[2]
            item_g = QTableWidgetItem(str(g))
            item_g.setData(Qt.UserRole, row)  # original index survives sort
            table.setItem(row, 0, item_g)
            table.setItem(row, 1, QTableWidgetItem(str(x)))
            table.setItem(row, 2, QTableWidgetItem(str(lyr)))
        table.setSortingEnabled(True)
        table.scrollToBottom()
        count_label.setText(f"{len(matches)} pairs saved → {output_path}")

    def _refresh_pending():
        g    = state['gcamp_id']
        hits = state['xenium_hits']
        gstr = str(g) if g is not None else "—"
        if not hits:
            xstr = "—"
        elif len(hits) == 1:
            xstr = f"{hits[0][0]} ({hits[0][1]})"
        else:
            xstr = f"{', '.join(str(v) for v, _ in hits)} ({len(hits)} layers)"
        if g is None and not hits:
            pending_label.setText("Pending: GCaMP=—  |  Xenium=—")
            pending_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #888;")
        elif g is not None and hits:
            pending_label.setText(f"Pending: GCaMP={gstr}  |  Xenium={xstr}  ← Enter")
            pending_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #2ecc71;")
        else:
            pending_label.setText(f"Pending: GCaMP={gstr}  |  Xenium={xstr}")
            pending_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #e67e22;")

    def _save():
        df = _pd.DataFrame(matches, columns=[
            'gcamp_id', 'xenium_cell_id', 'xenium_layer',
            'gcamp_z', 'gcamp_y', 'gcamp_x',
            'xenium_z', 'xenium_y', 'xenium_x',
        ])
        df['coord_system'] = coord_system
        df.to_csv(output_path, index=False)

    def _arm(cell_type):
        """Toggle armed state for a cell type; clicking again cancels."""
        if state['armed'] == cell_type:
            state['armed'] = None
            status.setText("Pick cancelled.")
        else:
            state['armed'] = cell_type
            label = 'GCaMP' if cell_type == 'gcamp' else 'Xenium'
            status.setText(f"Armed: click on a {label} cell in the viewer.")
        _refresh_armed()

    def _check_dupes():
        """Update dupe_label based on current pending picks vs existing matches."""
        existing_gcamps  = {m[0] for m in matches}
        existing_xeniums = {m[1] for m in matches}
        msgs = []
        g = state['gcamp_id']
        if g is not None and g in existing_gcamps:
            msgs.append(f"GCaMP {g} already in table")
        for x, _ in state['xenium_hits']:
            if x in existing_xeniums:
                msgs.append(f"Xenium {x} already in table")
        dupe_label.setText("  ⚠ " + " | ".join(msgs) if msgs else "")

    def _confirm():
        g    = state['gcamp_id']
        hits = state['xenium_hits']
        if g is None or not hits:
            status.setText("Need both GCaMP and Xenium IDs before confirming.")
            return
        gp = state['gcamp_pos'] or (None, None, None)
        xp = state['xenium_pos'] or (None, None, None)
        for x, lyr in hits:
            matches.append((int(g), int(x), str(lyr),
                            gp[0], gp[1], gp[2],
                            xp[0], xp[1], xp[2]))
        _save()
        _refresh_table()
        state['gcamp_id'] = None
        state['gcamp_pos'] = None
        state['xenium_hits'] = []
        state['xenium_pos'] = None
        dupe_label.setText("")
        _refresh_pending()
        saved_str = ", ".join(f"Xenium {x}" for x, _ in hits)
        status.setText(f"Saved: GCaMP {g}  ↔  {saved_str}")

    def _clear():
        state['gcamp_id'] = None
        state['gcamp_pos'] = None
        state['xenium_hits'] = []
        state['xenium_pos'] = None
        state['armed'] = None
        dupe_label.setText("")
        _refresh_armed()
        _refresh_pending()
        status.setText("Cleared.")

    def _undo():
        if not matches:
            status.setText("Nothing to undo.")
            return
        removed = matches.pop()
        _save()
        _refresh_table()
        status.setText(f"Undid last row: GCaMP {removed[0]}  ↔  Xenium {removed[1]}  ({removed[2]})")

    def _update_completed_cells():
        matched_gcamp_ids = {m[0] for m in matches}

        gcamp_layer = next(
            (l for l in viewer.layers
             if isinstance(l, _napari.layers.Labels) and l.name.startswith(gcamp_layer_prefix)),
            None,
        )
        if gcamp_layer is None:
            status.setText("Could not find GCaMP labels layer.")
            return

        if sdata is None or gcamp_table_key not in sdata.tables:
            status.setText("Pass sdata= to add_cell_matcher to enable completed-cell highlighting.")
            return

        gcamp_table = sdata.tables[gcamp_table_key]
        attrs = gcamp_table.uns.get('spatialdata_attrs', {})
        instance_key = attrs.get('instance_key', 'label')

        if instance_key not in gcamp_table.obs.columns:
            status.setText(f"Instance key '{instance_key}' not found in gcamp_table.obs.")
            return

        label_ids = gcamp_table.obs[instance_key].values

        # Update the active napari-spatialdata annotation column so re-application
        # callbacks produce the correct colors if the user interacts with the GUI.
        gcamp_table.obs['cell_labels_color'] = pd.Categorical(
            ['matched' if int(lid) in matched_gcamp_ids else 'unmatched' for lid in label_ids],
            categories=['unmatched', 'matched'],
        )
        gcamp_table.uns['cell_labels_color_colors'] = np.array(['#2323ff', '#ff0000'])

        # Also persist for downstream analysis
        gcamp_table.obs['is_matched'] = gcamp_table.obs['cell_labels_color']
        gcamp_table.uns['is_matched_colors'] = gcamp_table.uns['cell_labels_color_colors'].copy()

        # napari >=0.4.20 uses DirectLabelColormap on layer.colormap, not layer.color
        from collections import defaultdict
        from napari.utils.colormaps import DirectLabelColormap

        red         = np.array([1.0, 0.0, 0.0, 1.0])
        blue        = np.array([35/255, 35/255, 1.0, 1.0])   # #2323ff
        transparent = np.zeros(4)

        cmap_dict = defaultdict(lambda: transparent)
        for lid in label_ids:
            cmap_dict[int(lid)] = red if int(lid) in matched_gcamp_ids else blue

        gcamp_layer.colormap = DirectLabelColormap(color_dict=cmap_dict)

        n = len(matched_gcamp_ids)
        status.setText(f"Updated: {n} matched GCaMP {'cell' if n == 1 else 'cells'} highlighted.")

    def _delete_selected():
        selected_rows = {idx.row() for idx in table.selectedIndexes()}
        if not selected_rows:
            status.setText("No rows selected.")
            return
        orig_indices = set()
        for row in selected_rows:
            item = table.item(row, 0)
            if item is not None:
                orig_indices.add(item.data(Qt.UserRole))
        for idx in sorted(orig_indices, reverse=True):
            matches.pop(idx)
        _save()
        _refresh_table()
        status.setText(f"Deleted {len(orig_indices)} row(s).")

    def _navigate_to(row, col):
        item = table.item(row, 0)
        if item is None:
            return
        orig_idx = item.data(Qt.UserRole)
        if orig_idx is None or orig_idx >= len(matches):
            return
        m = matches[orig_idx]
        # m[3:6] = gcamp z,y,x ; m[6:9] = xenium z,y,x
        pos = m[3:6] if col == 0 else m[6:9]
        if any(p is None for p in pos):
            status.setText("No coordinates stored for this row (loaded from older CSV).")
            return
        z, y, x = pos
        # z is stored as the integer step index at pick time; restore it directly.
        # For a 3D labels layer in an N-dim viewer, z lives at viewer axis (ndim - 3).
        # camera.center uses world coordinates (y, x) in 2D mode.
        z_step = int(round(z))
        z_axis = max(0, viewer.dims.ndim - 3)
        steps = list(viewer.dims.current_step)
        steps[z_axis] = z_step
        viewer.dims.current_step = tuple(steps)
        viewer.camera.center = (y, x)
        label = 'GCaMP' if col == 0 else 'Xenium'
        status.setText(f"Navigated to {label} cell — z_step={z_step} (axis {z_axis}), y={y:.2f}, x={x:.2f}")

    def _show_row_coords():
        selected = table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        item = table.item(row, 0)
        if item is None:
            return
        orig_idx = item.data(Qt.UserRole)
        if orig_idx is None or orig_idx >= len(matches):
            return
        m = matches[orig_idx]
        gz, gy, gx = m[3], m[4], m[5]
        xz, xy, xx = m[6], m[7], m[8]
        def _fmt(v):
            return f"{v:.2f}" if v is not None else "?"
        status.setText(
            f"GCaMP {m[0]}: z={_fmt(gz)}, y={_fmt(gy)}, x={_fmt(gx)}  |  "
            f"Xenium {m[1]}: z={_fmt(xz)}, y={_fmt(xy)}, x={_fmt(xx)}"
        )

    # Canvas click handler — fires on every left-click while armed.
    # napari mouse_drag_callbacks must be generator functions.
    def _on_canvas_click(viewer_obj, event):
        if event.button == 1 and state['armed'] is not None:
            armed = state['armed']
            pos = viewer_obj.cursor.position
            state['armed'] = None
            _refresh_armed()
            if armed == 'gcamp':
                val, layer, err = _gcamp_label_at_pos(pos)
                if err:
                    status.setText(err)
                elif val is None or val == 0:
                    status.setText("Clicked background in GCaMP layer. Try again.")
                else:
                    state['gcamp_id'] = int(val)
                    centroid = _label_centroid_world(layer, val)
                    z_axis = viewer.dims.ndim - layer.ndim
                    z_step = float(viewer.dims.current_step[z_axis])
                    state['gcamp_pos'] = (z_step, centroid[1], centroid[2]) if centroid else None
                    _refresh_pending()
                    _check_dupes()
                    status.setText(f"GCaMP cell {val} picked.")
            else:
                hits, hit_layers, warn = _xenium_label_at_pos(pos)
                if not hits:
                    status.setText(warn)
                else:
                    state['xenium_hits'] = hits
                    # Use centroid of first hit layer for navigation (representative position)
                    centroid = _label_centroid_world(hit_layers[0], hits[0][0])
                    z_axis = viewer.dims.ndim - hit_layers[0].ndim
                    z_step = float(viewer.dims.current_step[z_axis])
                    state['xenium_pos'] = (z_step, centroid[1], centroid[2]) if centroid else None
                    _refresh_pending()
                    _check_dupes()
                    if warn:
                        status.setText(warn)
                    elif len(hits) == 1:
                        status.setText(f"Xenium cell {hits[0][0]} picked from {hits[0][1]}.")
                    else:
                        status.setText(f"{len(hits)} overlapping Xenium cells picked — will save {len(hits)} rows on confirm.")
        yield

    viewer.mouse_drag_callbacks.append(_on_canvas_click)

    table.cellDoubleClicked.connect(_navigate_to)
    table.itemSelectionChanged.connect(_show_row_coords)

    pick_gcamp_btn.clicked.connect(lambda: _arm('gcamp'))
    pick_xenium_btn.clicked.connect(lambda: _arm('xenium'))
    confirm_btn.clicked.connect(_confirm)
    clear_btn.clicked.connect(_clear)
    undo_btn.clicked.connect(_undo)
    delete_sel_btn.clicked.connect(_delete_selected)
    update_completed_btn.clicked.connect(_update_completed_cells)

    _refresh_table()
    _refresh_pending()
    _refresh_armed()

    # Return and Escape use application-wide shortcuts so they fire
    # regardless of whether napari canvas or the dock widget has focus.
    _sc_confirm = QShortcut(QKeySequence(Qt.Key_Return), widget)
    _sc_confirm.setContext(Qt.ApplicationShortcut)
    _sc_confirm.activated.connect(_confirm)

    _sc_clear = QShortcut(QKeySequence(Qt.Key_Escape), widget)
    _sc_clear.setContext(Qt.ApplicationShortcut)
    _sc_clear.activated.connect(_clear)

    # ------------------------------------------------------------------
    # Dock the widget
    # ------------------------------------------------------------------
    viewer.window.add_dock_widget(widget, name='Cell Matcher', area='right')

    return widget, matches
