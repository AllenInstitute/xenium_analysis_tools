import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors

######## Functions for adding mapping info to sdata
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

def map_to_broad_subclass_name(subclass_name, patterns):
    if pd.isna(subclass_name):
        return 'Other'
    
    subclass_str = str(subclass_name).strip()
    
    for broad_name, pattern_list in patterns.items():
        for pattern in pattern_list:
            # Exact match (case insensitive)
            if subclass_str.lower() == pattern.lower():
                return broad_name
            
            # Contains match (case insensitive)
            if pattern.lower() in subclass_str.lower():
                return broad_name
    return 'Other'

def add_broad_types(adata):
    broad_subclass_name_patterns = {
        'L2/3 IT': ['L2/3 IT', 'L2/3IT', 'L23 IT', 'L2-3 IT'],
        'L4/5 IT': ['L4/5 IT', 'L4/5IT', 'L4 IT', 'L45 IT', 'L4-5 IT'],
        'L5 IT': ['L5 IT', 'L5IT'],
        'L5 ET': ['L5 ET', 'L5ET'],
        'L6 IT': ['L6 IT', 'L6IT'],
        'L6 CT': ['L6 CT', 'L6CT'],
        'Lamp5': ['Lamp5', 'LAMP5'],
        'Vip': ['Vip', 'VIP'],
        'Sncg': ['Sncg', 'SNCG'],
        'Sst': ['Sst', 'SST'],
        'Pvalb': ['Pvalb', 'PVALB', 'PV'],
        'NN': ['NN', 'Neuronal Non-Neuronal', 'Neuronal-Non-Neuronal'],
        'L6b': ['L6b', 'L6-b', 'L6 b'],
        'L2 IT': ['L2 IT', 'L2IT', 'L2-IT'],
        'Ob-in': ['Ob-in', 'OB-in', 'OB in', 'Ob in'],
        'STR': ['STR Prox1 Lhx6', 'STR Prox1 LHX6', 'STR Prox1-Lhx6', 'STR Prox1-LHX6'],
        'CB Gran': ['CB Gran', 'CB Granule', 'CB Granular', 'CBGran'],
    }

    adata.obs['broad_class_name'] = 'Other'
    adata.obs.loc[adata.obs['class_name'].str.contains('Glut', case=False, na=False), 'broad_class_name'] = 'Glut'
    adata.obs.loc[adata.obs['class_name'].str.contains('GABA', case=False, na=False), 'broad_class_name'] = 'GABA'
    
    def parse_numeric_class(class_name):
        try:
            if pd.isna(class_name):
                return False
            parts = str(class_name).strip().split(' ')
            if len(parts) > 0 and parts[0].isdigit():
                return int(parts[0]) >= 30
            return False
        except (ValueError, IndexError):
            return False
    
    adata.obs.loc[adata.obs['class_name'].apply(parse_numeric_class), 'broad_class_name'] = 'NN'

    adata.obs['broad_subclass_name'] = adata.obs['subclass_name'].apply(
        lambda x: map_to_broad_subclass_name(x, broad_subclass_name_patterns)
    )
    adata.obs['broad_class_name'] = adata.obs['broad_class_name'].astype('category')
    adata.obs['broad_subclass_name'] = adata.obs['broad_subclass_name'].astype('category')

    return adata

def get_shared_colormap(data,
                        gaba_palette='plasma',
                        glut_palette='cividis',
                        nn_palette='copper'):
    joined_colormap = {}

    def _add_color_for_subclasses_supertypes(subclass_data, broad_subclass_names, cmap):
        for i, sbcl in enumerate(broad_subclass_names):
            # Convert to Hex string
            broad_subclass_name_col = mcolors.to_hex(cmap[i])
            joined_colormap[sbcl] = broad_subclass_name_col
            
            # Get subclasses
            subclass_mask = subclass_data['broad_subclass_name'] == sbcl
            subclasses = sorted(subclass_data.loc[subclass_mask, 'subclass_name'].unique())

            # All subclasses of the same broad_subclass_name assigned the same color
            for subclass in subclasses:
                joined_colormap[subclass] = broad_subclass_name_col

            # Supertypes as assigned progressively desaturated versions of the broad_subclass_name color
            supertypes = sorted(subclass_data.loc[subclass_mask, 'supertype_name'].unique())
            saturation_factors = np.linspace(1.0, 0.4, num=len(supertypes))
            for supertype, saturation_factor in zip(supertypes, saturation_factors):
                desaturated = sns.desaturate(cmap[i], saturation_factor)
                joined_colormap[supertype] = mcolors.to_hex(desaturated)

    # GABA types
    joined_colormap['GABA'] = mcolors.to_hex((0.890, 0.102, 0.110)) # tab:red
    gaba_data = data.obs[data.obs['broad_class_name'] == 'GABA']
    gaba_subclasses = sorted(gaba_data['broad_subclass_name'].unique())
    n_gaba_subclasses = len(gaba_subclasses)
    cmap = sns.color_palette(gaba_palette, n_colors=n_gaba_subclasses)
    _add_color_for_subclasses_supertypes(gaba_data, gaba_subclasses, cmap)

    # Glut types
    joined_colormap['Glut'] = mcolors.to_hex((0.121, 0.466, 0.705)) # tab:blue
    glut_data = data.obs[data.obs['broad_class_name'] == 'Glut']
    glut_subclasses = sorted(glut_data['broad_subclass_name'].unique())
    n_glut_subclasses = len(glut_subclasses)
    cmap = sns.color_palette(glut_palette, n_colors=n_glut_subclasses)
    _add_color_for_subclasses_supertypes(glut_data, glut_subclasses, cmap)

    # NN types
    joined_colormap['NN'] = mcolors.to_hex((0.549, 0.337, 0.294)) #'tab:brown'
    nn_data = data.obs[data.obs['broad_class_name'] == 'NN']
    nn_subclasses = sorted(nn_data['broad_subclass_name'].unique())
    n_nn_subclasses = len(nn_subclasses)
    cmap = sns.color_palette(nn_palette, n_colors=n_nn_subclasses)
    _add_color_for_subclasses_supertypes(nn_data, nn_subclasses, cmap)

    return joined_colormap

def add_colormap_adata(adata, colormap):
    def validate_color(color):
        """Ensure the color is a valid format (hex string or RGB tuple)."""
        if isinstance(color, str) and color.startswith("#"):
            return color  # Valid hex color
        elif isinstance(color, (tuple, list)) and len(color) in [3, 4]:
            return tuple(color)  # Valid RGB(A) tuple
        else:
            return 'gray'  # Default to 'gray' if invalid

    broad_class_name_colors = []
    for tp in sorted(adata.obs['broad_class_name'].unique()):
        color = validate_color(colormap.get(tp, 'gray'))
        broad_class_name_colors.append(color)

    broad_subclass_name_colors = []
    for tp in sorted(adata.obs['broad_subclass_name'].unique()):
        color = validate_color(colormap.get(tp, 'gray'))
        broad_subclass_name_colors.append(color)

    subclass_colors = []
    for tp in sorted(adata.obs['subclass_name'].unique()):
        color = validate_color(colormap.get(tp, 'gray'))
        subclass_colors.append(color)

    supertype_colors = []
    for tp in sorted(adata.obs['supertype_name'].unique()):
        color = validate_color(colormap.get(tp, 'gray'))
        supertype_colors.append(color)

    # Store as numpy arrays
    adata.uns['broad_class_name_colors'] = np.array(broad_class_name_colors, dtype=object)
    adata.uns['broad_subclass_name_colors'] = np.array(broad_subclass_name_colors, dtype=object)
    adata.uns['supertype_name_colors'] = np.array(supertype_colors, dtype=object)
    adata.uns['subclass_name_colors'] = np.array(subclass_colors, dtype=object)

    return adata