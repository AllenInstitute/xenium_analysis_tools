import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors
######## Functions for adding mapping info to sdata
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