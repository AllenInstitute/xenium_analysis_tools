import warnings

# Suppress noisy cell-type-mapper warnings
warnings.filterwarnings("ignore", message=".*not listed in marker lookup.*")
warnings.filterwarnings("ignore", message=".*had too few markers in query set.*")

from cell_type_mapper.cli.transcribe_to_obs import TranscribeToObsRunner
from cell_type_mapper.cli.validate_h5ad import ValidateH5adRunner

import json
import spatialdata as sd
import pandas as pd
import numpy as np
import anndata as ad
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from pathlib import Path
from tqdm import tqdm

def combine_sections_adatas(sections_paths):
    # Initialize
    sections_adatas: list[ad.AnnData] = []

    # Loop through sections
    for s_path in tqdm(sections_paths, desc='Loading section adatas'):
        # Load anndata from table
        adata = ad.read_zarr(s_path / 'tables' / 'table')

        # Format section/cell ids
        section_num = int(s_path.stem.split('_')[1])
        adata.obs['section'] = section_num
        adata.obs['original_cell_id'] = adata.obs['cell_id']
        adata.obs['cell_id'] = [f"{c_id}_{sec}" for c_id, sec in zip(adata.obs['cell_id'], adata.obs['section'])]
        adata.obs.set_index('cell_id', inplace=True, drop=False)

        # Clear bulky containers
        adata.uns = {}
        adata.obsm = {}
        sections_adatas.append(adata)
        del adata
    
    # Combine all sections into a single AnnData object
    sections_adatas = ad.concat(
        sections_adatas, 
        axis=0, 
        join="outer", 
        merge="same"
    )
    sections_adatas.var['gene_symbol'] = sections_adatas.var.index
    sections_adatas.var.set_index(sections_adatas.var['gene_ids'], inplace=True, drop=False)

    return sections_adatas


def get_drop_nodes(v1_types_config, abc_cache, save_plot=False, output_folder=None):
    # Taxonomy filters - nodes to drop for mapping
    nodes_to_drop=[]

    # If specified any specific nodes to drop in config
    drop_nodes_dict = v1_types_config.get('drop_nodes_dict', None) 
    if drop_nodes_dict:
        for h_level in drop_nodes_dict:
            nodes_to_drop.extend([(h_level, cl) for cl in drop_nodes_dict[h_level]])
        print(f"Dropping {len(nodes_to_drop)} nodes based on drop_nodes_dict.")

    print("Filtering to only include V1 cell type nodes...")
    v1_types_path = v1_types_config.get('v1_types_path', '/root/capsule/code/v1_merfish_cells.csv')
    h_level = v1_types_config.get('h_level', 'subclass')
    min_cells = v1_types_config.get('min_cells', 0)
    if Path(v1_types_path).exists():
        print(f"Loading V1 MERFISH cell types from {v1_types_path}...")
        v1_merfish_cells = pd.read_csv(v1_types_path)
    else:
        print(f"V1 MERFISH cell types file not found at {v1_types_path}. Attempting to generate it from ABC cache...")
        v1_merfish_cells = get_v1_merfish_cells(abc_cache, df_path=v1_types_path)
    if save_plot:
        plot_cell_counts_heatmap(v1_merfish_cells, min_cells=min_cells, save_path=output_folder / 'v1_merfish_cell_counts_heatmap.svg')

    # Filter out specified layers if provided in config
    if v1_types_config.get('drop_layers', None):
        v1_merfish_cells = v1_merfish_cells.loc[~v1_merfish_cells['parcellation_substructure'].isin(v1_types_config.get('drop_layers'))]
    
    # Filter df to only include rows where cell count is above min_cells threshold if specified
    v1_nodes_to_drop = get_nodes_to_drop(v1_merfish_cells, abc_cache, h_level=h_level, min_cells=min_cells)
    print(f"Dropping {len(v1_nodes_to_drop)} {h_level} nodes not present in V1 MERFISH data with at least {min_cells if min_cells>0 else 1} cell(s).")
    nodes_to_drop.extend(v1_nodes_to_drop)
    
    return nodes_to_drop

def get_abc_paths(abc_cache):
    try:
        precomputed_stats_path = abc_cache.get_data_path(   
            directory='WMB-10X',
            file_name='precomputed_stats_ABC_revision_230821'
        )
        mouse_markers_path = abc_cache.get_data_path(
            directory='WMB-10X',
            file_name='mouse_markers_230821'
        )
        gene_mapper_db_path = abc_cache.get_data_path(
            directory='mmc-gene-mapper',
            file_name='mmc_gene_mapper.2025-08-04'
        )
    except Exception as e:
        precomputed_stats_path = '/root/capsule/data/abc_atlas/mapmycells/WMB-10X/20240831/precomputed_stats_ABC_revision_230821.h5'
        mouse_markers_path = '/root/capsule/data/abc_atlas/mapmycells/WMB-10X/20240831/mouse_markers_230821.json'
        gene_mapper_db_path = '/root/capsule/data/abc_atlas/mapmycells/mmc-gene-mapper/20250630/mmc_gene_mapper.2025-08-04.db'
    return precomputed_stats_path, mouse_markers_path, gene_mapper_db_path

    
def get_v1_merfish_cells(abc_cache=None, df_path=None):
    if df_path and df_path.exists():
        v1_merfish_cells = pd.read_csv(df_path, index_col=0)
    else:
        # Get MERFISH CCF metadata
        print('V1 cell df not found, generating new one...')
        if abc_cache is None:
            raise ValueError("abc_cache must be provided if path to df does not exist")
        merfish_ccf_metadata = abc_cache.get_metadata_dataframe(
                    directory='MERFISH-C57BL6J-638850-CCF', 
                    file_name='cell_metadata_with_parcellation_annotation'
                ).set_index('cell_label')
        v1_merfish_cells = merfish_ccf_metadata.loc[merfish_ccf_metadata['parcellation_structure']=='VISp']
        # Save created df
        print(f"Saving df to: {df_path}")
        v1_merfish_cells.to_csv(df_path)
    return v1_merfish_cells

def get_nodes_to_drop(cells_df, abc_cache, h_level='subclass', min_cells=0):
    # Load the taxonomy
    taxonomy_df = abc_cache.get_metadata_dataframe(
                        directory='WMB-taxonomy',
                        file_name='cluster_to_cluster_annotation_membership'
                    )
    # Group the cells by the specified hierarchy level
    grouped_cells = cells_df.groupby(h_level).size()

    # Identify clusters below the minimum cell threshold
    filtered_clusters = grouped_cells[grouped_cells>=min_cells].index.tolist()

    # Get the valid clusters from the taxonomy
    valid_clusters = [np.unique(taxonomy_df.loc[taxonomy_df['cluster_annotation_term_name']==cl,'cluster_annotation_term_label'])[0] for cl in filtered_clusters]

    # Determine nodes to drop
    clusters_to_drop = np.setdiff1d(taxonomy_df.loc[taxonomy_df['cluster_annotation_term_set_name']==h_level,'cluster_annotation_term_label'].unique(), valid_clusters)
    nodes_to_drop = [(h_level, cl) for cl in clusters_to_drop]
    return nodes_to_drop

def validate_input_adata(h5ad_path, output_dir, mouse_markers_path, db_path, round_to_int=False, layer='X', output_json=None):
    if output_json is None:
        output_json = tempfile.mkstemp(suffix='.json')[1]
    validate_config = {
        'h5ad_path': str(h5ad_path),
        'round_to_int': round_to_int,
        'layer': layer,
        'output_dir': str(output_dir),
        'output_json': output_json,
        'gene_mapping': {'db_path': db_path}
    }
    try:
        print("Starting validation of input data...")
        validation_runner = ValidateH5adRunner(args=[], input_data=validate_config)
        validation_runner.run()
        validated_path = json.load(open(validate_config['output_json'], 'rb'))['valid_h5ad_path']
        markers_json = json.load(open(mouse_markers_path, 'rb'))
        markers = []
        for key in list(markers_json.keys())[:-2]:
            markers += markers_json[key]
        use_validated_h5ad = any('ENSMUSG' in marker for marker in markers)
        if use_validated_h5ad:
            print(f"Using validated path: {validated_path}")
            return validated_path
        else:
            print(f'Using original path: {h5ad_path}')
            return h5ad_path
    except Exception as e:
        print(f"Validation failed: {e}")
        raise e

def format_mapping_outputs(extended_results_path, mapped_adata_path, mapping_params, h5ad_path=None):
    try:
        if h5ad_path is None:
            with open(extended_results_path, 'r') as f:
                extended_results = json.load(f)
            h5ad_path = extended_results['config']['query_path']
        ad_config = {
            'h5ad_path': str(h5ad_path), 
            'result_path': str(extended_results_path), 
            'new_h5ad_path': str(mapped_adata_path),
            'clobber': bool(mapping_params['clobber'])
        }
        TranscribeToObsRunner(args=[], input_data=ad_config).run()
        return True
    except Exception as e:
        print(f"Formatting mapping outputs failed: {e}")
        return False


def plot_cell_counts_heatmap(v1_merfish_cells, min_cells=5, save_path=None):
    if isinstance(v1_merfish_cells, str) or isinstance(v1_merfish_cells, Path):
        v1_merfish_cells = pd.read_csv(v1_merfish_cells)

    # Get the cell counts
    cell_counts = v1_merfish_cells.groupby(['subclass','parcellation_substructure']).size().reset_index(name='cell_count')
    pivot_data = cell_counts.pivot(index='parcellation_substructure', columns='subclass', values='cell_count').fillna(0)

    # Add total row (sum across all parcellation substructures)
    total_row = pivot_data.sum(axis=0)
    total_row.name = 'Total (All Layers)'
    pivot_data_with_total = pd.concat([pivot_data, total_row.to_frame().T])

    # Create the plot with total row
    plt.figure(figsize=(20, 6))
    ax = sns.heatmap(pivot_data_with_total, annot=True, fmt='.0f', cmap='viridis', 
                    cbar_kws={'label': 'Cell Count'})
    plt.title('Cell Count Heatmap: Subclass vs Parcellation Substructure')
    plt.xlabel('Subclass')
    plt.ylabel('Parcellation Substructure')

    # Highlight the total row with a different color or border
    ax.axhline(y=len(pivot_data), color='white', linewidth=3)

    # Add boxes around columns where total is below threshold
    for i, (subclass, total_count) in enumerate(total_row.items()):
        if total_count < min_cells:
            # Draw rectangle around entire column
            rect = Rectangle((i, 0), 1, len(pivot_data_with_total), 
                            linewidth=0, edgecolor='None', facecolor='gray', 
                            linestyle='-', alpha=0.65)
            ax.add_patch(rect)

    plt.tight_layout()

    # Save as SVG if requested
    if save_path is not None:
        plt.savefig(str(save_path), format='svg', bbox_inches='tight')

    plt.show()
