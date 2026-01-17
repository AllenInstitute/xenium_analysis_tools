import warnings

# Suppress noisy cell-type-mapper warnings
warnings.filterwarnings("ignore", message=".*not listed in marker lookup.*")
warnings.filterwarnings("ignore", message=".*had too few markers in query set.*")

from cell_type_mapper.cli.from_specified_markers import FromSpecifiedMarkersRunner
from cell_type_mapper.cli.transcribe_to_obs import TranscribeToObsRunner
from cell_type_mapper.cli.validate_h5ad import ValidateH5adRunner
import tempfile

import gc
import json
import spatialdata as sd
import pandas as pd
import numpy as np
import anndata as ad
import tempfile

def get_sections_to_process(sections_path, select_sections=None):
    """Get list of section paths to process."""
    if not sections_path.exists():
        raise FileNotFoundError(f"Sections path does not exist: {sections_path}")
    
    all_section_zarrs = sorted([p for p in sections_path.iterdir() if p.suffix == '.zarr'])
    
    if not all_section_zarrs:
        raise ValueError(f"No .zarr files found in {sections_path}")
    
    if not select_sections:
        return all_section_zarrs
    
    name_to_path = {p.name: p for p in all_section_zarrs}
    available_numbers = [p.stem.split('_')[-1] for p in all_section_zarrs]
    
    selected_zarrs = []
    for n in select_sections:
        section_name = f"section_{n}.zarr"
        if section_name in name_to_path:
            selected_zarrs.append(name_to_path[section_name])
        else:
            print(f"Warning: Section {n} not found. Available: {available_numbers}")
    
    if not selected_zarrs:
        raise ValueError("No valid sections found to process")
    
    return sorted(selected_zarrs, key=lambda p: int(p.stem.split('_')[-1]))
    
def get_v1_merfish_cells(abc_cache=None, output_path=None):
    if output_path and output_path.exists():
        v1_merfish_cells = pd.read_csv(output_path, index_col=0)
    else:
        # Get MERFISH CCF metadata
        print('V1 cell df not found, generating new one...')
        if abc_cache is None:
            raise ValueError("abc_cache must be provided if output_path does not exist")
        merfish_ccf_metadata = abc_cache.get_metadata_dataframe(
                    directory='MERFISH-C57BL6J-638850-CCF', 
                    file_name='cell_metadata_with_parcellation_annotation'
                ).set_index('cell_label')
        v1_merfish_cells = merfish_ccf_metadata.loc[merfish_ccf_metadata['parcellation_structure']=='VISp']
        if output_path:
            print(f'Saving df to: {output_path}")
            v1_merfish_cells.to_csv(output_path)
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

def get_filter_masks(adata, var_filters=None, obs_filters=None):
    gene_mask = np.ones(adata.n_vars, dtype=bool)
    cell_mask = np.ones(adata.n_obs, dtype=bool)
    if var_filters:
        for var_key, var_filter in var_filters.items():
            if var_key not in adata.var.columns:
                continue
            if var_filter.get('eq', None) is not None:
                gene_mask &= (adata.var[var_key] == var_filter['eq']).values
            if var_filter.get('in', None) is not None:
                gene_mask &= adata.var[var_key].isin(var_filter['in']).values
            if var_filter.get('gt', None) is not None:
                gene_mask &= (adata.var[var_key] > var_filter['gt']).values
            if var_filter.get('lt', None) is not None:
                gene_mask &= (adata.var[var_key] < var_filter['lt']).values
    if obs_filters:
        for obs_key, obs_filter in obs_filters.items():
            if obs_key not in adata.obs.columns:
                continue
            if obs_filter.get('eq', None) is not None:
                cell_mask &= (adata.obs[obs_key] == obs_filter['eq']).values
            if obs_filter.get('in', None) is not None:
                cell_mask &= adata.obs[obs_key].isin(obs_filter['in']).values
            if obs_filter.get('gt', None) is not None:
                cell_mask &= (adata.obs[obs_key] > obs_filter['gt']).values
            if obs_filter.get('lt', None) is not None:
                cell_mask &= (adata.obs[obs_key] < obs_filter['lt']).values
    return cell_mask, gene_mask

def generate_section_h5ad(data_path, save_path, var_filters=None, obs_filters=None):
    if data_path.suffix == '.zarr':
        section_sdata = sd.read_zarr(data_path)
        adata = section_sdata['table'].copy()
        del section_sdata
    else:
        adata = ad.io.read_zarr(data_path)
    
    cell_mask, gene_mask = get_filter_masks(adata, var_filters=var_filters, obs_filters=obs_filters)
    adata = adata[cell_mask, gene_mask]
    
    # Strip the unnecessary items
    adata.uns = {}
    adata.obsm = {}
    adata.var['gene_symbol'] = adata.var_names
    adata.var.index = adata.var['gene_ids']
    adata.write_h5ad(save_path)
    return save_path

def setup_paths_and_names(section_path, save_results_path, mapping_config):
    """Set up paths and naming conventions for a section."""
    section_name = section_path.stem

    # Naming conventions from mapping config
    input_data_folder_name = mapping_config.get('input_data_folder_name', 'input_data')
    mapped_data_folder_name = mapping_config.get('mapped_data_folder_name', 'mapped_data')
    input_h5ad_name = mapping_config.get('input_h5ad_name', 'input_cellxgene.h5ad')
    mapped_data_h5ad_name = mapping_config.get('mapped_data_h5ad_name', 'mapped_cellxgene.h5ad')
    basic_results_name = mapping_config.get('basic_results_name', 'basic_results.csv')
    extended_results_name = mapping_config.get('extended_results_name', 'extended_results.json')

    # Set up paths
    input_data_folder = save_results_path / section_name / input_data_folder_name
    input_data_folder.mkdir(parents=True, exist_ok=True)

    # Path to table AnnData
    table_path = section_path / 'tables' / 'table'
    cellxgene_input_path = input_data_folder / input_h5ad_name

    # Section output folder
    output_data_folder = save_results_path / section_name / mapped_data_folder_name
    output_data_folder.mkdir(parents=True, exist_ok=True)

    # Output paths
    extended_results_path = output_data_folder / extended_results_name
    basic_results_path = output_data_folder / basic_results_name
    mapped_adata_path = output_data_folder / mapped_data_h5ad_name
    section_save_path = save_results_path / f"{section_name}.h5ad"

    return (table_path, input_data_folder, cellxgene_input_path, output_data_folder,
            extended_results_path, basic_results_path,
            mapped_adata_path, section_save_path)

def get_overwrite_flags(mapping_config):
    overwrite_all_steps = mapping_config.get('overwrite_all_steps', False)
    overwrite_input_data = mapping_config.get('overwrite_input_data', False)
    overwrite_running_mapping = mapping_config.get('overwrite_running_mapping', False)
    overwrite_formatting = mapping_config.get('overwrite_formatting', False)
    overwrite_merge = mapping_config.get('overwrite_merge', False)
    overwrite_any = any([overwrite_all_steps, overwrite_input_data, overwrite_running_mapping, overwrite_formatting, overwrite_merge])
    return overwrite_any, overwrite_all_steps, overwrite_input_data, overwrite_running_mapping, overwrite_formatting, overwrite_merge

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

def merge_mapping_to_table(mapped_adata_path, table_path, section_save_path):
    # Read in the mapped AnnData
    mapped_adata = ad.read_h5ad(mapped_adata_path)
    mapped_adata.obs.columns = mapped_adata.obs.columns.str.replace('CDM_', "")

    # Clean up column names
    mapped_adata.obs.rename(
        columns={col: col.replace(" (non-expanded)", "") for col in mapped_adata.obs.columns}, 
        inplace=True
    )
    name_cols = [col for col in mapped_adata.obs.columns if col.endswith('_name')]
    
    # Check first name_cols to see if all NaNs (ie mapping failed)
    if name_cols:
        first_name_col = name_cols[0]
        if mapped_adata.obs[first_name_col].isna().all():
            print(f"All values in {first_name_col} are NaN, indicating mapping may have failed for {section_name}.")
            return False
    else:
        print(f"No name columns found in mapped AnnData indicating mapping may have failed for {section_name}.")
        return False

    # Load original section SpatialData table
    section_sd_table = ad.io.read_zarr(table_path)
    orig_table_shape = section_sd_table.shape
    print(f"Original table: \n\t{section_sd_table.n_obs} cells \n\t{section_sd_table.n_vars} genes")
    print(f"Mapped adata: \n\t{mapped_adata.n_obs} cells \n\t{mapped_adata.n_vars} genes")

    # Update obs
    mapping_obs_cols = [col for col in mapped_adata.obs.columns if col not in section_sd_table.obs.columns]
    print(f"Adding {len(mapping_obs_cols)} columns to obs.")
    section_sd_table.obs = section_sd_table.obs.merge(
        mapped_adata.obs[mapping_obs_cols], 
        left_index=True, 
        right_index=True, 
        how='left', 
        suffixes=('', '_mapped')
    )

    # Update vars
    mapping_var_cols = [col for col in mapped_adata.var.columns if col not in section_sd_table.var.columns]
    if mapping_var_cols:
        print(f"Adding {len(mapping_var_cols)} columns to var.")
    section_sd_table.var = section_sd_table.var.merge(
        mapped_adata.var[mapping_var_cols], 
        left_index=True, 
        right_index=True, 
        how='left', 
        suffixes=('', '_mapped')
    )
    print(f"Original table shape: {orig_table_shape}")
    print(f"Updated table shape: {section_sd_table.shape}")

    # Save merged table as h5ad
    section_sd_table.write_h5ad(section_save_path)
    
    del mapped_adata, section_sd_table
    gc.collect()
    return True 

def map_single_section(section_path, logger, save_results_path, 
                   mapping_config, mapping_params, type_assignment,
                   precomputed_stats_path, mouse_markers_path, gene_mapper_db_path,
                   var_filters, obs_filters):
    """Process a single section."""

    # Set up paths and names
    section_name = section_path.stem
    (table_path, input_data_folder, cellxgene_input_path, output_data_folder,
     extended_results_path, basic_results_path,
     mapped_adata_path, section_save_path) = setup_paths_and_names(
        section_path, save_results_path, mapping_config
    )
    
    # Determine overwrite flags
    (overwrite_any, overwrite_all_steps, overwrite_input_data, 
     overwrite_running_mapping, overwrite_formatting, overwrite_merge) = get_overwrite_flags(mapping_config)

    # Check if section already mapped and if any overwrite flags are set
    if section_save_path.exists() and not overwrite_any:
        logger.info(f"{section_name} already mapped, skipping.")
        return True
    else:
        logger.info(f"Mapping {section_name}...")

    # ----- Generate cellxgene input h5ad -----
    if cellxgene_input_path.exists() and not overwrite_input_data:
        logger.info(f"Cellxgene input h5ad already exists at {cellxgene_input_path}, skipping generation.")
    else:
        cellxgene_input_path = generate_section_h5ad(data_path=table_path, 
                                    save_path=cellxgene_input_path, 
                                    var_filters=var_filters, 
                                    obs_filters=obs_filters)

    # ----- Map types -----
    if extended_results_path.exists() and basic_results_path.exists() and not overwrite_running_mapping:
        logger.info(f"Already ran mapping for section")
        query_path = None
    else:
        logger.info(f"Starting mapping process for {section_name}")
        # Validate input data before running mapper
        if mapping_config.get('validate_input', False):
            query_path = validate_input_adata(cellxgene_input_path, input_data_folder, mouse_markers_path, gene_mapper_db_path)
        else:
            logger.info("Not validating input data before mapping")
            query_path = cellxgene_input_path

        # Set up the mapping runner
        mapper_config = {
            'query_path': query_path,
            'extended_result_path': str(extended_results_path),
            'csv_result_path': str(basic_results_path),
            'flatten': True if mapping_params.get('mapping_type') == 'flat' else False,
            'precomputed_stats': {'path': precomputed_stats_path},
            'query_markers': {'serialized_lookup': mouse_markers_path},
            'type_assignment': type_assignment,
            'gene_mapping': {'db_path': str(gene_mapper_db_path)},
            'nodes_to_drop': mapping_params.get('nodes_to_drop', None),
            'verbose_stdout': False,
            'tmp_dir': '/tmp'
        }
        
        logger.info("Mapping configuration prepared")
        logger.info(f"Query path: {mapper_config['query_path']}")
        logger.info(f"Output paths: \n\tExtended: {extended_results_path}, \n\tBasic: {basic_results_path}")

        # Run the mapper
        logger.info(f"Starting cell type mapping for {section_name}...")
        try:
            runner = FromSpecifiedMarkersRunner(args=[], input_data=mapper_config)
            runner.run()
            logger.info(f"Mapping completed for {section_name}")
        except Exception as e:
            logger.error(f"Mapping failed for {section_name}: {e}")
            return False

    # ----- Format output data -----
    if mapped_adata_path.exists() and not overwrite_formatting:
        logger.info(f"Formatted AnnData already exists at {mapped_adata_path}, skipping generation.")
    else:
        logger.info(f"Generating formatted AnnData: {mapped_adata_path}")
        format_out = format_mapping_outputs(extended_results_path, mapped_adata_path, mapping_params, h5ad_path=query_path)
        if not format_out:
            logger.error(f"Formatting mapping outputs failed for {section_name}")
            return False
    
    # ----- Merge mapping results back into original SpatialData table -----
    if section_save_path.exists() and not overwrite_merge:
        logger.info(f"Formatted AnnData already exists at {mapped_adata_path}, skipping generation.")
    else:
        logger.info(f"Merging mapping results back into original table for {section_name}")
        merged_to_table = merge_mapping_to_table(mapped_adata_path, table_path, section_save_path)
        if not merged_to_table:
            logger.error(f"Merging mapping results failed for {section_name}")
            return False
    return True
