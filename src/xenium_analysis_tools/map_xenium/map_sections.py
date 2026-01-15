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
    
def get_v1_merfish_subclasses(abc_cache, output_path=None, visp_cluster_min_cells=0):
    if output_path and output_path.exists():
        v1_merfish_cells = pd.read_csv(output_path, index_col=0)
    else:
        # Get MERFISH CCF metadata
        print('V1 cell df not found, generating new one...')
        merfish_ccf_metadata = abc_cache.get_metadata_dataframe(
                    directory='MERFISH-C57BL6J-638850-CCF', 
                    file_name='cell_metadata_with_parcellation_annotation'
                ).set_index('cell_label')
        v1_merfish_cells = merfish_ccf_metadata.loc[merfish_ccf_metadata['parcellation_structure']=='VISp']
        if output_path:
            v1_merfish_cells.to_csv(output_path)

    # Load the taxonomy
    taxonomy_df = abc_cache.get_metadata_dataframe(
                        directory='WMB-taxonomy',
                        file_name='cluster_to_cluster_annotation_membership'
                    )
    v1_subclasses = v1_merfish_cells.groupby('subclass').size()
    v1_subclasses = v1_subclasses[v1_subclasses>=visp_cluster_min_cells].index.tolist()
    valid_subclasses = [np.unique(taxonomy_df.loc[taxonomy_df['cluster_annotation_term_name']==subcl,'cluster_annotation_term_label'])[0] for subcl in v1_subclasses]
    subclasses_to_drop = np.setdiff1d(taxonomy_df.loc[taxonomy_df['cluster_annotation_term_set_name']=='subclass','cluster_annotation_term_label'].unique(), valid_subclasses)
    nodes_to_drop = [('subclass', cl) for cl in subclasses_to_drop]
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

def map_single_section(section_path, logger, save_results_path, 
                   mapping_config, mapping_params, type_assignment,
                   precomputed_stats_path, mouse_markers_path, gene_mapper_db_path,
                   var_filters, obs_filters):
    """Process a single section."""
    section_name = section_path.stem
    
    # Set up paths
    input_data_folder = save_results_path / section_name / 'input_data'
    input_data_folder.mkdir(parents=True, exist_ok=True)

    # Path to table AnnData
    table_path = section_path / 'tables' / 'table'
    cellxgene_input_path = input_data_folder / 'input_cellxgene.h5ad'

    # Section output folder
    output_data_folder = save_results_path / section_name / 'mapped_data'
    output_data_folder.mkdir(parents=True, exist_ok=True)

    # Output paths
    extended_results_path = output_data_folder / 'extended_results.json'
    basic_results_path = output_data_folder / 'basic_results.csv'
    mapped_adata_path = output_data_folder / 'mapped_cellxgene.h5ad'

    # Full updated Table AnnData path (will be all cells in SpatialData object, not just mapped)
    section_save_path = save_results_path / f"{section_name}.h5ad"

    # Check if section already mapped and if any overwrite flags are set
    if section_save_path.exists() and not np.any([mapping_params.get('overwrite_all_steps', False),
                                                    mapping_params.get('overwrite_input_data', False),
                                                    mapping_params.get('overwrite_running_mapping', False),
                                                    mapping_params.get('overwrite_final_output', False)]):
        logger.info(f"{section_name} already mapped, skipping.")
        return True
    else:
        logger.info(f"Mapping {section_name}...")

    # ----- Generate cellxgene input h5ad -----
    if cellxgene_input_path.exists() and not mapping_params.get('overwrite_input_data', False):
        logger.info(f"Cellxgene input h5ad already exists at {cellxgene_input_path}, skipping generation.")
    else:
        cellxgene_input_path = generate_section_h5ad(data_path=table_path, save_path=cellxgene_input_path, var_filters=var_filters, obs_filters=obs_filters)

    # ----- Map types -----
    if extended_results_path.exists() and basic_results_path.exists() and not mapping_params.get('overwrite_running_mapping', False):
        logger.info(f"Already ran mapping for section, skipping.")
    else:
        logger.info(f"Starting mapping process for {section_name}")
        
        path_h5ad = str(cellxgene_input_path)
        # Safer way to get log path if needed
        log_path = None
        for handler in logger.handlers:
            if hasattr(handler, 'baseFilename'):
                log_path = str(handler.baseFilename)
                break
        
        validate_config = {
            'h5ad_path': str(cellxgene_input_path),
            'round_to_int': False,
            'layer': 'X',
            'output_dir': str(input_data_folder),
            'output_json': tempfile.mkstemp(suffix='.json')[1],
            'gene_mapping': {'db_path': gene_mapper_db_path}
        }
        
        # Only add log_path if we found one
        if log_path:
            validate_config['log_path'] = log_path

        # Validate input data before running mapper
        if mapping_config.get('validate_input', False):
            logger.info("Starting validation of input h5ad...")
            try:
                validation_runner = ValidateH5adRunner(args=[], input_data=validate_config)
                validation_runner.run()
                logger.info("Validation completed successfully")
                
                validated_path = json.load(open(validate_config['output_json'], 'rb'))['valid_h5ad_path']
                markers_json = json.load(open(mouse_markers_path, 'rb'))
                markers = []
                for key in list(markers_json.keys())[:-2]:
                    markers += markers_json[key]
                use_validated_h5ad = any('ENSMUSG' in marker for marker in markers)
                if use_validated_h5ad:
                    logger.info(f"Using validated path: {validated_path}")
                    path_h5ad = validated_path
                else:
                    logger.info(f'Using original path: {path_h5ad}')
                    
            except Exception as e:
                logger.error(f"Validation failed: {e}")
                raise e
        else:
            logger.info("Skipping input validation")

        # Set up the mapping runner
        mapper_config = {
            'query_path': path_h5ad,
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
        logger.info(f"Output paths - Extended: {extended_results_path}, Basic: {basic_results_path}")

        # Run the mapper
        logger.info(f"Starting cell type mapping for {section_name}...")
        try:
            runner = FromSpecifiedMarkersRunner(args=[], input_data=mapper_config)
            runner.run()
            logger.info(f"Mapping completed successfully for {section_name}")
        except Exception as e:
            logger.error(f"Mapping failed for {section_name}: {e}")
            raise e

    # ----- Format output data -----
    logger.info(f"Starting output formatting for {section_name}")
    if mapped_adata_path.exists() and not mapping_params.get('overwrite_final_output', False):
        logger.info(f"AnnData for mapping results already exists at {mapped_adata_path}, skipping generation.")
    else:
        logger.info(f"Generating mapped h5ad: {mapped_adata_path}")
        try:
            # Generate h5ad version of mapping outputs
            ad_config = {
                    'result_path': str(extended_results_path), 
                    'h5ad_path': str(path_h5ad), 
                    'new_h5ad_path': str(mapped_adata_path),
                    'clobber': bool(mapping_params['clobber'])
                }
            TranscribeToObsRunner(args=[], input_data=ad_config).run()
            logger.info(f"Output formatting completed for {section_name}")
        except Exception as e:
            logger.error(f"Output formatting failed for {section_name}: {e}")
            raise e

    # ----- Final mapping output formatting/merging -----
    if not mapped_adata_path.exists():
        raise FileNotFoundError(f"Mapped AnnData file not found at {mapped_adata_path} - mapping may have failed.")
    mapped_adata = ad.read_h5ad(mapped_adata_path)

    # Clean up column names
    mapped_adata.obs.columns = mapped_adata.obs.columns.str.replace('CDM_', "")
    mapped_adata.obs.rename(
        columns={col: col.replace(" (non-expanded)", "") for col in mapped_adata.obs.columns}, 
        inplace=True
    )
    name_cols = [col for col in mapped_adata.obs.columns if col.endswith('_name')]
    numeric_cols = [col for col in mapped_adata.obs.columns if col.endswith('_probability') or col.endswith('_correlation')]

    # Check first name_cols to see if all NaNs (ie mapping failed)
    if name_cols:
        first_name_col = name_cols[0]
        if mapped_adata.obs[first_name_col].isna().all():
            logger.warning(f"All values in {first_name_col} are NaN, indicating mapping may have failed for {section_name}.")
            print(f"✗ {section_name} failed: All mapped names are NaN")
            return False
    else:
        logger.warning(f"No name columns found in mapped AnnData indicating mapping may have failed for {section_name}.")
        print(f"✗ {section_name} failed: No name columns found")
        return False

    # Section mapping summary
    if name_cols:
        name_summary = pd.DataFrame({
            'count': mapped_adata.obs[name_cols].count(),
            'nunique': mapped_adata.obs[name_cols].nunique(),
            'most_frequent': mapped_adata.obs[name_cols].mode().iloc[0] if len(mapped_adata.obs) > 0 else None,
            'frequency': [mapped_adata.obs[col].value_counts().iloc[0] if len(mapped_adata.obs[col].value_counts()) > 0 else 0 for col in name_cols]
        })
        print("Name Columns Summary:")
        print(name_summary)
        print("\n")
        logger.info(f"Name Columns Summary:\n{name_summary}\n")
    if numeric_cols:
        numeric_summary = pd.DataFrame({
            'count': mapped_adata.obs[numeric_cols].count(),
            'mean': mapped_adata.obs[numeric_cols].mean(),
            'min': mapped_adata.obs[numeric_cols].min(),
            'max': mapped_adata.obs[numeric_cols].max()
        })
        print("Numeric Columns Summary:")
        print(numeric_summary)
        logger.info(f"Numeric Columns Summary:\n{numeric_summary}\n")


    # ----- Merge mapping results back into original SpatialData table -----
    # Load original section SpatialData table
    section_sd_table = ad.io.read_zarr(table_path)
    orig_table_shape = section_sd_table.shape
    logger.info(f"Original table: \n\t{section_sd_table.n_obs} cells \n\t{section_sd_table.n_vars} genes")
    logger.info(f"Mapped adata: \n\t{mapped_adata.n_obs} cells \n\t{mapped_adata.n_vars} genes")

    # Update obs
    mapping_obs_cols = [col for col in mapped_adata.obs.columns if col not in section_sd_table.obs.columns]
    logger.info(f"Adding {len(mapping_obs_cols)} columns to obs.")
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
        logger.info(f"Adding {len(mapping_var_cols)} columns to var.")
    section_sd_table.var = section_sd_table.var.merge(
        mapped_adata.var[mapping_var_cols], 
        left_index=True, 
        right_index=True, 
        how='left', 
        suffixes=('', '_mapped')
    )
    logger.info(f"Original table shape: {orig_table_shape}")
    logger.info(f"Updated table shape: {section_sd_table.shape}")

    section_sd_table.write_h5ad(section_save_path)
    
    del mapped_adata, section_sd_table
    gc.collect()
    
    print(f"{section_name} completed successfully")
    return True 