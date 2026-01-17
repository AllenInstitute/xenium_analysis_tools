from pathlib import Path
import tqdm
import os

# MapMyCells mapping pipeline
from cell_type_mapper.test_utils.cache_wrapper import AbcCacheWrapper

# Local imports
from xenium_analysis_tools.map_xenium.map_sections import ( 
    get_abc_paths, 
    get_sections_to_process,
    get_v1_merfish_cells,
    get_nodes_to_drop,
    map_single_section,
)
from xenium_analysis_tools.utils.io_utils import (
    load_config, 
    setup_logging,
    get_partial_dataset,
    is_complete_mapping_results,
)

# Environment setup (limit threads for numpy operations)
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

def map_sections(dataset_name: str, config_path: str=None, select_sections: list[int]|None = None, sections_parent_folder='data'):
    # ---- Set up ----
    config = load_config(config_path)

    # Paths/directories
    paths = config['paths']
    processing_config = config['processing_control']
    mapping_config = config['mapping_config']
    sections_sd_path = Path(paths[f'{sections_parent_folder}_root']) / f"{dataset_name}{processing_config['save_processed_dataset_suffix']}"
    save_mapped_sections_parent_folder = processing_config['save_mapped_data_parent_folder']
    save_mapped_sections_path = Path(paths[f'{save_mapped_sections_parent_folder}_root']) / f"{dataset_name}{processing_config['save_mapped_dataset_suffix']}"
    save_mapped_sections_path.mkdir(parents=True, exist_ok=True)

    # Logger
    logger, log_file_path = setup_logging(save_mapped_sections_path)

    # Print out where sections are being saved
    logger.info(f"Dataset Name: {dataset_name}")
    logger.info(f"Configuration loaded from {config_path}")
    logger.info(f"Sections are being loaded from: {sections_sd_path}")
    logger.info(f"Mapped sections will be saved to: {save_mapped_sections_path}")

    # If specified, copy sections from data folder instead of re-generating
    if processing_config['check_data_folder_mapped']:
        logger.info("Checking and copying sections from data folder if exists...")
        data_folder_sections_path = Path(paths['data_root']) / f'{dataset_name}{processing_config["save_mapped_dataset_suffix"]}'
        get_partial_dataset(source_path=data_folder_sections_path, 
                    dest_path=save_mapped_sections_path, 
                    pattern='section_*', 
                    subset_ids=select_sections, 
                    is_complete_func=is_complete_mapping_results, 
                    func_args={'input_folder_name': mapping_config.get('input_data_folder_name', 'input_data'),
                                    'mapped_folder_name': mapping_config.get('mapped_data_folder_name', 'mapped_data'),
                                    'input_data_files': [mapping_config.get('input_h5ad_name', 'input_cellxgene.h5ad')],
                                    'mapped_data_files': [mapping_config.get('basic_results_name', 'basic_results.csv'),
                                                        mapping_config.get('extended_results_name', 'extended_results.json'),
                                                        mapping_config.get('mapped_data_h5ad_name', 'mapped_cellxgene.h5ad')]
                                }
                    )

    # Get sections to map
    try:
        section_zarrs = get_sections_to_process(sections_sd_path, select_sections)
        print(f"\nProcessing {len(section_zarrs)} sections: {[p.stem.split('_')[-1] for p in section_zarrs]}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # ABC Atlas paths
    abc_atlas_path = Path(paths['abc_path'])
    abc_cache = AbcCacheWrapper.from_local_cache(abc_atlas_path)
    precomputed_stats_path, mouse_markers_path, gene_mapper_db_path = get_abc_paths(abc_cache)

    # ----- Filtering -----
    # Cell and gene filters
    var_filters = mapping_config.get('var_filters', None)
    obs_filters = mapping_config.get('obs_filters', None)
    if obs_filters:
        logger.info("Applying filters to cells:")
        for col,filt in obs_filters.items():
            logger.info(f"{col}: {filt}")
    if var_filters:
        logger.info("Applying filters to genes:")
        for col,filt in var_filters.items():
            logger.info(f"{col}: {filt}")

    # Taxonomy filters - nodes to drop for mapping
    nodes_to_drop=[]
    # If specified any specific nodes to drop in config
    drop_nodes_dict = mapping_config.get('drop_nodes_dict', None)
    if drop_nodes_dict:
        for h_level in drop_nodes_dict:
            nodes_to_drop.extend([(h_level, cl) for cl in drop_nodes_dict[h_level]])
        logger.info(f"Dropping {len(nodes_to_drop)} nodes based on drop_nodes_dict.")
    # Filter to only V1 cells
    filter_v1_types_config = mapping_config.get('filter_mapping_v1_types', None)
    if filter_v1_types_config and filter_v1_types_config.get('enabled', False):
        h_level = filter_v1_types_config.get('h_level', 'subclass')
        min_cells = filter_v1_types_config.get('min_cells', 0)
        v1_types_df_name = filter_v1_types_config.get('saved_df_name', 'v1_merfish_cells.csv')
        if v1_types_df_name:
            v1_types_path = Path(paths['data_root']) / v1_types_df_name
        else:
            v1_types_path = None
        v1_merfish_cells = get_v1_merfish_cells(abc_cache, df_path=v1_types_path)
        v1_nodes_to_drop = get_nodes_to_drop(v1_merfish_cells, abc_cache, h_level=h_level, min_cells=min_cells)
        logger.info(f"Dropping {len(v1_nodes_to_drop)} {h_level} nodes not present in V1 MERFISH data with at least {min_cells} cells.")
        nodes_to_drop.extend(v1_nodes_to_drop)

    # ----- Mapper parameters -----
    mapping_params = mapping_config.get('mapping_params', {})
    mapping_params['nodes_to_drop'] = nodes_to_drop

    # n_processors for mapper
    num_workers = mapping_params.get('num_workers', None)
    if num_workers == 'max':
        num_workers = os.cpu_count()
    elif num_workers is None:
        num_workers = 4
    if num_workers > os.cpu_count():
        num_workers = os.cpu_count()

    # Type assignment parameters for mapper
    type_assignment = {
        'normalization': mapping_params.get('normalization', 'raw'),
        'bootstrap_iteration': int(mapping_params.get('bootstrap_iteration', 100)),  
        'bootstrap_factor': float(mapping_params.get('bootstrap_factor', 0.5)),      
        'n_runners_up': int(mapping_params.get('n_runner_ups', 0)),                
        'chunk_size': int(mapping_params.get('chunk_size', 5000)),                 
        'n_processors': num_workers,                  
        'rng_seed': int(mapping_params.get('rng_seed', 42))                        
    }

    # ----- Process sections -----
    successful_sections = []
    failed_sections = []
    logger.info(f"\n=== Starting processing of {len(section_zarrs)} sections ===")
    with tqdm.tqdm(total=len(section_zarrs), desc="Processing sections", 
                    unit="section", position=0, leave=True) as pbar:
            
        for idx, section_path in enumerate(section_zarrs, 1):
            section_name = section_path.stem
            
            # Update progress bar description with current section
            pbar.set_description(f"Processing {section_name}")
            logger.info(f"\n[{idx}/{len(section_zarrs)}] Processing {section_name}...")

            try:
                success = map_single_section(
                    section_path=section_path,
                    logger=logger,
                    save_results_path=save_mapped_sections_path,
                    mapping_config=mapping_config,
                    mapping_params=mapping_params,
                    type_assignment=type_assignment,
                    precomputed_stats_path=precomputed_stats_path,
                    mouse_markers_path=mouse_markers_path,
                    gene_mapper_db_path=gene_mapper_db_path,
                    var_filters=var_filters,
                    obs_filters=obs_filters
                )
                if success:
                    successful_sections.append(section_name)
                    pbar.set_postfix({"✓": len(successful_sections), "✗": len(failed_sections)})
                else:
                    failed_sections.append((section_name, "Mapping section failed"))
                    pbar.set_postfix({"✓": len(successful_sections), "✗": len(failed_sections)})
                    
            except Exception as e:
                failed_sections.append((section_name, str(e)))
                logger.error(f"Failed to process {section_name}: {e}")
                pbar.set_postfix({"✓": len(successful_sections), "✗": len(failed_sections)})
                continue
            finally:
                # Always update progress bar
                pbar.update(1)
        
        # Final update
        pbar.set_description("Processing complete")

    # Final summary
    print(f"\n=== FINAL RESULTS ===")
    print(f"Successful: {len(successful_sections)}/{len(section_zarrs)}")
    if successful_sections:
        print(f"Successfully processed: {[s for s in successful_sections]}")
    if failed_sections:
        print("Failed sections:")
        for name, error in failed_sections:
            print(f"  - {name}: {error}")

    logger.info(f"Pipeline completed. Success: {len(successful_sections)}, Failed: {len(failed_sections)}")
