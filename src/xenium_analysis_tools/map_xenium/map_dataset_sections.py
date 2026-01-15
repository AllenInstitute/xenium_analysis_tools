from pathlib import Path
import tqdm
import os

# MapMyCells mapping pipeline
from cell_type_mapper.test_utils.cache_wrapper import AbcCacheWrapper

# Local imports
from xenium_analysis_tools.map_xenium.map_sections import (
    get_v1_merfish_subclasses, 
    get_abc_paths, 
    get_sections_to_process,
    map_single_section,
)
from xenium_analysis_tools.utils.io_utils import (
    load_config, 
    setup_logging,
)

# Environment setup (limit threads for numpy operations)
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

def map_sections(dataset_name: str, config_path: str, select_sections: list[int]|None = None, sections_parent_folder='data'):
    # Early validation
    config = load_config(config_path)
    processing_config = config['processing_control']
    paths = config['paths']
    sections_path = Path(paths[f'{sections_parent_folder}_root']) / f"{dataset_name}{processing_config['save_processed_dataset_suffix']}"

    try:
        section_zarrs = get_sections_to_process(sections_path, select_sections)
        print(f"Processing {len(section_zarrs)} sections: {[p.stem.split('_')[-1] for p in section_zarrs]}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # ---- Set up ----
    save_results_path = Path(paths[f'{processing_config['save_mapped_data_parent_folder']}_root']) / f"{dataset_name}{processing_config['save_mapped_dataset_suffix']}"
    save_results_path.mkdir(parents=True, exist_ok=True)
    logger, log_file_path = setup_logging(save_results_path)
    logger.info(f"Running: {dataset_name}")
    
    # Paths
    abc_atlas_path = Path(paths['abc_path'])
    abc_cache = AbcCacheWrapper.from_local_cache(abc_atlas_path)
    precomputed_stats_path, mouse_markers_path, gene_mapper_db_path = get_abc_paths(abc_cache)

    # Get configurations
    mapping_config = config['mapping_config']
    mapping_params = mapping_config['mapping_params']

    # Get section zarrs
    section_zarrs = get_sections_to_process(sections_path, select_sections)

    logger.info(f"Found {len(section_zarrs)} section zarrs to map.")

    # Filters
    var_filters = mapping_config.get('var_filters', None)
    obs_filters = mapping_config.get('obs_filters', None)

    # Determine nodes to drop based on V1 subclass cells only option
    if mapping_config.get('v1_subclass_cells_only',False):
        v1_cells_path = Path(paths['scratch_root']) / mapping_config.get('v1_merfish_cells_path', None)
        v1_min_cells = mapping_config.get('v1_visp_cluster_min_cells', 0)
        nodes_to_drop = get_v1_merfish_subclasses(abc_cache, output_path=v1_cells_path, visp_cluster_min_cells=v1_min_cells)
        logger.info(f"Number of nodes to drop: {len(nodes_to_drop) if nodes_to_drop else 0}")
    else:
        nodes_to_drop = None
    mapping_params['nodes_to_drop'] = nodes_to_drop

    # Number of workers
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
        'n_processors': int(mapping_params.get('num_workers', 4)),                  
        'rng_seed': int(mapping_params.get('rng_seed', 42))                        
    }

    # ----- Process sections -----
    successful_sections = []
    failed_sections = []
    
    logger.info(f"\n=== Starting processing of {len(section_zarrs)} sections ===")
    
    # Create progress bar
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
                    save_results_path=save_results_path,
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
                    failed_sections.append((section_name, "Mapping validation failed"))
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