from pathlib import Path
import tqdm
import os
import gc 
import pandas as pd
import numpy as np

from xenium_analysis_tools.utils.io_utils import (
    atomic_write_sdata, 
    is_complete, 
    is_complete_store, 
    load_config, 
    setup_logging,
    get_sections_df,
    get_partial_dataset
)
from xenium_analysis_tools.process_xenium.process_spatialdata import read_xenium_slide

def find_xenium_bundle(bundle_name, data_folder='/root/capsule/data'):
    data_folder = Path(data_folder)
    search_paths = [
        data_folder / 'xenium_data',
        data_folder / 'Xenium_output_pilot'
    ]
    search_paths = [path for path in search_paths if path.exists()]
    all_dirs = np.concatenate([list(folder.iterdir()) for folder in search_paths])
    output_folders = np.concatenate([list(folder.glob('output-*')) for folder in search_paths])
    subfolders = np.setdiff1d(all_dirs, output_folders)
    path_to_bundle = None
    found_dirs = [dir for dir in output_folders if dir.name == bundle_name]
    if found_dirs:
        path_to_bundle = found_dirs[0]
    else:
        for sub in subfolders:
            found_dirs = [dir for dir in list(sub.iterdir()) if dir.name == bundle_name]
            if found_dirs:
                path_to_bundle = found_dirs[0]
                break
    return path_to_bundle
    
def generate_slides(dataset_name: str, config_path: str=None, select_sections: list[int]|None = None):
    """
    Generate slide-level SpatialData objects from raw Xenium data bundles.
    """
    # ---- Set up ----
    config = load_config(config_path)

    # Paths/directories
    paths = config['paths']
    processing_config = config['processing_control']
    raw_data_folder = Path(paths['data_root']) / dataset_name
    save_sections_parent_folder = processing_config['save_initial_data_parent_folder']
    save_sections_path = Path(paths[f'{save_sections_parent_folder}_root']) / f"{dataset_name}{processing_config['save_initial_dataset_suffix']}"
    save_sections_path.mkdir(parents=True, exist_ok=True)

    # Logger
    logger, log_file_path = setup_logging(save_sections_path)

    # Print out where slides are being saved
    logger.info(f"Dataset Name: {dataset_name}")
    logger.info(f"Configuration loaded from {config_path}")
    logger.info(f"Raw data folder: {raw_data_folder}")
    logger.info(f"Slides will be saved to: {save_sections_path}")

    # If specified, copy slides from data folder instead of re-generating
    if processing_config['check_data_folder_slides']:
        logger.info("Checking and copying slides from data folder if exist...")
        data_folder_slides_path = Path(paths['data_root']) / f'{dataset_name}{processing_config["save_initial_dataset_suffix"]}'
        get_partial_dataset(data_folder_slides_path, save_sections_path, pattern='slide_*', subset_ids=select_sections)

    # Get the slides information
    sections_df = get_sections_df(raw_data_folder)

    # Limit sections, if specified
    if select_sections is not None:
        logger.info(f"Limiting processing to sections: {select_sections}")
        sections_df = sections_df[sections_df['section'].isin(select_sections)]
    
    # Set up processing loop
    logger.info(f"Processing {len(sections_df)} sections from {sections_df['slide_id'].nunique()} slide(s)")
    unique_slides = sections_df.groupby('slide_id')

    # ---- Run processing ----
    for slide_id in tqdm.tqdm(unique_slides.groups.keys(), 
                                desc="Processing slides", 
                                unit="slide",
                                total=len(unique_slides.groups.keys())):
        try:
            # Get slide information
            group = unique_slides.get_group(slide_id)
            slide_row = group.iloc[0]
            raw_slide_path = raw_data_folder / slide_row['dir']
            save_slide_path = save_sections_path / f"{processing_config['save_initial_dataset_prefix']}{slide_id}.zarr"
            logger.info(f"Processing slide {slide_id}...")

            # Check if already generated
            if is_complete(save_slide_path, check_store=True):
                logger.info(f"Slide {slide_id} already processed. Skipping.")
                continue
            logger.info(f"Generating SpatialData object for slide {slide_id}...")

            # Make sure experiment file exists - if not, try to find alternative location
            if not (raw_slide_path / 'experiment.xenium').exists():
                logger.info(f"Experiment file not found for slide {slide_id} at {raw_slide_path / 'experiment.xenium'}")
                logger.info(f"Looking for alternative experiment file...")
                path_to_bundle = find_xenium_bundle(slide_row['dir'], data_folder=paths['data_root'])
                if path_to_bundle is not None:
                    logger.info(f"Found alternative experiment file in {path_to_bundle.parent}")
                    raw_slide_path = path_to_bundle
                else:
                    logger.error(f"Could not find experiment file for slide {slide_id}. Skipping.")
                    continue

            # Read Xenium slide and save
            logger.info(f"Reading Xenium bundle: {raw_slide_path}")
            sdata_reader_params = config.get('sdata_reader_params', {})
            if sdata_reader_params.get('n_jobs') == "max": sdata_reader_params['n_jobs'] = os.cpu_count()
            logger.info(f"Using sdata_reader_params: {sdata_reader_params}")
            sdata = read_xenium_slide(raw_slide_path, sdata_reader_params, dask_config=config.get('dask_config', None))
            atomic_write_sdata(sdata, save_slide_path)
            del sdata; gc.collect()
            logger.info(f"Slide {slide_id} processing complete.")
        except Exception as e:
            logger.error(f"Error processing slide {slide_id}: {e}", exc_info=True)

    # ---- Validate processed slides ----
    validated_sections = pd.Series(dtype=bool, index=unique_slides.groups.keys())
    for slide_id in tqdm.tqdm(unique_slides.groups.keys(),
                                desc="Processing slides", 
                                unit="slide",
                                total=len(unique_slides.groups.keys())):
        group = unique_slides.get_group(slide_id)
        slide_row = group.iloc[0]
        raw_slide_path = raw_data_folder / slide_row['dir']
        save_slide_path = save_sections_path / f"slide_{slide_id}.zarr"
        if is_complete_store(save_slide_path):
            logger.info(f"Slide {slide_id} verified as complete.")
            validated_sections[slide_id] = True
        else:
            logger.warning(f"Slide {slide_id} is incomplete or corrupted.")
            validated_sections[slide_id] = False

    # If ALL slides validated, then can remove _SUCCESS flags so not in outputted data
    if np.all(validated_sections) and len(validated_sections)==sections_df['slide_id'].nunique():
        logger.info("All slides have been successfully validated - removing flags.")
        for zarr_dir in save_sections_path.glob("*.zarr"):
            (zarr_dir / "_SUCCESS").unlink(missing_ok=True)


