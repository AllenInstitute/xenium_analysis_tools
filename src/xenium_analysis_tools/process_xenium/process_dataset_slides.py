from xenium_analysis_tools.utils.io_utils import (
    atomic_write_sdata, 
    is_complete, 
    is_complete_store, 
    load_config, 
    setup_logging,
    get_sections_df
)
from xenium_analysis_tools.process_xenium.process_spatialdata import (
    process_metadata, 
    get_dapi_zstack
)
from xenium_analysis_tools.process_xenium.divide_sections import (
    crop_dapi_image,
    crop_filter_sdata,
    reset_section_coordinates
)
from xenium_analysis_tools.process_xenium.validate_sections import (
    plot_section_bboxes
)
from pathlib import Path
import tqdm
import pandas as pd
import numpy as np
import spatialdata as sd

def process_slides(dataset_name: str, config_path: str=None, select_sections: list[int]|None = None, slides_parent_folder='data'):
    """
    Process slide-level SpatialData objects into section-level SpatialData objects.
    """
    
    # ---- Set up ----
    config = load_config(config_path)
    paths = config['paths']
    processing_config = config['processing_control']
    raw_data_folder = Path(paths['data_root']) / dataset_name
    slide_sd_path = Path(paths[f'{slides_parent_folder}_root']) / f"{dataset_name}{processing_config['save_initial_dataset_suffix']}"
    save_sections_parent_folder = processing_config['save_processed_data_parent_folder']
    save_sections_path = Path(paths[f'{save_sections_parent_folder}_root']) / f"{dataset_name}{processing_config['save_processed_dataset_suffix']}"
    save_sections_path.mkdir(parents=True, exist_ok=True)
    logger, log_file_path = setup_logging(save_sections_path)
    logger.info(f"Dataset Name: {dataset_name}")
    logger.info(f"Configuration loaded from {config_path}")
    logger.info(f"Raw data folder: {raw_data_folder}")
    logger.info(f"Slides are being loaded from: {slide_sd_path}")
    logger.info(f"Processed sections will be saved to: {save_sections_path}")
    sections_df = get_sections_df(raw_data_folder)
    # Limit sections, if specified
    if select_sections is not None:
        logger.info(f"Limiting processing to sections: {select_sections}")
        sections_df = sections_df[sections_df['section'].isin(select_sections)]
    logger.info(f"Total slides found: {len(sections_df)}")
    unique_slides = sections_df.groupby('slide_id')

    # ---- (Optional) Move data from a data asset to results folder ----
    # Load partially processed sections from data asset and save to results to further process
    if config['processing_control'].get('load_processed_from_asset', False):
        logger.info(f"Looking for processed sections in a data asset...")
        dataset_asset_folder = Path(config['paths']['data_root']) / f"{dataset_name}{config['processing_control']['save_processed_dataset_suffix']}"
        if dataset_asset_folder.exists():
            logger.info(f"Loading processed sections from data asset at {dataset_asset_folder}")
            sections_in_folder = list(dataset_asset_folder.glob('section_*.zarr'))
            for section_zarr_path in tqdm.tqdm(sections_in_folder, desc="Moving processed sections from asset"):
                section_save_path = save_sections_path / section_zarr_path.name
                if section_save_path.exists() and is_complete_store(section_save_path):
                    logger.info(f"Section {section_zarr_path.name} already exists in results folder. Skipping.")
                    continue
                logger.info(f"Loading section {section_zarr_path.stem} from asset and saving to results folder...")
                try:
                    sdata = sd.read_zarr(section_zarr_path)
                    atomic_write_sdata(sdata, section_save_path, overwrite=True)
                    del sdata
                except Exception as e:
                    logger.error(f"Error loading section {section_zarr_path.name} from asset: {e}")
                    continue
        else:
            logger.info(f"No processed data asset found at {dataset_asset_folder}. Continuing without loading from asset.")

    # ---- Run processing ----
    logger.info(f"Starting processing for dataset: {dataset_name}")
    unique_slides = sections_df.groupby('slide_id')
    for slide_id in tqdm.tqdm(unique_slides.groups.keys(), desc="Processing slides", unit="slide", total=len(unique_slides.groups.keys())):
        group = unique_slides.get_group(slide_id)
        slide_row = group.iloc[0]
        raw_slide_path = raw_data_folder / slide_row['dir']
        slide_sdata_path = slide_sd_path / f"{processing_config['save_initial_dataset_prefix']}{slide_id}.zarr"
        if not slide_sdata_path.exists():
            logger.warning(f"Slide data not found for slide {slide_id} at {slide_sdata_path}! Skipping.")
            continue
        slide_sections = slide_row['slide_sections']
        process_sections = []
        for section in slide_sections:
            section_zarr = f"{processing_config['save_processed_dataset_prefix']}{section}.zarr"
            section_save_path = save_sections_path / section_zarr
            if is_complete_store(section_save_path):
                logger.info(f"{section_zarr} data already complete at {section_save_path}. Skipping.")
                continue
            process_sections.append(section)
        if not process_sections:
            logger.info(f"All sections for slide {slide_id} are already processed. Skipping slide.")
            continue
        logger.info(f"Processing slide {slide_id} sections: {[str(s) for s in process_sections]}")

        # --- Process slide SpatialData ---
        # Load slide SpatialData
        slide_sdata = sd.read_zarr(slide_sdata_path)

        # Get additional metadata from raw data xenium bundle
        logger.info(f"Processing metadata for slide {slide_id}...")
        slide_sdata = process_metadata(slide_sdata, raw_slide_path, slide_sections)
        ome_tif_path = raw_slide_path / slide_sdata['table'].uns['section_metadata']['images']['morphology_filepath']
        logger.info(f"Loading DAPI z-stack from {ome_tif_path.name}...")
        dapi_image = get_dapi_zstack(ome_tif_path=ome_tif_path, sdata=slide_sdata)

        # --- Finish processing if single section slide ---
        # Can just add DAPI z-stack to slide SpatialData
        if len(slide_sections) == 1:
            logger.info(f"Single section slide {slide_id}. Adding DAPI z-stack to slide SpatialData.")
            slide_sdata['dapi_zstack'] = dapi_image
            section_zarr = f"{processing_config['save_processed_dataset_prefix']}{section}.zarr"
            section_save_path = save_sections_path / section_zarr
            logger.info(f"Saving processed slide SpatialData for section {slide_sections[0]} to {section_save_path}...")
            atomic_write_sdata(slide_sdata, section_save_path)
            continue

        # --- Divide slide into sections ---
        for section in process_sections:
            logger.info(f"Separating section {section}...")
            section_bbox = slide_sdata['table'].uns['sections_bboxes'][str(section)]
            logger.info(f"Cropping DAPI z-stack...")
            cropped_dapi_zstack = crop_dapi_image(dapi_image, section_bbox)
            logger.info(f"Cropping SpatialData...")
            cropped_sdata = crop_filter_sdata(slide_sdata, section_bbox)

            logger.info(f"Resetting coordinates for section elements...")
            section_sdata = reset_section_coordinates(
                cropped_sdata=cropped_sdata,
                cropped_dapi_zstack=cropped_dapi_zstack,
                full_dapi_zstack=dapi_image,
                slide_sdata=slide_sdata,
                section=section
            )
            section_save_path = save_sections_path / f"{processing_config['save_processed_dataset_prefix']}{section}.zarr"
            logger.info(f"Saving processed section SpatialData to {section_save_path}...")
            atomic_write_sdata(section_sdata, section_save_path)
            
            # Clean up variables between sections
            clear_vars = ['cropped_sdata', 'cropped_dapi_zstack']
            for var in clear_vars:
                if var in locals():
                    del locals()[var]

        # Clean up variables between slides
        clear_vars = ['section_sdata', 'dapi_image', 'cropped_sdata', 'cropped_dapi_zstack']
        for var in clear_vars:
            if var in locals():
                del locals()[var]

    # ---- Plot derived bounding boxes used for section splits ----
    # Plot section bounding boxes on slide to verify separation based on FOVs
    validation_params = config.get('validation_params', {})
    if validation_params.get('plot_bboxes', False):
        import anndata as ad
        # Make subfolder for plots
        save_plots_folder = save_sections_path / 'divided_sections_plots'
        save_plots_folder.mkdir(parents=True, exist_ok=True)
        for slide_id in tqdm.tqdm(unique_slides.groups.keys(), desc="Processing slides", unit="slide", total=len(unique_slides.groups.keys())):
            group = unique_slides.get_group(slide_id)
            slide_row = group.iloc[0]
            slide_sections = slide_row['slide_sections']

            if len(slide_sections)==1:
                logger.info(f"Single section slide {slide_id}. Skipping bbox plotting.")
                continue
            
            # See if plot has already been created
            slide_sdata_path = slide_sd_path / f"{processing_config['save_initial_dataset_prefix']}{slide_id}.zarr"
            save_plot_path = save_plots_folder / f"{slide_sdata_path.stem}.png"
            if save_plot_path.exists():
                logger.info(f"Plot already exists at {save_plot_path}. Skipping.")
                continue
            
            # Get the SpatialData for the combined slide
            if not slide_sdata_path.exists():
                logger.warning(f"Slide data not found for slide {slide_id} at {slide_sdata_path}! Skipping.")
                continue

            # Load slide data
            logger.info(f"Plotting section bboxes for slide {slide_id}...")
            try:
                slide_sdata = sd.read_zarr(slide_sdata_path)
            except Exception as e:
                logger.error(f"Error loading slide SpatialData for slide {slide_id} from {slide_sdata_path}: {e}. Skipping")
                continue

            # Get bboxes for all sections from first section
            section_sdata_path = save_sections_path / f"{processing_config['save_processed_dataset_prefix']}{slide_sections[0]}.zarr"
            adata = ad.io.read_zarr(section_sdata_path / 'tables' / 'table')
            sections_bboxes = adata.uns['sections_bboxes']
            fov_df = adata.uns['fov_metadata']

            # Generate the plot, save, close
            plot_section_bboxes(slide_sdata, sections_bboxes, fov_df=fov_df, show_fovs=True, save_path=save_plot_path)

            # Clean up
            if 'slide_sdata' in locals():
                del slide_sdata
            if 'adata' in locals():
                del adata

    # ---- Validating processed sections ----
    logger.info("Validating processed sections...")
    all_sections = np.unique(sections_df['section'].values)
    validated_sections = pd.Series(dtype=bool, index=all_sections)
    for section in tqdm.tqdm(all_sections,
                                unit="section",
                                total=len(all_sections)):
        section_zarr = f"{processing_config['save_processed_dataset_prefix']}{section}.zarr"
        section_save_path = save_sections_path / section_zarr
        if not is_complete_store(section_save_path):
            logger.warning(f"Section {section} is incomplete or corrupted.")
            validated_sections[section] = False
            continue
        else:
            validated_sections[section] = True

    # If ALL sections validated, then can remove _SUCCESS flags so not in outputted data
    if np.all(validated_sections) and len(validated_sections)==all_sections.shape[0]:
        logger.info("All sections have been successfully validated - removing flags if present.")
        for zarr_dir in save_sections_path.glob("*.zarr"):
            (zarr_dir / "_SUCCESS").unlink(missing_ok=True)
