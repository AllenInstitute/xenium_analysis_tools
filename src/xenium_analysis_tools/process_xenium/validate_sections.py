import time
import gc
import zarr
import pandas as pd
from pathlib import Path
import dask.array as da
import spatialdata as sd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from xenium_analysis_tools.utils.io_utils import is_complete_store

def validate_section_sizes(section, section_zarr, sections_df, plot_fovs=False):
    """
    Validate bounding box sizes against image dimensions.
    """
    try:
        section_row = sections_df[sections_df['section'] == section]
        if section_row.empty:
            return {'match': None, 'error': 'Section not found in dataframe'}
        
        # Ensure slide_sections is available (handled in validate_all_sections)
        slide_sections = section_row.iloc[0]['slide_sections']
        
        # Skip single-section slides if they weren't cropped
        if len(slide_sections) <= 1:
            return {'match': None}
                
        # Load bounding box data
        section_bboxes_path = section_zarr / 'tables' / 'table' / 'uns' / 'sections_bboxes' / str(section)
        if not section_bboxes_path.exists():
            return {'match': False, 'error': 'Bounding box path does not exist'}
        
        # Open Zarr group for bbox
        section_bboxes = zarr.open_group(str(section_bboxes_path), mode='r')
        bbox_data = {key: section_bboxes[key][()] for key in section_bboxes.keys()}
        
        # Load image data
        section_mf_zarr = section_zarr / 'images' / 'morphology_focus' / '0'
        if not section_mf_zarr.exists():
            return {'match': False, 'error': 'Morphology focus image does not exist'}
        
        mf_dapi_chan = da.from_zarr(str(section_mf_zarr))
        
        # Calculate dimensions
        y_bbox_size = bbox_data['y_max'] - bbox_data['y_min']
        x_bbox_size = bbox_data['x_max'] - bbox_data['x_min']
        y_img_size = mf_dapi_chan.shape[1]
        x_img_size = mf_dapi_chan.shape[2]
        
        # Check match
        size_match = (y_bbox_size == y_img_size and x_bbox_size == x_img_size)
        
        if not size_match:
            msg = (f"Mismatch: Img({y_img_size}, {x_img_size}) vs "
                   f"BBox({y_bbox_size}, {x_bbox_size})")
            logger.warning(f"Section {section}: {msg}")
            return {'match': False, 'error': msg}
        
        return {'match': True}
        
    except Exception as e:
        error_msg = f"Size validation error: {str(e)}"
        logger.error(error_msg)
        return {'match': False, 'error': error_msg}

def validate_zarr(section_zarr, sections_df, test_load=False, test_sizes=False):
    # Extract section ID from filename (assuming 'section_101.zarr')
    try:
        section = int(section_zarr.stem.split('_')[1])
    except (IndexError, ValueError):
        logger.error(f"Could not parse section ID from {section_zarr.name}")
        return {'overall_pass': False, 'section': str(section_zarr.name)}

    results = {
        'section': section,
        'is_complete': False,
        'load_success': True,
        'load_time': None,
        'load_error': None,
        'size_match': None,
        'size_error': None,
        'overall_pass': False
    }
    
    # Test 1: Structure
    if not section_zarr.is_dir():
        results['load_error'] = "Directory does not exist"
        results['load_success'] = False
        return results
    
    results['is_complete'] = is_complete_store(section_zarr)
    
    # Test 2: Load
    if test_load and results['is_complete']:
        try:
            start_load = time.time()
            sdata = sd.read_zarr(section_zarr)
            # Basic sanity check (accessing shapes forces a small read)
            _ = sdata.shapes.keys()
            results['load_time'] = time.time() - start_load
            del sdata
            gc.collect()
        except Exception as e:
            results['load_success'] = False
            results['load_error'] = str(e)
            if 'sdata' in locals(): del sdata
    
    # Test 3: Sizes
    if test_sizes and results['is_complete'] and results['load_success']:
        size_result = validate_section_sizes(section, section_zarr, sections_df)
        results['size_match'] = size_result['match']
        results['size_error'] = size_result.get('error')
    
    # Overall Pass Logic
    # Pass if complete AND loaded AND (size check matched OR size check was skipped/None)
    results['overall_pass'] = (
        results['is_complete'] and
        results['load_success'] and
        (results['size_match'] is not False)
    )
    
    return results

def validate_all_sections(results_folder, sections_df, test_load=True, test_sizes=True):
    """
    Main entry point for validation.
    """
    logger.info(f"Starting validation for {len(sections_df)} sections in {results_folder}")
    
    # --- HELPER: Ensure 'slide_sections' column exists ---
    # The pipeline uses 'slide_id' (e.g., '1_2'). We need to split this back into lists [1, 2]
    # so validate_section_sizes knows if it was a multi-section slide.
    if 'slide_sections' not in sections_df.columns:
        if 'slide_id' in sections_df.columns:
            # Reconstruct list from ID string "1_2_3" -> [1, 2, 3]
            sections_df['slide_sections'] = sections_df['slide_id'].apply(
                lambda x: [int(s) for s in str(x).split('_')] if pd.notnull(x) else []
            )
        else:
            logger.warning("'slide_id' missing from manifest. Skipping size validation.")
            test_sizes = False

    validation_results = []
    
    for section in sections_df['section'].unique():
        section_zarr = results_folder / f'section_{section}.zarr'
        
        logger.info(f"Validating section {section}...")
        result = validate_zarr(section_zarr, sections_df, test_load=test_load, test_sizes=test_sizes)
        validation_results.append(result)
        
        status = "PASS" if result['overall_pass'] else "FAIL"
        logger.info(f"  -> {status}")

    # Create DataFrame
    results_df = pd.DataFrame(validation_results)
    
    # Print/Log Summary
    print_validation_summary(results_df)
    
    return results_df

def print_validation_summary(results_df):
    total = len(results_df)
    passed = results_df['overall_pass'].sum()
    failed = total - passed
    
    logger.info("=== VALIDATION SUMMARY ===")
    logger.info(f"Total sections:      {total}")
    logger.info(f"Sections passed:     {passed} ({passed/total:.1%})")
    logger.info(f"Sections failed:     {failed}")
    
    if failed > 0:
        failed_ids = results_df[~results_df['overall_pass']]['section'].tolist()
        logger.error(f"Failed Section IDs: {failed_ids}")
        # Log reasons for failure
        for _, row in results_df[~results_df['overall_pass']].iterrows():
            reason = []
            if not row['is_complete']: reason.append("Incomplete Zarr")
            if not row['load_success']: reason.append(f"Load Error: {row['load_error']}")
            if row['size_match'] is False: reason.append(f"Size Mismatch: {row['size_error']}")
            logger.error(f"  Section {row['section']}: {', '.join(reason)}")

def plot_section_bboxes(sdata, sections_bboxes, fov_df=None, show_fovs=False, save_path=None, close_fig=True):
    """
    Plots the DAPI image from SpatialData with overlaid Section Bounding Boxes.
    Optionally overlays individual FOVs to verify assignments.
    
    Parameters:
    - sdata: SpatialData object containing the images (expects 'morphology_focus'/'DAPI').
    - sections_bboxes: Dict of {section_id: {x_min, x_max, y_min, y_max...}}.
    - fov_df: (Optional) DataFrame containing FOV metadata. Required if show_fovs=True.
    - show_fovs: (Bool) If True, overlays individual FOV rectangles and names.
    - save_path: (String) Path to save the figure.
    - close_fig: (Bool) Whether to close the figure after plotting.
    """
    try:
        import spatialdata_plot
    except ImportError:
        logger.warning("spatialdata_plot not found. Please install it to use this function.")
        return

    # 1. Setup Figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 2. Render Base Image (DAPI)
    # We render this first to establish the coordinate system and background
    try:
        sdata.pl.render_images(
            'morphology_focus', channel='DAPI', norm=plt.Normalize(vmin=0, vmax=500)
        ).pl.show(ax=ax, colorbar=False, title='')
    except Exception as e:
        logger.warning(f"Plotting render failed (check element names): {e}")
        # Continue anyway to allow plotting boxes on empty canvas if needed
    
    # 3. Setup Consistent Colors
    # specific map to ensure Section 1 box matches Section 1 FOVs
    slide_sections = sorted(list(sections_bboxes.keys()))
    cmap = plt.get_cmap('tab10')
    section_colors = {str(sec): cmap(i % 10) for i, sec in enumerate(slide_sections)}

    # 4. Optional: Draw Individual FOVs
    if show_fovs:
        if fov_df is None:
            logger.warning("show_fovs=True but no fov_df provided. Skipping FOV overlay.")
        else:
            df = fov_df.copy()
            # Calculate bounds if missing
            if 'x_min' not in df.columns:
                df['x_min'] = df['x']
                df['y_min'] = df['y']
                df['width'] = df['width']
                df['height'] = df['height']
            
            # Scale to pixel coords
            df['x_min'] /= df['pixel_size']
            df['y_min'] /= df['pixel_size']
            df['width'] /= df['pixel_size']
            df['height'] /= df['pixel_size']

            for idx, row in df.iterrows():
                sec_id = str(row['section']) if 'section' in df.columns else None
                # Use section color if available, else gray
                color = section_colors.get(sec_id, 'gray')
                
                # Draw FOV Rectangle (Faint)
                rect = patches.Rectangle(
                    (row['x_min'], row['y_min']), 
                    row['width'], row['height'],
                    linewidth=1, edgecolor=color, facecolor=color, alpha=0.15
                )
                ax.add_patch(rect)
                
                # Draw FOV Name (Small text)
                ax.text(
                    row['x_min'] + row['width'] / 2,
                    row['y_min'] + row['height'] / 2,
                    str(row['fov_name']),
                    ha='center', va='center', fontsize=6, color='white', alpha=0.8,
                    fontweight='bold'
                )

    # 5. Draw Section Bounding Boxes (The "Group" container)
    for section_id, bbox in sections_bboxes.items():
        sec_str = str(section_id)
        color = section_colors.get(sec_str, 'white')
        
        # Draw Section Box (Thick outline)
        rect = patches.Rectangle(
            (float(bbox['x_min']), float(bbox['y_min'])),
            float(bbox['x_max'] - bbox['x_min']),
            float(bbox['y_max'] - bbox['y_min']),
            edgecolor=color, facecolor='none', linewidth=3, 
            linestyle='--', label=f"Section {sec_str}"
        )
        ax.add_patch(rect)
        
        # Add Section Label at the top-left of the box
        ax.text(
            float(bbox['x_min']), float(bbox['y_min']) - 20, # Slightly above the box
            f" {sec_str} ", 
            color='white', backgroundcolor=color,
            ha='left', va='bottom', fontweight='bold', fontsize=12,
            bbox=dict(facecolor=color, edgecolor='none', alpha=0.8, boxstyle='round,pad=0.2')
        )

    # 6. Final Polish
    # Create a custom legend for Sections only (avoids cluttering with every FOV)
    legend_elements = [
        patches.Patch(facecolor=section_colors[sec], edgecolor=section_colors[sec], 
                      alpha=0.5, label=f'Section {sec}') 
        for sec in slide_sections
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), title="Sections")
    
    ax.set_title(f"Layout: {len(slide_sections)} Sections" + (" (with FOVs)" if show_fovs else ""))
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if close_fig:
        plt.close(fig)
    else:
        return fig, ax