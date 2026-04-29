from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from collections import defaultdict

#############################################
################ Plots ######################
#############################################
def plot_section_overview(section_bundle_paths, dataset_id=None, save_path=None):
    '''
    Plot overview scans with FOV locations for each section bundle.
    Args:
        section_bundle_paths: dict of {section: path} for each section bundle.
        dataset_id: Optional dataset identifier for plot titles.
        save_path: Optional path to save the resulting figure.
    '''
    sections_by_slide = defaultdict(dict)
    for sec, path in section_bundle_paths.items():
        if not path:
            continue
        exp = json.loads((Path(path) / "experiment.xenium").read_text())
        slide_id = exp.get("slide_id", "unknown")
        sections_by_slide[slide_id][sec] = path

    slide_ids = sorted(sections_by_slide.keys())
    n_slides = len(slide_ids)

    # Count unique bundles (for colormap sizing)
    all_unique_bundles = {path for paths in sections_by_slide.values() for path in paths.values()}
    cmap = plt.colormaps["tab20"].resampled(max(len(all_unique_bundles), 1))

    fig, axes = plt.subplots(1, n_slides, figsize=(5 * n_slides, 10))
    if n_slides == 1:
        axes = [axes]

    color_idx = 0
    for ax, slide_id in zip(axes, slide_ids):
        sections = sections_by_slide[slide_id]
        first_path = next(iter(sections.values()))
        overview_img = Image.open(Path(first_path) / "aux_outputs" / "overview_scan.png")
        ax.imshow(overview_img)

        # Group section numbers by bundle path
        bundle_to_sections = defaultdict(list)
        for sec, path in sections.items():
            bundle_to_sections[path].append(sec)

        for path, sec_list in sorted(bundle_to_sections.items(), key=lambda x: min(x[1])):
            bundle = Path(path)
            fov_locs = json.loads((bundle / "aux_outputs" / "overview_scan_fov_locations.json").read_text())["fov_locations"]
            color = cmap(color_idx)
            color_idx += 1
            label = " & ".join(str(s) for s in sorted(sec_list))

            xs, ys = [], []
            for loc in fov_locs.values():
                rect = patches.Rectangle(
                    (loc["x"], loc["y"]), loc["width"], loc["height"],
                    linewidth=0.8, edgecolor=color, facecolor=color, alpha=0.3
                )
                ax.add_patch(rect)
                xs.append(loc["x"] + loc["width"] / 2)
                ys.append(loc["y"] + loc["height"] / 2)

            cx, cy = np.mean(xs), np.mean(ys)
            ax.text(cx, cy, label, color="white", fontsize=9, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7, linewidth=0))

        title = f"Slide {slide_id}" if dataset_id is None else f"Dataset {dataset_id}  |  Slide {slide_id}"
        ax.set_title(title, fontsize=13)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

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
        print("spatialdata_plot not found. Please install it to use this function.")
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Base Image (DAPI) — find the actual channel name (may have been renamed to 'dapi')
    try:
        import xarray as xr
        mf = sdata['morphology_focus']
        if hasattr(mf, 'keys'):  # multiscale DataTree
            import spatialdata as _sd
            mf_img = _sd.get_pyramid_levels(mf, n=0)
            mf_img = mf_img['image'] if hasattr(mf_img, 'image') else mf_img
        else:
            mf_img = mf
        chan_names = list(mf_img.coords['c'].values) if 'c' in mf_img.coords else []
        dapi_chan = next((c for c in chan_names if c.lower() == 'dapi'), chan_names[0] if chan_names else 'DAPI')
        sdata.pl.render_images(
            'morphology_focus', channel=dapi_chan, norm=plt.Normalize(vmin=0, vmax=500)
        ).pl.show(ax=ax, colorbar=False, title='', coordinate_systems=['global'])
    except Exception as e:
        print(f"Plotting render failed (check element names): {e}")
    
    slide_sections = sorted(list(sections_bboxes.keys()))
    cmap = plt.get_cmap('tab10')
    section_colors = {str(sec): cmap(i % 10) for i, sec in enumerate(slide_sections)}

    # Plot FOVs
    if show_fovs:
        if fov_df is None:
            print("show_fovs=True but no fov_df provided. Skipping FOV overlay.")
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

    # Draw section bboxes
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