
from spatialdata.transformations import Scale, Translation, Sequence, Identity, set_transformation
import xarray as xr
import numpy as np
import pandas as pd

def add_micron_coord_sys(sdata, pixel_size=None, z_step=None):
    # Define the pixel scaling factor
    if pixel_size is None and 'table' in sdata:
        pixel_size = sdata['table'].uns['section_metadata']['pixel_size']
    if z_step is None and 'table' in sdata:
        z_step = sdata['table'].uns['section_metadata']['z_step_size']
    else:
        z_step = 1.0

    # 2D Images (channel, y, x)
    # c = 1.0 (channels are discrete)
    scale_yx = Scale([pixel_size, pixel_size], axes=("y", "x"))
    scale_cyx = Scale([1.0, pixel_size, pixel_size], axes=("c", "y", "x"))

    # For 3D Z-Stacks (c, z, y, x)
    # c = 1.0 (channels are discrete)
    # z = 3.0 (microns per plane)
    # y, x = 0.2125 (microns per pixel)
    scale_czyx = Scale(
        [1.0, z_step, pixel_size, pixel_size], 
        axes=("c", "z", "y", "x")
    )

    # Identity transform for elements already in microns
    identity = Identity()

    # --- Images ---
    for image_name in sdata.images:
        dims = sdata[image_name].dims if not isinstance(sdata[image_name], xr.core.datatree.DataTree) else sdata[image_name]['scale0'].dims
        if 'z' in dims:
            set_transformation(
                sdata.images[image_name], 
                scale_czyx, 
                to_coordinate_system="microns"
            )
        else:
            set_transformation(
                sdata.images[image_name], 
                scale_cyx, 
                to_coordinate_system="microns"
            )

    # --- Labels ---
    # Both labels are (y, x)
    for label_name in sdata.labels:
        set_transformation(
            sdata.labels[label_name], 
            scale_yx, 
            to_coordinate_system="microns"
        )

    # --- Shapes & Points ---
    # Already in microns
    for shape_name in sdata.shapes:
        set_transformation(
            sdata.shapes[shape_name], 
            identity, 
            to_coordinate_system="microns"
        )

    for point_name in sdata.points:
        set_transformation(
            sdata.points[point_name], 
            identity, 
            to_coordinate_system="microns"
        )
    return sdata

def add_mapped_cells_cols(sdata, mapped_h5ad_path):
    import scanpy as sc
    mapped_h5ad = sc.read_h5ad(mapped_h5ad_path)
    mapping_obs_cols = np.setdiff1d(mapped_h5ad.obs.columns, sdata['table'].obs.columns)
    if len(mapping_obs_cols) == 0:
        print("No new columns to add from mapped data")
    else:
        print(f"Adding {len(mapping_obs_cols)} columns from mapped data: {mapping_obs_cols}")
        sdata['table'].obs = sdata['table'].obs.merge(
            mapped_h5ad.obs[mapping_obs_cols],
            left_index=True,
            right_index=True,
            how='outer'
        )
    mapping_vars_cols = np.setdiff1d(mapped_h5ad.var.columns, sdata['table'].var.columns)
    if len(mapping_vars_cols) == 0:
        print("No new columns to add from mapped data")
    else:
        print(f"Adding {len(mapping_vars_cols)} columns from mapped data: {mapping_vars_cols}")
        sdata['table'].var = sdata['table'].var.merge(
            mapped_h5ad.var[mapping_vars_cols],
            left_index=True,
            right_index=True,
            how='outer'
        )
    return sdata

def get_transcripts_bboxes(transcripts, id_col='cell_labels'):
    transcripts = transcripts.compute() if hasattr(transcripts, 'compute') else transcripts
    # If no transcripts, return empty dict quickly
    cell_label_bboxes = {}
    if transcripts.shape[0] == 0:
        cell_label_bboxes = {}
    else:
        # Aggregate min/max per cell label for z, y, x
        grouped = transcripts.groupby(id_col)[['z', 'y', 'x']].agg(['min', 'max'])

        import numpy as np
        for cell_label, row in grouped.iterrows():
            # Skip background / unmapped label if present
            if cell_label == 0:
                continue
            z_min = int(np.floor(row[('z', 'min')]))
            y_min = int(np.floor(row[('y', 'min')]))
            x_min = int(np.floor(row[('x', 'min')]))
            z_max = int(np.ceil(row[('z', 'max')]))
            y_max = int(np.ceil(row[('y', 'max')]))
            x_max = int(np.ceil(row[('x', 'max')]))
            cell_label_bboxes[cell_label] = (z_min, y_min, x_min, z_max, y_max, x_max)
    return cell_label_bboxes