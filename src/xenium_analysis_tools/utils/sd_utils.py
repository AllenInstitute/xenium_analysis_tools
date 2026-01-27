import spatialdata as sd
from spatialdata.transformations import Scale, Identity, set_transformation

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
        if 'z' in sdata[image_name].dims:
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