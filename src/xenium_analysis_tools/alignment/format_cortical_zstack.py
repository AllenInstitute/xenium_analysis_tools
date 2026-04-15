import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import spatialdata as sd
import tifffile
import xarray as xr
from skimage.measure import regionprops_table
from spatialdata.models import Image3DModel, Labels3DModel, TableModel

from xenium_analysis_tools.alignment.align_sections import write_sdata_elements

def extract_size(name):
    import re
    pattern = r'(\d+)x(\d+)x(\d+)'
    match = re.search(pattern, name)

    if match:
        width, height, depth = match.groups()
        return int(width), int(height), int(depth)
    return None, None, None

def load_dataset_config(dataset_id, code_root):
    datasets_path = code_root / "datasets_names_dict.json"
    if not datasets_path.exists():
        raise FileNotFoundError(f"datasets_names_dict.json not found: {datasets_path}")

    with open(datasets_path, "r") as file_handle:
        datasets_config = json.load(file_handle)

    dataset_key = str(dataset_id)
    if dataset_key not in datasets_config:
        raise KeyError(f"Dataset {dataset_key} not found in {datasets_path}")

    return datasets_config[dataset_key]


def resolve_paths_from_dataset(dataset_id, data_root, scratch_root, code_root):
    dataset_config = load_dataset_config(dataset_id, code_root)

    zstack_data_asset_folder = dataset_config.get("zstack_data_asset_folder")
    zstack_masks_folder = dataset_config.get("zstack_masks_folder")
    zstack_img_gcamp_relpath = dataset_config.get("zstack_img_gcamp_path")
    zstack_masks_gcamp_relpath = dataset_config.get("zstack_masks_gcamp_path")
    zstack_img_dextran_relpath = dataset_config.get("zstack_img_dextran_path")
    zstack_masks_dextran_relpath = dataset_config.get("zstack_masks_dextran_path")

    missing_fields = [
        field_name
        for field_name, value in {
            "zstack_data_asset_folder": zstack_data_asset_folder,
            "zstack_masks_folder": zstack_masks_folder,
            "zstack_img_gcamp_path": zstack_img_gcamp_relpath,
            "zstack_masks_gcamp_path": zstack_masks_gcamp_relpath,
        }.items()
        if not value
    ]
    if missing_fields:
        raise ValueError(
            f"Dataset {dataset_id} is missing required z-stack fields: {', '.join(missing_fields)}"
        )

    output_path = (
        scratch_root
        / f"xenium_{dataset_id}_alignment"
        / "zstacks"
        / "zstack.zarr"
    )

    return {
        "gcamp_image": data_root / zstack_data_asset_folder / zstack_img_gcamp_relpath,
        "gcamp_mask": data_root / zstack_masks_folder / zstack_masks_gcamp_relpath,
        "dextran_image": (
            data_root / zstack_data_asset_folder / zstack_img_dextran_relpath
            if zstack_img_dextran_relpath
            else None
        ),
        "dextran_mask": (
            data_root / zstack_masks_folder / zstack_masks_dextran_relpath
            if zstack_masks_dextran_relpath
            else None
        ),
        "output_path": output_path,
    }

def create_zstack_da(
    tif_path,
    name,
    add_chan=True,
    dims=("z", "y", "x"),
    fov_um=(450.0, 400.0, 400.0),
):
    data = tifffile.imread(tif_path)
    pixel_sizes = [fov / pix for fov, pix in zip(fov_um, data.shape)]
    coords = {d: np.arange(data.shape[i]) * pixel_sizes[i] for i, d in enumerate(dims)}
    
    if add_chan:
        data = np.expand_dims(data, axis=0)
        coords["c"] = [name]
        current_dims = ("c",) + dims
    else:
        current_dims = dims
    
    da = xr.DataArray(data, coords=coords, dims=current_dims, name=name)
    da.attrs |= {f"scale_{d}": s for d, s in zip(dims, pixel_sizes)}
    return da

def add_zstack_images(sdata, chan_name, zstack_img_path, chunks=(1, 256, 256, 256)):
    img_da = create_zstack_da(zstack_img_path, name=chan_name)
    img_model = Image3DModel.parse(img_da, c_coords=[chan_name], chunks=chunks)
    sdata.images[chan_name] = img_model
    return sdata

def add_zstack_masks(sdata, chan_name, zstack_masks_path, chunks=(1, 256, 256, 256)):
    img_da = create_zstack_da(zstack_masks_path, name=chan_name)
    img_da = img_da[0]
    img_model = Labels3DModel.parse(img_da, chunks=chunks)
    sdata.labels[f"{chan_name}_labels"] = img_model
    return sdata


def get_mask_props(
    sdata,
    lab,
    include_intensity_props=False,
    rp_props=None,
    intensity_percentiles=(5, 95),
):
    if rp_props is None:
        rp_props = [
            "bbox",
            "area",
            "extent",
            "axis_minor_length",
            "axis_major_length",
            "intensity_mean",
        ]

    label_img = np.asarray(sdata[lab].data).squeeze()

    if include_intensity_props:
        intensity_img = np.asarray(sdata[lab.split("_")[0]].data).squeeze()
        if label_img.shape != intensity_img.shape:
            raise ValueError(
                f"Shape mismatch: labels={label_img.shape}, intensity={intensity_img.shape}"
            )

        def quartiles(regionmask, intensity, percentiles=intensity_percentiles):
            vals = np.asarray(intensity)[regionmask]
            if vals.size == 0:
                return tuple([np.nan] * len(percentiles))
            q = np.nanpercentile(vals, percentiles)
            return tuple(q.tolist())

        df = pd.DataFrame(
            regionprops_table(
                label_img.astype(np.int32),
                properties=["label", "centroid"] + rp_props,
                intensity_image=intensity_img,
                extra_properties=(quartiles,),
                separator="_",
            )
        )

        df = df.rename(
            columns={
                f"quartiles_{i}": f"intensity_perc_{p}"
                for i, p in enumerate(intensity_percentiles)
            }
        )
    else:
        rp_props_no_intensity = [p for p in rp_props if not p.startswith("intensity")]
        df = pd.DataFrame(
            regionprops_table(
                label_img.astype(np.int32),
                properties=["label", "centroid"] + rp_props_no_intensity,
                separator="_",
            )
        )

    return df


def get_zstack_sdata(
    zstack_img_gcamp_path,
    zstack_masks_gcamp_path,
    zstack_img_dextran_path=None,
    zstack_masks_dextran_path=None,
    include_intensity_props=False,
):
    sdata = sd.SpatialData()

    print("Generating images...")
    sdata = add_zstack_images(sdata, "gcamp", zstack_img_gcamp_path)
    if zstack_img_dextran_path is not None:
        sdata = add_zstack_images(sdata, "dextran", zstack_img_dextran_path)

    print("Generating labels...")
    sdata = add_zstack_masks(sdata, "gcamp", zstack_masks_gcamp_path)
    if zstack_masks_dextran_path is not None:
        sdata = add_zstack_masks(sdata, "dextran", zstack_masks_dextran_path)

    print("Generating cell annotations...")
    for lab in sdata.labels.keys():
        labels_df = get_mask_props(
            sdata,
            lab,
            include_intensity_props=include_intensity_props,
        )
        labels_df["region"] = lab
        labels_df.rename(columns={"label": "cell_labels"}, inplace=True)

        adata = ad.AnnData(obs=labels_df.reset_index(drop=True))
        lab_table = TableModel.parse(
            adata,
            region=lab,
            region_key="region",
            instance_key="cell_labels",
        )
        sdata[f"{lab.split('_')[0]}_table"] = lab_table

    print("Adding metadata...")
    gcamp_metadata_path = zstack_masks_gcamp_path.parent / "segmentation_processing.json"
    if gcamp_metadata_path.exists():
        with open(gcamp_metadata_path, "r") as file_handle:
            segmentation_metadata = json.load(file_handle)
        sdata["gcamp_table"].uns["segmentation_metadata"] = segmentation_metadata
        sdata["gcamp_table"].uns["segmentation_metadata"].update(
            {"image_source": zstack_img_gcamp_path.name}
        )
        sdata["gcamp_table"].uns["segmentation_metadata"].update(
            {"segmentation_source": zstack_masks_gcamp_path.name}
        )

    if zstack_masks_dextran_path is not None and zstack_img_dextran_path is not None:
        dextran_metadata_path = zstack_masks_dextran_path.parent / "segmentation_processing.json"
        if dextran_metadata_path.exists():
            with open(dextran_metadata_path, "r") as file_handle:
                segmentation_metadata = json.load(file_handle)
            sdata["dextran_table"].uns["segmentation_metadata"] = segmentation_metadata
            sdata["dextran_table"].uns["segmentation_metadata"].update(
                {"image_source": zstack_img_dextran_path.name}
            )
            sdata["dextran_table"].uns["segmentation_metadata"].update(
                {"segmentation_source": zstack_masks_dextran_path.name}
            )

    return sdata

def build_parser():
    parser = argparse.ArgumentParser(
        description="Build a cortical z-stack SpatialData object from image and mask TIFFs."
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=None,
        help="Dataset ID to resolve paths from code/datasets_names_dict.json.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/root/capsule/data"),
        help="Root folder containing dataset assets.",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path("/root/capsule/scratch"),
        help="Root folder for generated alignment outputs.",
    )
    parser.add_argument(
        "--code-root",
        type=Path,
        default=Path("/root/capsule/code"),
        help="Root folder containing datasets_names_dict.json.",
    )
    parser.add_argument(
        "--gcamp-image",
        type=Path,
        default=None,
        help="Path to the GCaMP image TIFF.",
    )
    parser.add_argument(
        "--gcamp-mask",
        type=Path,
        default=None,
        help="Path to the GCaMP segmentation mask TIFF.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Destination zarr path for the output SpatialData object.",
    )
    parser.add_argument(
        "--dextran-image",
        type=Path,
        default=None,
        help="Optional path to the dextran image TIFF.",
    )
    parser.add_argument(
        "--dextran-mask",
        type=Path,
        default=None,
        help="Optional path to the dextran segmentation mask TIFF.",
    )
    parser.add_argument(
        "--include-intensity-props",
        action="store_true",
        help="Include intensity-based regionprops and percentile summaries.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output zarr if it already exists.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers to use when writing SpatialData elements.",
    )
    return parser


def populate_args_from_dataset(args):
    if args.dataset_id is None:
        return args

    resolved_paths = resolve_paths_from_dataset(
        dataset_id=args.dataset_id,
        data_root=args.data_root,
        scratch_root=args.scratch_root,
        code_root=args.code_root,
    )

    if args.gcamp_image is None:
        args.gcamp_image = resolved_paths["gcamp_image"]
    if args.gcamp_mask is None:
        args.gcamp_mask = resolved_paths["gcamp_mask"]
    if args.dextran_image is None:
        args.dextran_image = resolved_paths["dextran_image"]
    if args.dextran_mask is None:
        args.dextran_mask = resolved_paths["dextran_mask"]
    if args.output_path is None:
        args.output_path = resolved_paths["output_path"]

    return args


def validate_args(args):
    missing_required = [
        argument_name
        for argument_name, value in {
            "gcamp_image": args.gcamp_image,
            "gcamp_mask": args.gcamp_mask,
            "output_path": args.output_path,
        }.items()
        if value is None
    ]
    if missing_required:
        raise ValueError(
            "Missing required inputs. Provide --dataset-id or set these explicitly: "
            + ", ".join(missing_required)
        )

    required_paths = {
        "gcamp_image": args.gcamp_image,
        "gcamp_mask": args.gcamp_mask,
    }
    optional_paths = {
        "dextran_image": args.dextran_image,
        "dextran_mask": args.dextran_mask,
    }

    for label, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    for label, path in optional_paths.items():
        if path is not None and not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    if (args.dextran_image is None) != (args.dextran_mask is None):
        raise ValueError(
            "Provide both --dextran-image and --dextran-mask together, or omit both."
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args = populate_args_from_dataset(args)
    validate_args(args)

    czstack_sdata = get_zstack_sdata(
        zstack_img_gcamp_path=args.gcamp_image,
        zstack_masks_gcamp_path=args.gcamp_mask,
        zstack_img_dextran_path=args.dextran_image,
        zstack_masks_dextran_path=args.dextran_mask,
        include_intensity_props=args.include_intensity_props,
    )

    write_sdata_elements(
        czstack_sdata,
        args.output_path,
        overwrite=args.overwrite,
        num_workers=args.num_workers,
    )
    print(f"Wrote cortical z-stack SpatialData to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())