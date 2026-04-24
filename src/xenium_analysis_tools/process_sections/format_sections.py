from pathlib import Path
import json
import pandas as pd
from sklearn.cluster import KMeans
import spatialdata_io
import shutil
import os
import spatialdata as sd
import xarray as xr
import numpy as np
import dask.array as da
import zarr
import tifffile
from tqdm.notebook import tqdm
from spatialdata import bounding_box_query
from collections import defaultdict

from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    PointsModel,
    TableModel,
)

from spatialdata.transformations import (
    get_transformation,
    set_transformation,
    remove_transformation,
    Identity,
    Scale,
    Sequence,
)

from xenium_analysis_tools.utils.sd_utils import (
    extract_scale_transform,
    write_sdata_elements,
    add_micron_coord_sys,
    get_ome_metadata,
    rename_chans,
)
from xenium_analysis_tools.process_sections.validation_plots import plot_section_bboxes, plot_section_overview

#############################################
##### Find Xenium bundles for sections ######
#############################################
class SectionFinder:
    """Finds Xenium output directories for sections in a reference table."""
    def __init__(self, data_dirs: list[str | Path],
                 ref_table_path: str | Path | None = None,
                 sections_bundles: dict | None = None):
        self.dirs = [Path(d) for d in data_dirs]
        self.ref_table = pd.read_csv(ref_table_path) if ref_table_path else None
        self.sections_bundles = sections_bundles or {}

    def _is_match(self, path: Path, parts: list[str]) -> bool:
        name = path.name.lower()
        return all(pt in name for pt in parts)

    def _search(self, path: Path, parts: list[str], depth: int) -> str | None:
        if self._is_match(path, parts):
            return str(path)
        if depth == 0 or path.name.lower().startswith("output-xet"):
            return None
        try:
            for child in path.iterdir():
                if child.is_dir():
                    result = self._search(child, parts, depth - 1)
                    if result:
                        return result
        except PermissionError:
            pass
        return None

    def find(self, slide_id: str | int, roi_name: str | int) -> str | None:
        """Return the path matching slide_id + roi_name, or None if not found."""
        parts = [str(slide_id).lower(), str(roi_name).lower()]
        for d in self.dirs:
            if not d.exists():
                continue
            if d.name.lower().startswith("output-xet"):
                if self._is_match(d, parts):
                    return str(d)
            else:
                result = self._search(d, parts, depth=4)
                if result:
                    return result
        return None

    def find_by_bundle_name(self, bundle_name: str) -> str | None:
        """Return the path whose directory name exactly matches bundle_name, or None."""
        parts = [bundle_name.lower()]
        for d in self.dirs:
            if not d.exists():
                continue
            if d.name.lower() == bundle_name.lower():
                return str(d)
            result = self._search(d, parts, depth=4)
            if result:
                return result
        return None

    def find_all(self, ref_table: pd.DataFrame | None = None,
                 sections_bundles: dict | None = None,
                 slide_col: str = "slide_id",
                 roi_col: str = "ROI_name",
                 section_col: str = "section") -> dict[int, str | None]:
        """Return {section: path} for every section.

        Uses sections_bundles (argument, then instance default) if non-empty —
        the ref table is not consulted in that case. Falls back to ref table
        lookup when no bundle names are available.
        """
        known = sections_bundles if sections_bundles is not None else self.sections_bundles
        if known:
            return {
                sec: self.find_by_bundle_name(bundle_name) if bundle_name else None
                for sec, bundle_name in known.items()
            }

        table = ref_table if ref_table is not None else self.ref_table
        if table is None:
            raise ValueError("No sections_bundles or reference table provided.")
        return {
            row[section_col]: self.find(row[slide_col], row[roi_col])
            for _, row in table.iterrows()
        }

    def update_datasets_info(self, dataset_id: int | str, datasets_info_path: str | Path,
                              ref_table: pd.DataFrame | None = None,
                              section_paths: dict | None = None) -> None:
        """Update the bundle_names entry for dataset_id in datasets_names_dict.json.

        If section_paths ({section: path}) is provided, uses it directly and
        replaces the existing sections_bundles entirely. Otherwise looks up
        paths via find_all and merges into the existing entry.
        """
        datasets_info_path = Path(datasets_info_path)
        with open(datasets_info_path) as f:
            datasets_info = json.load(f)

        if section_paths is None:
            section_paths = self.find_all(ref_table)
            replace = False
        else:
            replace = True
        bundle_names_dict = {sec: Path(p).name if p else None for sec, p in section_paths.items()}

        xenium_entry = (
            datasets_info
            .setdefault(str(dataset_id), {})
            .setdefault("paths", {})
            .setdefault("xenium", {})
        )
        if replace or not isinstance(xenium_entry.get("sections_bundles"), dict):
            xenium_entry["sections_bundles"] = {}
        sections_bundles = xenium_entry["sections_bundles"]
        for sec, bundle_name in bundle_names_dict.items():
            if bundle_name:
                sections_bundles[sec] = bundle_name

        with open(datasets_info_path, "w") as f:
            json.dump(datasets_info, f, indent=4)


def auto_detect_section_order(bundle_paths):
    """Detect incorrect section ordering and return {old_sec: new_sec} corrections.

    Within each slide, sections are expected to be numbered in ascending order
    spatially: left column top-to-bottom, then right column top-to-bottom.
    A negative correlation between section numbers and their spatial rank
    indicates the labeling direction was reversed, in which case a reversal
    map (new = min + max - old) is returned for that slide's sections.

    Parameters
    ----------
    bundle_paths : dict
        {"sec1_sec2": bundle_path, ...} as produced by the bundle_to_sections
        grouping step.

    Returns
    -------
    dict
        {old_sec: new_sec} corrections. Empty dict if ordering is already correct.
    """
    slide_bundles = {}
    for bundle_key, bundle_path in bundle_paths.items():
        overview_fov_path = Path(bundle_path) / "aux_outputs" / "overview_scan_fov_locations.json"
        with open(overview_fov_path) as _f:
            _fov_locs = json.load(_f)["fov_locations"]
        x_centroid = np.mean([v["x"] + v["width"] / 2 for v in _fov_locs.values()])
        y_centroid = np.mean([v["y"] + v["height"] / 2 for v in _fov_locs.values()])

        name = Path(bundle_path).name
        slide_id = name.split('__')[1] if '__' in name else name

        slide_bundles.setdefault(slide_id, []).append({
            'sections': sorted(int(s) for s in bundle_key.split('_')),
            'x_centroid': x_centroid,
            'y_centroid': y_centroid,
        })

    corrections = {}
    for slide_id, bundles in slide_bundles.items():
        x_vals = np.array([b['x_centroid'] for b in bundles])
        y_vals = np.array([b['y_centroid'] for b in bundles])
        x_range = x_vals.max() - x_vals.min()
        y_range = y_vals.max() - y_vals.min()

        # Detect multi-column layout: split at the largest gap in x-centroids.
        # Using the gap (rather than KMeans or median) handles uneven column
        # sizes (e.g. 3 bundles left, 5 bundles right) without misclassification.
        if len(bundles) > 2 and x_range > 0.3 * y_range:
            x_sorted = np.sort(x_vals)
            gap_idx = int(np.argmax(np.diff(x_sorted)))
            x_thresh = (x_sorted[gap_idx] + x_sorted[gap_idx + 1]) / 2
            left  = sorted([b for b in bundles if b['x_centroid'] <  x_thresh], key=lambda b: b['y_centroid'])
            right = sorted([b for b in bundles if b['x_centroid'] >= x_thresh], key=lambda b: b['y_centroid'])
            ordered = left + right
        else:
            ordered = sorted(bundles, key=lambda b: b['y_centroid'])

        # Assign a cumulative rank midpoint to each bundle (accounting for
        # multi-section bundles occupying more than one sequential position).
        pos, rank_mids = 0, []
        for b in ordered:
            n = len(b['sections'])
            rank_mids.append(pos + (n + 1) / 2.0)
            pos += n

        mean_sections = np.array([np.mean(b['sections']) for b in ordered])
        corr = float(np.corrcoef(mean_sections, rank_mids)[0, 1]) if len(ordered) > 1 else 1.0

        if corr < -0.5:
            # Sections are numbered in the reverse spatial direction.
            # Apply the reversal: new = lo + hi - old  (e.g. 1..10 → 10..1)
            all_sections = sorted(s for b in bundles for s in b['sections'])
            lo, hi = all_sections[0], all_sections[-1]
            for s in all_sections:
                new_s = lo + hi - s
                if new_s != s:
                    corrections[s] = new_s

    return corrections

def get_section_bundles(
    dataset_id,
    dataset_info,
    datasets_info_path,
    search_folders=None,
    ref_table_path=None,
    data_root=Path("/data/"),
    plots_folder=None,
):
    """Find, validate, and (if needed) correct section-to-bundle mappings.

    Parameters
    ----------
    dataset_id : int | str
    dataset_info : dict
        The entry for this dataset from datasets_names_dict.json.
    datasets_info_path : Path | str
        Path to datasets_names_dict.json (used to persist corrected ordering).
    search_folders : list[str], optional
        Sub-folders of data_root to search. Defaults to ['mfish-xenium-u01-learning'].
    ref_table_path : Path | str | None
        Reference table CSV; used when sections_bundles is not in dataset_info.
    data_root : Path
    plots_folder : Path | None
        If provided, section-overview plots are saved here.

    Returns
    -------
    dict
        {section: bundle_path} for all sections, with corrected ordering applied.
    """
    if search_folders is None:
        search_folders = ['mfish-xenium-u01-learning']

    finder = SectionFinder(
        data_dirs=[data_root / folder for folder in search_folders],
        ref_table_path=ref_table_path,
        sections_bundles=dataset_info.get('paths', {}).get('xenium', {}).get('sections_bundles'),
    )

    # {section_number: bundle_path}
    section_bundle_paths = finder.find_all()

    # {bundle_key: bundle_path}  e.g. {"1_2": "/path/to/bundle"}
    bundle_to_sections = defaultdict(list)
    for sec, path in section_bundle_paths.items():
        bundle_to_sections[path].append(sec)
    bundle_paths = {
        "_".join(str(s) for s in sorted(sections)): path
        for path, sections in bundle_to_sections.items()
    }

    if plots_folder is not None:
        plot_section_overview(
            section_bundle_paths, dataset_id=dataset_id,
            save_path=plots_folder / "sections_order.png",
        )

    corrected_sections = auto_detect_section_order(bundle_paths)
    if corrected_sections:
        print("Detected ordering corrections (original → corrected):")
        for orig, corr in sorted(corrected_sections.items()):
            print(f"  section {orig:>3} → {corr}")

        # Remap {section: path} using the correction map
        corrected_section_bundle_paths = {
            corrected_sections.get(s_n, s_n): path
            for s_n, path in section_bundle_paths.items()
        }

        if plots_folder is not None:
            plot_section_overview(
                corrected_section_bundle_paths, dataset_id=dataset_id,
                save_path=plots_folder / "sections_order_corrected.png",
            )

        # Persist the corrected bundle names so subsequent runs skip re-detection
        finder.update_datasets_info(
            dataset_id, datasets_info_path,
            section_paths=corrected_section_bundle_paths,
        )
        return corrected_section_bundle_paths
    else:
        print("Section ordering is already correct — no corrections needed.")
        return section_bundle_paths

##############################
######## Read bundles ########
##############################

def generate_section_sdata(bundle_path, save_path, cells_as_circles=True, gex_only=False, n_jobs=4):
    import dask
    sdata = spatialdata_io.xenium(
            path=bundle_path,
            cells_as_circles=cells_as_circles,
            gex_only=gex_only,
            n_jobs=n_jobs,
    )
    with dask.config.set(scheduler='threads', num_workers=n_jobs):
        sdata.write(save_path)
    del sdata

def save_section_sdata(bundle_path, save_path, cells_as_circles=True, gex_only=False, n_jobs=4):
    # Function for parsing if the sdata already exists, and if not, generating it with error handling to catch and clean up corrupted files.
    if save_path.exists():
        try:
            sd.read_zarr(save_path)
            print(f"SpatialData for section already exists at {save_path}. Skipping generation.")
        except Exception as e:
            print(f"Error reading existing sdata: {e}. Deleting corrupted file and regenerating.")
            shutil.rmtree(save_path)
            try:
                generate_section_sdata(bundle_path, save_path, cells_as_circles=cells_as_circles, gex_only=gex_only, n_jobs=n_jobs)
            except Exception as e:
                print(f"Error generating sdata: {e}. Delete partial data at {save_path} and try again.")
                if save_path.exists():
                    shutil.rmtree(save_path)
    else:
        try:
            generate_section_sdata(bundle_path, save_path, cells_as_circles=cells_as_circles, gex_only=gex_only, n_jobs=n_jobs)
        except Exception as e:
            print(f"Error generating sdata: {e}. Delete partial data at {save_path} and try again.")
            if save_path.exists():
                shutil.rmtree(save_path)

def check_existing_sdata_paths(save_paths):
    valid_paths = []
    for sp in save_paths:
        sp = Path(sp)
        if sp.exists():
            try:
                sd.read_zarr(sp)
                valid_paths.append(True)
                print(f"Existing SpatialData at {sp} is valid.")
            except Exception as e:
                print(f"Error reading existing sdata at {sp}: {e}. Consider deleting this file to regenerate.")
                valid_paths.append(False)
        else:
            print(f"{sp.name} doesn't exist - will generate")
            valid_paths.append(False)
    if all(valid_paths):
        print("All existing SpatialData files are valid. Skipping generation.")
        return True
    else:
        print("Some SpatialData files are missing or corrupted.")
        return False

#############################
#### Sections metadata ######
#############################

def get_fov_metadata(xenium_data_path):
    """Add FOV metadata with section assignments"""
    xenium_data_path = Path(xenium_data_path)
    morphology_fov_locations_file = xenium_data_path / 'aux_outputs' / 'morphology_fov_locations.json'
    with open(morphology_fov_locations_file, 'r') as f:
        fov_locations = json.load(f)

    fov_df = pd.DataFrame.from_dict(fov_locations['fov_locations'], orient='index')
    fov_df.reset_index(names='fov_name', inplace=True)
    fov_df['units'] = fov_locations.get('units', 'microns')
    return fov_df

def get_section_metadata(xenium_data_path):
    """Add section metadata to anndata"""
    xenium_data_path = Path(xenium_data_path)
    metrics_summary_csv = xenium_data_path / 'metrics_summary.csv'
    section_metadata = pd.read_csv(metrics_summary_csv, nrows=1)
    section_metadata = section_metadata.iloc[0].replace({np.nan: None}).to_dict()
    
    experiment_xenium_json = xenium_data_path / 'experiment.xenium'
    with open(experiment_xenium_json, 'r') as f:
        experiment_metadata = json.load(f)
    
    section_metadata.update(experiment_metadata)
    section_metadata.update({'xenium_bundle_name': xenium_data_path.name})

    return section_metadata

def assign_fov_sections(df, section_order, print_counts=False):
    """
    Assign section IDs using K-Means clustering and pre-calculate 
    bounding box coordinates for downstream processing.
    """
    df = df.copy()
    
    # 1. Calculate boundaries (Required for get_fov_bboxes)
    df['x_min'] = df['x']
    df['x_max'] = df['x'] + df['width']
    df['y_min'] = df['y']
    df['y_max'] = df['y'] + df['height']

    # 2. Find centroids for Clustering
    df['cx'] = df['x'] + df['width'] / 2
    df['cy'] = df['y'] + df['height'] / 2
    
    # 3. Perform K-Means Clustering
    n_sections = len(section_order)
    coords = df[['cx', 'cy']].values
    
    # random_state=42 ensures reproducibility
    kmeans = KMeans(n_clusters=n_sections, random_state=42, n_init=10)
    df['cluster_label'] = kmeans.fit_predict(coords)
    
    # 4. Map Clusters to Spatial Order
    # Calculate the average position for each cluster to sort them
    cluster_stats = df.groupby('cluster_label').agg({
        'cy': 'mean', 
        'cx': 'mean'
    }).reset_index()
    
    # Determine dominant axis (Vertical vs Horizontal)
    y_range = df['cy'].max() - df['cy'].min()
    x_range = df['cx'].max() - df['cx'].min()
    
    if y_range > x_range:
        # Sort Top-to-Bottom
        cluster_stats = cluster_stats.sort_values('cy')
    else:
        # Sort Left-to-Right
        cluster_stats = cluster_stats.sort_values('cx')
        
    label_to_order = {label: idx for idx, label in enumerate(cluster_stats['cluster_label'])}
    
    df['section'] = df['cluster_label'].map(label_to_order).map(lambda i: section_order[i])
    
    if print_counts:
        print(f"Assigned {n_sections} sections using K-Means.")
        for sec in section_order:
            count = (df['section'] == sec).sum()
            print(f"Section {sec}: {count} FOVs")
        
    return df.drop(columns=['cx', 'cy', 'cluster_label'])

def get_fov_bboxes(fov_metadata):
    """Vectorized calculation of FOV bounding boxes per section."""
    grouped = fov_metadata.groupby('section')
    pixel_sizes = grouped['pixel_size'].first()
    
    bboxes = grouped.agg({
        'x_min': 'min', 'x_max': 'max',
        'y_min': 'min', 'y_max': 'max'
    })
    
    bboxes_normalized = bboxes.div(pixel_sizes, axis=0)
    bboxes_normalized = bboxes_normalized.round().astype(int)
    return bboxes_normalized.to_dict(orient='index')

def process_metadata(sdata, xenium_bundle_path, slide_sections):
    # Main function to process metadata and store in anndata.uns
    anndata = sdata['table'].copy()
    section_metadata = get_section_metadata(xenium_bundle_path)
    section_metadata['sections_on_slide'] = slide_sections

    fov_metadata = get_fov_metadata(xenium_bundle_path)
    fov_metadata = assign_fov_sections(fov_metadata, sorted(slide_sections))
    fov_metadata['pixel_size'] = section_metadata.get('pixel_size', 0.2125)

    section_bboxes = get_fov_bboxes(fov_metadata)
    section_bboxes_str_keys = {str(k): v for k, v in section_bboxes.items()}
    section_bboxes = section_bboxes_str_keys

    anndata.uns['section_metadata'] = section_metadata
    anndata.uns['sections_bboxes'] = section_bboxes    
    anndata.uns['fov_metadata'] = fov_metadata
    sdata['table'] = anndata

    return sdata


######################################################
###### Formatting after generating SpatialData #######
######################################################

def fix_points_category(sdata):
    txs = sdata['transcripts'].copy()
    txs["feature_name"] = txs["feature_name"].cat.as_known()
    del sdata['transcripts']
    txs = PointsModel.parse(txs)
    sdata.points['transcripts'] = txs
    return sdata

def get_dapi_zstack(ome_tif_path, sdata):
    if not ome_tif_path.exists():
        raise FileNotFoundError(f"DAPI file not found: {ome_tif_path}")

    datatree_obj = xr.DataTree()
    tiff_store = tifffile.imread(ome_tif_path, aszarr=True)
    tiff_store.path = str(ome_tif_path)
    z_tiff = zarr.open(tiff_store, mode='r')

    for level_name in list(sdata['morphology_focus'].keys()):
        level_id = level_name.rsplit('scale')[1]
        dask_data = da.from_zarr(z_tiff[level_id])
        dask_data = dask_data[np.newaxis, :, :, :]

        parsed_image = Image3DModel.parse(
                dask_data, dims=['c', 'z', 'y', 'x'], c_coords=['DAPI'], chunks='auto',
        )
        parsed_image.attrs.update(get_ome_metadata(ome_tif_path, level_n=int(level_id)))
        parsed_image.attrs['scale_level'] = int(level_id)
        datatree_obj[f'scale{level_id}'] = xr.Dataset({'image': parsed_image})
    return datatree_obj

def crop_dapi_image(dapi_image, bbox):
    """Crop a multiscale DAPI DataTree to a pixel bounding box."""
    cropped_dapi_image = xr.DataTree()
    for scale_level in dapi_image.keys():
        level_image = dapi_image[scale_level].image
        if scale_level != 'scale0':
            scale_factors = (
                np.array(dapi_image['scale0'].image.shape)
                / np.array(level_image.shape)
            )
            scale_tf = Scale(scale_factors, axes=level_image.dims)
            level_image = level_image.copy()  # don't mutate the shared cached DataTree
            set_transformation(
                level_image,
                Sequence([scale_tf, Identity()]),
                to_coordinate_system='global',
            )
        cropped_scale = bounding_box_query(
            level_image,
            axes=('y', 'x'),
            min_coordinate=[bbox['y_min'], bbox['x_min']],
            max_coordinate=[bbox['y_max'], bbox['x_max']],
            target_coordinate_system='global',
        )
        parsed_image = Image3DModel.parse(
            cropped_scale, dims=['c', 'z', 'y', 'x'], c_coords=['DAPI'], chunks='auto',
        )
        cropped_dapi_image[scale_level] = xr.Dataset({'image': parsed_image})
    return cropped_dapi_image

def crop_filter_sdata(sdata, bbox, c_system='global', crop_transcripts_separately=True, transcripts_df=None):
    """Crop SpatialData to a bounding box, handling transcripts separately to avoid
    index issues with multi-partition Dask DataFrames.

    Parameters
    ----------
    transcripts_df : pd.DataFrame | None
        Pre-computed (already .compute()'d) transcript DataFrame. When provided the
        parquet read is skipped, which is the main speedup for multi-section bundles.
    """
    if crop_transcripts_separately:
        element_names = []
        for attr_name in ['images', 'labels', 'shapes', 'tables']:
            element_names.extend(getattr(sdata, attr_name, {}).keys())

        if element_names:
            cropped_sdata = sdata.subset(element_names=element_names).query.bounding_box(
                axes=('y', 'x'),
                min_coordinate=[bbox['y_min'], bbox['x_min']],
                max_coordinate=[bbox['y_max'], bbox['x_max']],
                target_coordinate_system=c_system,
            )
        else:
            cropped_sdata = sd.SpatialData()

        if hasattr(sdata, 'points') and sdata.points and 'transcripts' in sdata.points:
            transform = get_transformation(sdata['transcripts'], to_coordinate_system=c_system)
            if transcripts_df is None:
                _tx = sdata['transcripts'].copy()
                if hasattr(_tx, 'npartitions') and _tx.npartitions > 1:
                    _tx = _tx.reset_index(drop=True)
                transcripts_df = _tx.compute() if hasattr(_tx, 'compute') else _tx

            scale_tf = extract_scale_transform(transform)
            if scale_tf is not None:
                axes_idx = {ax: i for i, ax in enumerate(scale_tf.axes)}
                x_global = transcripts_df['x'] * scale_tf.scale[axes_idx['x']]
                y_global = transcripts_df['y'] * scale_tf.scale[axes_idx['y']]
            else:
                x_global = transcripts_df['x']
                y_global = transcripts_df['y']

            mask = (
                (y_global >= bbox['y_min']) & (y_global <= bbox['y_max']) &
                (x_global >= bbox['x_min']) & (x_global <= bbox['x_max'])
            )
            filtered_transcripts = transcripts_df[mask]
            if len(filtered_transcripts) > 0:
                filtered_transcripts = filtered_transcripts.reset_index(drop=True)
                cropped_sdata.points['transcripts'] = sd.models.PointsModel.parse(filtered_transcripts)
    else:
        cropped_sdata = sdata.query.bounding_box(
            axes=('y', 'x'),
            min_coordinate=[bbox['y_min'], bbox['x_min']],
            max_coordinate=[bbox['y_max'], bbox['x_max']],
            target_coordinate_system=c_system,
        )
    return cropped_sdata

##############################################
#### SectionDivider — multi-section splits ###
##############################################

class SectionDivider:
    """Crops a multi-section slide SpatialData into individual per-section zarrs.

    Usage
    -----
    divider = SectionDivider(sdata, bundle_path, save_folder, num_workers=8)
    divider.run(sections)
    """

    def __init__(self, sdata, bundle_path, save_folder, num_workers=8, n_section_workers=1):
        self.sdata = sdata
        self.bundle_path = Path(bundle_path)
        self.save_folder = Path(save_folder)
        self.num_workers = num_workers
        self.n_section_workers = n_section_workers
        self._dapi_image = None
        self._transcripts_df_cache = None

    @property
    def dapi_image(self):
        if self._dapi_image is None:
            ome_tif_path = self.bundle_path / 'morphology.ome.tif'
            self._dapi_image = get_dapi_zstack(ome_tif_path, self.sdata)
        return self._dapi_image

    @property
    def _transcripts_df(self):
        """Read and cache the full transcript DataFrame once for the whole bundle.

        Multi-section bundles previously re-read the parquet file once per section.
        Caching here means the read happens exactly once regardless of section count.
        """
        if self._transcripts_df_cache is None:
            tx = self.sdata['transcripts']
            if hasattr(tx, 'npartitions') and tx.npartitions > 1:
                tx = tx.reset_index(drop=True)
            print("  Loading transcripts into memory (cached for all sections)...")
            self._transcripts_df_cache = tx.compute() if hasattr(tx, 'compute') else tx.copy()
        return self._transcripts_df_cache

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scale_bbox(bbox, transform):
        """Return bbox origin scaled from pixels to the element's coordinate system."""
        scale_tf = extract_scale_transform(transform)
        if scale_tf is None:
            print("No scale transform found, using original bbox.")
            return bbox
        axes_idx = {ax: i for i, ax in enumerate(scale_tf.axes)}
        return {
            'x_min': bbox['x_min'] / scale_tf.scale[axes_idx['x']],
            'y_min': bbox['y_min'] / scale_tf.scale[axes_idx['y']],
        }

    @staticmethod
    def _rebuild_multiscale(element, model_type, transforms):
        """Rebuild a multiscale DataTree, applying clean transformations per level."""
        data_tree_obj = xr.DataTree()
        for scale_level in element.keys():
            level_image = element[scale_level].image

            if transforms[scale_level] is None:
                current_tf = get_transformation(level_image, to_coordinate_system='global')
                new_tf = extract_scale_transform(current_tf) or Identity()
            else:
                new_tf = transforms[scale_level]

            remove_transformation(level_image, to_coordinate_system='global')
            set_transformation(level_image, new_tf, to_coordinate_system='global')

            if model_type == 'label':
                parsed = Labels2DModel.parse(level_image, dims=list(level_image.dims), chunks='auto')
            else:
                spatial_dims = sum(d in ('y', 'x', 'z') for d in level_image.dims)
                parse_cls = Image2DModel if spatial_dims == 2 else Image3DModel
                parsed = parse_cls.parse(
                    level_image, dims=list(level_image.dims),
                    c_coords=level_image['c'].values, chunks='auto',
                )
            data_tree_obj[scale_level] = xr.Dataset({'image': parsed})
        return data_tree_obj

    def _reset_raster(self, cropped_el, ref_el, model_type):
        """Reset a multiscale raster by copying transforms from the reference element."""
        transforms = {
            scale: get_transformation(ref_el[scale].image, to_coordinate_system='global')
            for scale in ref_el.keys()
        }
        return self._rebuild_multiscale(cropped_el, model_type, transforms)

    def _reset_transcripts(self, transcripts, bbox, ref_transform):
        if hasattr(transcripts, 'npartitions') and transcripts.npartitions > 1:
            transcripts = transcripts.reset_index(drop=True)
        if hasattr(transcripts, 'compute'):
            transcripts = transcripts.compute()

        transcripts['y_slide'] = transcripts['y'].astype('float64')
        transcripts['x_slide'] = transcripts['x'].astype('float64')

        bbox_scaled = self._scale_bbox(bbox, ref_transform)
        transcripts['y'] = (transcripts['y'] - bbox_scaled['y_min']).astype('float64')
        transcripts['x'] = (transcripts['x'] - bbox_scaled['x_min']).astype('float64')

        for col, dtype in [('is_gene', 'str'), ('transcript_id', 'float64'),
                            ('overlaps_nucleus', 'float64'), ('codeword_index', 'float64')]:
            if col in transcripts.columns:
                transcripts[col] = transcripts[col].astype(dtype)

        parsed = sd.models.PointsModel.parse(transcripts)
        set_transformation(parsed, ref_transform, to_coordinate_system='global')
        return parsed

    def _reset_shapes(self, shapes_element, bbox, ref_transform):
        bbox_scaled = self._scale_bbox(bbox, ref_transform)
        shapes_element = shapes_element.copy()
        shapes_element.geometry = shapes_element.geometry.translate(
            xoff=-bbox_scaled['x_min'], yoff=-bbox_scaled['y_min']
        )
        parsed = sd.models.ShapesModel.parse(shapes_element)
        set_transformation(parsed, ref_transform, to_coordinate_system='global')
        return parsed

    def _reset_table(self, table, bbox, ref_transform):
        bbox_scaled = self._scale_bbox(bbox, ref_transform)
        table.obsm['spatial'][:, 0] -= bbox_scaled['x_min']
        table.obsm['spatial'][:, 1] -= bbox_scaled['y_min']
        if 'x' in table.obs.columns:
            table.obs['x'] -= bbox_scaled['x_min']
        if 'y' in table.obs.columns:
            table.obs['y'] -= bbox_scaled['y_min']

        # Ensure all uns dict keys are strings (zarr compatibility)
        for key, value in table.uns.items():
            if isinstance(value, dict) and any(isinstance(k, int) for k in value):
                table.uns[key] = {str(k): v for k, v in value.items()}

        return TableModel.parse(adata=table)

    # ------------------------------------------------------------------
    # Section assembly
    # ------------------------------------------------------------------

    def _build_section_sdata(self, cropped_sdata, cropped_dapi, section):
        """Assemble a section SpatialData with origin-relative coordinates."""
        sdata = self.sdata
        bbox = cropped_sdata['table'].uns['sections_bboxes'][str(section)]
        tx_transform = get_transformation(sdata['transcripts'])

        return sd.SpatialData(
            images={
                'morphology_focus': self._reset_raster(
                    cropped_sdata['morphology_focus'], sdata['morphology_focus'], 'image'),
                'dapi_zstack': self._reset_raster(
                    cropped_dapi, self.dapi_image, 'image'),
            },
            labels={
                'cell_labels': self._reset_raster(
                    cropped_sdata['cell_labels'], sdata['cell_labels'], 'label'),
                'nucleus_labels': self._reset_raster(
                    cropped_sdata['nucleus_labels'], sdata['nucleus_labels'], 'label'),
            },
            shapes={
                'cell_boundaries': self._reset_shapes(
                    cropped_sdata['cell_boundaries'], bbox,
                    get_transformation(sdata['cell_boundaries'])),
                'nucleus_boundaries': self._reset_shapes(
                    cropped_sdata['nucleus_boundaries'], bbox,
                    get_transformation(sdata['nucleus_boundaries'])),
                'cell_circles': self._reset_shapes(
                    cropped_sdata['cell_circles'], bbox,
                    get_transformation(sdata['cell_circles'])),
            },
            points={
                'transcripts': self._reset_transcripts(
                    cropped_sdata['transcripts'], bbox, tx_transform),
            },
            tables={
                'table': self._reset_table(
                    cropped_sdata['table'], bbox, tx_transform),
            },
        )

    def _annotate_section_metadata(self, section_sdata, sec):
        uns = section_sdata['table'].uns
        uns['section'] = sec
        uns['fov_metadata'] = self.sdata['table'].uns['fov_metadata'].loc[
            self.sdata['table'].uns['fov_metadata']['section'] == sec
        ]
        uns['sections_bboxes'] = self.sdata['table'].uns['sections_bboxes'][str(sec)]
        uns['full_section_shape'] = list(
            sd.get_pyramid_levels(self.sdata['morphology_focus'], n=0).shape
        )

    @staticmethod
    def _rename_image_channels(sdata):
        for el in list(sdata.images.keys()):
            rename_chans(sdata, el)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def _process_section(self, sec):
        section_save_path = self.save_folder / f"section_{sec}.zarr"
        if section_save_path.exists():
            try:
                sd.read_zarr(section_save_path)
                print(f"Section {sec} already exists at {section_save_path}. Skipping.")
                return
            except Exception as e:
                print(f"Corrupted section {sec}: {e}. Regenerating.")
                shutil.rmtree(section_save_path)

        section_bbox = self.sdata['table'].uns['sections_bboxes'][str(sec)]
        cropped_dapi = crop_dapi_image(self.dapi_image, section_bbox)
        cropped_sdata = crop_filter_sdata(self.sdata, section_bbox, transcripts_df=self._transcripts_df)

        section_sdata = self._build_section_sdata(cropped_sdata, cropped_dapi, sec)
        self._annotate_section_metadata(section_sdata, sec)
        self._rename_image_channels(section_sdata)
        add_micron_coord_sys(section_sdata)

        print(f"  -> {section_save_path}")
        write_sdata_elements(section_sdata, section_save_path, num_workers=self.num_workers)

    def run(self, sections):
        """Divide the slide into per-section zarrs. Handles single and multi-section bundles."""
        if len(sections) == 1:
            sec = sections[0]
            print("Single-section bundle detected, processing as one section...")
            self.sdata.images['dapi_zstack'] = self.dapi_image
            uns = self.sdata['table'].uns
            uns['section'] = sec
            uns['full_section_shape'] = list(
                sd.get_pyramid_levels(self.sdata['morphology_focus'], n=0).shape
            )
            if uns.get('sections_bboxes') is None:
                shape = uns['full_section_shape']
                uns['sections_bboxes'] = {
                    sec: {'x_min': 0, 'x_max': shape[2], 'y_min': 0, 'y_max': shape[1]}
                }
            self._rename_image_channels(self.sdata)
            add_micron_coord_sys(self.sdata)
            section_save_path = self.save_folder / f"section_{sec}.zarr"
            write_sdata_elements(self.sdata, section_save_path, num_workers=self.num_workers)
        else:
            # Warm both caches before spawning threads — both are lazy and would
            # race to initialise if accessed for the first time from multiple threads.
            _ = self._transcripts_df
            _ = self.dapi_image
            if self.n_section_workers > 1:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with tqdm(total=len(sections), desc="Processing sections", unit="section") as pbar:
                    with ThreadPoolExecutor(max_workers=self.n_section_workers) as executor:
                        futures = {executor.submit(self._process_section, sec): sec for sec in sections}
                        for future in as_completed(futures):
                            future.result()  # re-raise any exception from a worker
                            pbar.update(1)
            else:
                with tqdm(total=len(sections), desc="Processing sections", unit="section") as pbar:
                    for sec in sections:
                        pbar.set_description(f"Section {sec}")
                        self._process_section(sec)
                        pbar.update(1)


def divide_section_sdata(sdata, sections, bundle_path, save_folder, num_workers=8, n_section_workers=1):
    """Convenience wrapper around SectionDivider.run()."""
    SectionDivider(sdata, bundle_path, save_folder, num_workers, n_section_workers).run(sections)

#### Combined function to read bundle, process metadata, and write section zarrs
def process_xenium_bundle(section_n, bundle_path, tmp_folder, sections_tmp_folder, num_workers=8, plots_folder=None, n_section_workers=1):
    section_tmp_path = tmp_folder / f'section_{section_n}.zarr'
    sections = [int(s) for s in section_n.split('_')]  # ints so sorted() is numeric
    sections_save_paths = [sections_tmp_folder / f'section_{s}.zarr' for s in sections]

    print(f"\n{'='*60}")
    print(f"Section(s): {section_n}")
    print(f"  src : {bundle_path}")
    print(f"  tmp : {section_tmp_path}")
    print(f"  out : {[p.name for p in sections_save_paths]}")
    print(f"{'='*60}")

    if check_existing_sdata_paths(sections_save_paths):
        print("  [skip] All sections already processed.")
        if plots_folder is not None:
            plot_save_path = plots_folder / f'section_{section_n}_bboxes.png'
            if not plot_save_path.exists():
                print("  [skip] Plotting section bounding boxes...")
                # Prefer the bundle tmp zarr; fall back to first section zarr if it was cleaned up
                sdata_for_plot = sd.read_zarr(
                    section_tmp_path if section_tmp_path.exists() else sections_save_paths[0]
                )
                sdata_for_plot = process_metadata(sdata_for_plot, bundle_path, sections)
                plot_section_bboxes(
                    sdata_for_plot, sdata_for_plot['table'].uns['sections_bboxes'],
                    fov_df=sdata_for_plot['table'].uns['fov_metadata'],
                    show_fovs=True, save_path=plot_save_path,
                )
        return

    print("  [1/3] Reading bundle and generating SpatialData...")
    save_section_sdata(bundle_path=bundle_path, save_path=section_tmp_path, n_jobs=num_workers)

    if not section_tmp_path.exists():
        raise RuntimeError(
            f"Failed to generate SpatialData for bundle '{section_n}'. "
            f"The temporary zarr was not created at {section_tmp_path}. "
            "Check for disk space issues or other errors above."
        )

    print("  [2/3] Processing metadata and dividing sections...")
    bundle_path = Path(bundle_path)
    sdata = sd.read_zarr(section_tmp_path)
    sdata = fix_points_category(sdata)
    sdata = process_metadata(sdata, bundle_path, sections)
    divide_section_sdata(sdata, sections, bundle_path, sections_tmp_folder, num_workers=num_workers, n_section_workers=n_section_workers)

    # Delete the bundle tmp zarr now that all sections have been written.
    # This prevents tmp zarrs from accumulating across all bundles and filling the disk.
    if section_tmp_path.exists():
        shutil.rmtree(section_tmp_path)

    if plots_folder is not None:
        print("  [3/3] Plotting section bounding boxes...")
        plot_save_path = plots_folder / f'section_{section_n}_bboxes.png'
        if not plot_save_path.exists():
            plot_section_bboxes(
                sdata, sdata['table'].uns['sections_bboxes'],
                fov_df=sdata['table'].uns['fov_metadata'],
                show_fovs=True, save_path=plot_save_path,
            )
    print(f"  Done: {section_n}")