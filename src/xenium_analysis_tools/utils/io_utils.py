import shutil
from pathlib import Path
import spatialdata as sd
import json
import logging
import sys
import pandas as pd
from shutil import copytree, rmtree

def load_config(config_path=None):
    if config_path is not None:
        config_path = Path(config_path)
    else:
        config_path = '/code/params.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_logging(output_dir: Path):
    """
    Sets up logging to the specified directory.
    Returns (logger, log_file_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'pipeline_execution.log'
    
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    
    # Suppress noisy libraries
    logging.getLogger('ome_zarr').setLevel(logging.WARNING)
    logging.getLogger('tifffile').setLevel(logging.WARNING)
    logging.getLogger('xtiff').setLevel(logging.WARNING)

    return logging.getLogger(__name__), log_file

def get_sections_df(raw_data_folder):
    # Load reference table
    ref_path = list(raw_data_folder.glob('*_reference_table.csv'))[0]
    df = pd.read_csv(ref_path)
    
    # Add slide grouping
    df['slide_id'] = df.groupby('dir')['section'].transform(
        lambda x: "_".join(map(str, sorted(x.unique())))
    )
    df['slide_sections'] = df.groupby('dir')['section'].transform(
        lambda x: [sorted(x.unique())] * len(x)
    )
    return df

def is_complete_store(path_str, s3_fs=None):
    """
    Checks if a path contains valid Zarr v2/v3 metadata and required groups.
    Moved here from validate_sections.py to support manifest checks.
    """
    path = Path(path_str)
    
    if s3_fs is not None:
        try:
            # 1. Check for Zarr metadata
            has_v3 = s3_fs.exists(f"{path_str}/zarr.json")
            has_v2 = s3_fs.exists(f"{path_str}/.zgroup")
            if not (has_v3 or has_v2): return False

            # 2. Check for subgroups
            for group in ["images", "labels", "points", "shapes", "tables"]:
                if not s3_fs.exists(f"{path_str}/{group}"): return False
            return True
        except Exception:
            return False
    else:
        try:
            # 1. Check for Zarr metadata
            has_v3 = (path / "zarr.json").exists()
            has_v2 = (path / ".zgroup").exists()
            
            # If neither exists, it's not a valid zarr store
            if not (has_v3 or has_v2):
                return False

            # 2. Check for subgroups
            # Note: We return False immediately if a group is missing
            for group in ["images", "labels", "points", "shapes", "tables"]:
                if not (path / group).exists():
                    return False
            return True
        except Exception:
            return False

def is_complete(path: Path, check_store: bool = False) -> bool:
    """
    Checks if a path is complete. 
    Priority 1: Check for _SUCCESS flag (fastest, most atomic).
    Priority 2: Fallback to checking for valid Zarr metadata (slower).
    """
    # 1. Fast atomic check
    if (path / "_SUCCESS").exists():
        return True
    
    # 2. Check if it's a completed zarr, if requested
    if check_store:
        if path.is_dir() and is_complete_store(path):
            return True
        
    return False

def atomic_write_sdata(sdata: sd.SpatialData, path: Path, overwrite: bool = False):
    """
    Writes SpatialData and marks it as complete only on success.
    """
    if path.exists():
        if is_complete(path, check_store=True) and not overwrite:
            print(f"Skipping {path.name}: Already complete.")
            return
        shutil.rmtree(path)

    try:
        sdata.write(path)
        (path / "_SUCCESS").touch()
    except Exception as e:
        if path.exists():
            shutil.rmtree(path)
        raise e

def safe_copy_tree(src: Path, dst: Path):
    """Copy a folder and preserve the _SUCCESS flag."""
    if dst.exists():
        if is_complete(dst, check_store=True):
            return
        shutil.rmtree(dst)
    
    shutil.copytree(src, dst)

def get_partial_dataset(source_path, dest_path, pattern='section_*', subset_ids=None):
    """Copy slide data from source to destination, handling incomplete files."""
    # Find matches
    all_matches = list(source_path.glob(pattern))

    # Filter matches to only include sections in subset_ids
    if subset_ids is not None:
        matches = []
        for m in all_matches:
            section_ids = m.stem.split('_')[1:]
            if any(int(sid) in subset_ids for sid in section_ids):
                matches.append(m)
    else:
        matches = all_matches

    if not matches:
        print(f"No matches found in {source_path}")
        return
    
    # Create destination directory
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Copy slides
    for ma in matches:
        print(f"Checking {ma.name}...")
        dest_slide = dest_path / ma.name

        # Skip if destination already complete
        if dest_slide.exists() and is_complete_store(dest_slide):
            print(f"{ma.name} already complete")
            continue
        
        # Only copy if source is valid
        if not is_complete_store(ma):
            print(f"{ma.name} source incomplete, skipping")
            continue
        
        # Remove incomplete destination and copy
        if dest_slide.exists():
            rmtree(dest_slide)

        copytree(ma, dest_slide)
        print(f"Copied {ma.name}")