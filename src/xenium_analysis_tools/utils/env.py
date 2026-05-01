"""Environment detection and package-data path helpers."""
import os
from importlib.resources import files as _files
from pathlib import Path


def detect_env(root_path=None, data_root=None, scratch_root=None, results_root=None, code_root=None):
    """Detect the runtime environment and return resolved root paths.

    Priority:
      1. Code Ocean capsule  — detected via CO_CAPSULE_ID env var or /root/capsule path
      2. Explicit root_path  — sub-paths derived from it (unless individually overridden)
      3. Fallback            — parent of cwd used as root_path

    Any individually supplied argument (data_root, scratch_root, etc.) always wins.
    """
    IS_CODE_OCEAN = (
        os.environ.get('CO_CAPSULE_ID') is not None
        or Path('/root/capsule').exists()
    )

    if root_path is None:
        root_path = Path('/root/capsule') if IS_CODE_OCEAN else Path.cwd().parent

    root_path = Path(root_path)
    return {
        'root_path':     root_path,
        'data_root':     Path(data_root)    if data_root    is not None else root_path / 'data',
        'scratch_root':  Path(scratch_root) if scratch_root is not None else root_path / 'scratch',
        'results_root':  Path(results_root) if results_root is not None else root_path / 'results',
        'code_root':     Path(code_root)    if code_root    is not None else root_path / 'code',
        'is_code_ocean': IS_CODE_OCEAN,
    }


def get_datasets_json_path():
    """Return a Path to the bundled xenium_datasets.json inside the package."""
    return _files("xenium_analysis_tools.data").joinpath("xenium_datasets.json")
