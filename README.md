```markdown
# Xenium Analysis Tools

A Python library for processing and mapping Xenium spatial data, developed by the Allen Institute for Neural Dynamics.

## Installation

### Code Ocean Package Manager (Recommended)
This library can be installed directly via the Code Ocean environment manager.

1. Open your Capsule.
2. Go to the **Environment** tab.
3. In the **Pip** section, click **Add**.
4. Paste the following link:
   ```text
   git+[https://github.com/AllenInstitute/xenium_analysis_tools#egg=xenium-analysis-tools](https://github.com/AllenInstitute/xenium_analysis_tools#egg=xenium-analysis-tools)

```

5. Click **Launch Cloud Workstation** to build.

### Local Installation

To install locally or in a standard terminal:

```bash
pip install git+[https://github.com/AllenInstitute/xenium_analysis_tools.git](https://github.com/AllenInstitute/xenium_analysis_tools.git)

```

---

## Modules

The library is organized into three primary sub-packages designed to handle different stages of the Xenium analysis pipeline.

### 1. `process_xenium`

Tools for processing raw Xenium outputs, managing SpatialData objects, and preparing data for downstream analysis.

* **`process_dataset_slides`**: Main workflow for processing slides across an entire dataset.
* **`process_spatialdata`**: Core logic for manipulating and formatting Xenium `SpatialData` objects.
* **`divide_sections`**: Utilities for handling section boundaries and splitting data.
* **`validate_sections`**: Quality control checks to ensure section integrity before processing.
* **`generate_dataset_slides`**: Helper functions for creating slide-level representations.

### 2. `map_xenium`

Functions for mapping cell types to Xenium data using reference taxonomies.

* **`map_sections`**: Logic for mapping cell types on individual tissue sections.
* **`map_dataset_sections`**: Batch processing tools to apply mapping across multiple sections in a dataset.

### 3. `utils`

Shared utility functions used across the library.

* **`io_utils`**: Standardized functions for loading and saving Xenium data structures.

---

## Usage

Import the specific modules you need for your analysis workflow.

**Example: Processing a Dataset**

```python
from xenium_analysis_tools.process_xenium import process_dataset_slides
from xenium_analysis_tools.utils import io_utils

# Load your configuration or data path
data_path = "/path/to/xenium/data"

# Run the processing pipeline
process_dataset_slides.run(data_path)

```

**Example: Mapping Sections**

```python
from xenium_analysis_tools.map_xenium import map_dataset_sections

# Run cell type mapping on processed sections
map_dataset_sections.run_mapping(
    processed_data_path="/path/to/processed/data",
    taxonomy_ref="/path/to/taxonomy"
)

```

---

## Development

### Updating the Package

1. Make changes to the code in the `src/` directory.
2. Bump the version in `src/xenium_analysis_tools/__init__.py`.
3. Commit and push to GitHub.
4. Create and push a new tag matching the version (e.g., `v0.1.1`).

### Running Tests

This project uses `pytest`. Run the following in the root directory:

```bash
pytest tests/

```