# Xenium Analysis Tools

A Python library for processing and analyzing Xenium sections, utilizing SpatialData formatting.

## Installation

### In Code Ocean Package Manager:
pip: git+[https://github.com/AllenInstitute/xenium_analysis_tools#egg=xenium-analysis-tools]
* To get updated version of package, need to 'refresh' package image - usually removing and adding again works.

---
## params.json
Configurations for processing (folder/file names, MapMyCells parameters, options for functions, etc.)

## Modules
### 1. `process_xenium`

Tools for processing raw Xenium outputs, managing SpatialData objects, and preparing data for downstream analysis.

#### Main processing functions:
* **`generate_dataset_slides`**: Generate the slide-level SpatialData objects from Xenium bundles for a Xenium dataset.
* **`process_dataset_slides`**: Generate the section-level SpatialData objects from slides for a Xenium dataset.
#### Task-specific functions:
* **`process_spatialdata`**: Functions for processing/formatting Xenium `SpatialData` objects.
* **`divide_sections`**: Functions for dividing multi-section slides into individual sections w/independent coordinate systems.
* **`validate_sections`**: Quality control checks to ensure section processed correctly.

### 2. `map_xenium`

Tools for mapping cell types to Xenium data using reference taxonomies.

#### Main processing function:
* **`map_dataset_sections`**: Map multiple sections in a dataset using MapMyCells.
* **`map_sections`**: Functions for mapping cell types for an individual section.

### 3. `utils`
Shared utility functions used across the library.
* **`io_utils`**: Functions for loading data/organizing slides/sections, etc.

## CO Capsules
[generate_xenium_spatialdata](https://codeocean.allenneuraldynamics.org/capsule/8072328/tree)
* Runs generate_dataset_slides

[process_xenium_spatialdata](https://codeocean.allenneuraldynamics.org/capsule/4346497/tree)
* Runs process_dataset_slides

[map_xenium_types](https://codeocean.allenneuraldynamics.org/capsule/7531529/tree)
* Runs map_dataset_sections

[xenium_analysis_capsule](https://codeocean.allenneuraldynamics.org/capsule/7962049/tree) (currently updating...)
* Capsule with notebooks for examples of how to work with SpatialData objects, explanations of processing steps, plotting, etc.

## To-Dos:
[ ]: Finish updating xenium_analysis_capsule

[ ]: Consolidate & add Xenium data QC functions and make new capsule

[ ]: Consolidate & add mapping evaluation/QC functions
