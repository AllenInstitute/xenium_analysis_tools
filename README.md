# Xenium Analysis Tools

A Python library for processing and mapping Xenium sections using SpatialData formatting.

## Installation

### Code Ocean Package Manager
pip: git+[https://github.com/AllenInstitute/xenium_analysis_tools#egg=xenium-analysis-tools]


---

## Modules

### 1. `process_xenium`

Tools for processing raw Xenium outputs, managing SpatialData objects, and preparing data for downstream analysis.

#### Main processing functions:
* **`generate_dataset_slides`**: Generate the slide-level SpatialData objects from Xenium bundles for a Xenium dataset.
* **`process_dataset_slides`**: Generate the section-level SpatialData objects from slides for a Xenium dataset.
#### Task-specific functions:
* **`process_spatialdata`**: Core logic for manipulating and formatting Xenium `SpatialData` objects.
* **`divide_sections`**: Utilities for handling section boundaries and splitting data.
* **`validate_sections`**: Quality control checks to ensure section integrity before processing.

### 2. `map_xenium`

Functions for mapping cell types to Xenium data using reference taxonomies.

#### Main processing function:
* **`map_dataset_sections`**: Batch processing tools to apply mapping across multiple sections in a dataset.
* **`map_sections`**: Logic for mapping cell types on individual sections.

### 3. `utils`
Shared utility functions used across the library.
* **`io_utils`**: Standardized functions for loading and saving Xenium data structures.