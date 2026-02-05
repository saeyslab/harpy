# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## 0.3.0 - 2026-02-05

### Added

- Scalable extraction of single cell instances from multiplex images, and integration with PyTorch (PRs [#178](https://github.com/saeyslab/harpy/pull/178), [#180](https://github.com/saeyslab/harpy/pull/180), [#181](https://github.com/saeyslab/harpy/pull/181)).
- Deep feature extraction (PRs [#178](https://github.com/saeyslab/harpy/pull/178), [#180](https://github.com/saeyslab/harpy/pull/180), [#181](https://github.com/saeyslab/harpy/pull/181)).
- Chunk-wise streaming for integration in PyTorch (PR [#181](https://github.com/saeyslab/harpy/pull/181)).
- ViT-MAE deep feature extraction utilities for instance-level representation learning (PR [#186](https://github.com/saeyslab/harpy/pull/186)).
- Expanded MERSCOPE IO and plotting support (PRs [#166](https://github.com/saeyslab/harpy/pull/166), [#173](https://github.com/saeyslab/harpy/pull/173), [#174](https://github.com/saeyslab/harpy/pull/174)).
- Expanded MACSima IO support (PR [#167](https://github.com/saeyslab/harpy/pull/167)).
- Expanded Xenium IO support (PR [#169](https://github.com/saeyslab/harpy/pull/169)).
- Regionprops utilities (PR [#171](https://github.com/saeyslab/harpy/pull/171)).
- CUDA cluster support for Cellpose (PR [#155](https://github.com/saeyslab/harpy/pull/155)).
- Pooch env-var handling for datasets (PR [#150](https://github.com/saeyslab/harpy/pull/150)).

### Changed

- Improved shallow feature extraction and scaling and support for cupy (PR [#180](https://github.com/saeyslab/harpy/pull/180)).
- Cluster-intensity visualization and plotting (PR [#163](https://github.com/saeyslab/harpy/pull/163)).
- Standardized keys across IO/plot/table/shape modules (PR [#170](https://github.com/saeyslab/harpy/pull/170)).
- Simplified the aggregate/featurization paths (PR [#177](https://github.com/saeyslab/harpy/pull/177)).
- Removed the `datasets` dependency and updated the registry/benchmarks accordingly (PR [#148](https://github.com/saeyslab/harpy/pull/148)).

### Fixed

- Preserved point metadata through Dask operations; fixed MACSima IO naming/docs (PR [#187](https://github.com/saeyslab/harpy/pull/187)).
- Fixed MERSCOPE transcript IO bug (PR [#149](https://github.com/saeyslab/harpy/pull/149)).
- Fixed SpatialData chunk creation bug (PR [#152](https://github.com/saeyslab/harpy/pull/152)).

### Documentation

- Added documentation figures (PR [#185](https://github.com/saeyslab/harpy/pull/185)).
- Added/updated benchmark and technical-specs notebooks (PRs [#182](https://github.com/saeyslab/harpy/pull/182), [#183](https://github.com/saeyslab/harpy/pull/183), [#184](https://github.com/saeyslab/harpy/pull/184)).
- Updated installation instructions; removed `nptyping` (PR [#145](https://github.com/saeyslab/harpy/pull/145)).
- General docs refresh (PRs [#153](https://github.com/saeyslab/harpy/pull/153), [#165](https://github.com/saeyslab/harpy/pull/165)).

<details>
<summary>Merged pull requests</summary>

- PR [#187](https://github.com/saeyslab/harpy/pull/187) - 2026-02-05 — Fix MACSima IO naming/docs and preserve point metadata through Dask ops
- PR [#186](https://github.com/saeyslab/harpy/pull/186) — 2026-01-30 — Sc 88 xenium — Adds ViT-MAE featurization utilities/tests and Xenium-related updates, plus related pipeline/segmentation adjustments.
- PR [#185](https://github.com/saeyslab/harpy/pull/185) — 2026-01-20 — docs — Adds extract_instances documentation figures.
- PR [#184](https://github.com/saeyslab/harpy/pull/184) — 2026-01-20 — Sc 87 benchmark — Updates benchmark notebook.
- PR [#183](https://github.com/saeyslab/harpy/pull/183) — 2026-01-16 — Sc 87 notebook tech specs — Adds technical-specs notebook and docs index updates.
- PR [#182](https://github.com/saeyslab/harpy/pull/182) — 2026-01-15 — Benchmark notebook — Adds benchmark tutorial and dataset registry updates.
- PR [#181](https://github.com/saeyslab/harpy/pull/181) — 2026-01-14 — Sc 85 extract cupy — Adds Zarr iterable instances, CuPy-aware extraction path, and tests for large-scale instance handling.
- PR [#180](https://github.com/saeyslab/harpy/pull/180) — 2026-01-07 — Sc 84 extract — Refines extraction/aggregate utilities and related helpers.
- PR [#178](https://github.com/saeyslab/harpy/pull/178) — 2026-01-05 — Sc 83 extract — Extends instance extraction and featurization flow with updated neighbors/aggregate logic.
- PR [#177](https://github.com/saeyslab/harpy/pull/177) — 2025-12-16 — Sc 82 remove aggregate custom — Removes custom aggregate path and simplifies featurization/aggregate utilities.
- PR [#175](https://github.com/saeyslab/harpy/pull/175) — 2025-12-10 — Sc 82 aggregate — Updates aggregate utilities and tests; adjusts cluster-intensity plotting integration.
- PR [#174](https://github.com/saeyslab/harpy/pull/174) — 2025-12-08 — query points in plot sdata genes — Enables gene queries in `plot_sdata` and updates MERSCOPE IO/plot helpers.
- PR [#173](https://github.com/saeyslab/harpy/pull/173) — 2025-12-07 — Sc 80 merscope io — MERSCOPE IO adjustments and utility updates.
- PR [#171](https://github.com/saeyslab/harpy/pull/171) — 2025-12-03 — Sc 79 regionprops — Adds regionprops utilities and broad pipeline/plotting updates.
- PR [#170](https://github.com/saeyslab/harpy/pull/170) — 2025-12-01 — Sc 78 keys — Standardizes keys across IO/plot/table/shape modules and related tests.
- PR [#169](https://github.com/saeyslab/harpy/pull/169) — 2025-11-27 — Sc 77 xenium io — Adds Xenium IO support and transcriptomics dataset updates.
- PR [#167](https://github.com/saeyslab/harpy/pull/167) — 2025-11-24 — Sc 76 macsima io — Adds MACSIMA IO integration and dataset/plot updates.
- PR [#166](https://github.com/saeyslab/harpy/pull/166) — 2025-11-20 — Sc 75 merscope io — Adds MERSCOPE IO integration and plotting/pipeline updates.
- PR [#165](https://github.com/saeyslab/harpy/pull/165) — 2025-11-12 — Sc 76 docs — Docs updates (README/index).
- PR [#163](https://github.com/saeyslab/harpy/pull/163) — 2025-11-12 — Sc 74 plot intensity cluster ids — Adds cluster-intensity plotting and extensive pipeline/test coverage updates.
- PR [#162](https://github.com/saeyslab/harpy/pull/162) — 2025-10-28 — Sc 73 featurize — Adds/updates featurization APIs, utils, and tests.
- PR [#158](https://github.com/saeyslab/harpy/pull/158) — 2025-12-08 — pre-commit autoupdate — Tooling updates.
- PR [#157](https://github.com/saeyslab/harpy/pull/157) — 2025-09-30 — Sc 72 userwarnings — Improves user warnings and related tests.
- PR [#156](https://github.com/saeyslab/harpy/pull/156) — 2025-09-30 — pre-commit autoupdate — Tooling updates.
- PR [#155](https://github.com/saeyslab/harpy/pull/155) — 2025-09-25 — Sc 71 cuda cluster cellpose — Adds CUDA clustering support for Cellpose and updates docs.
- PR [#154](https://github.com/saeyslab/harpy/pull/154) — 2025-09-23 — pre-commit autoupdate — Tooling updates.
- PR [#153](https://github.com/saeyslab/harpy/pull/153) — 2025-09-15 — Sc 71 update docs — Docs refresh and notebook test updates.
- PR [#152](https://github.com/saeyslab/harpy/pull/152) — 2025-09-12 — bug create sdata chunks — Fixes SpatialData chunk creation bug.
- PR [#150](https://github.com/saeyslab/harpy/pull/150) — 2025-09-12 — Sc 69 pooch env variable — Adds Pooch env-var handling for datasets.
- PR [#149](https://github.com/saeyslab/harpy/pull/149) — 2025-09-11 — fix bug io transcripts merscope — Fixes MERSCOPE transcript IO bug.
- PR [#148](https://github.com/saeyslab/harpy/pull/148) — 2025-09-11 — SC67 remove datasets dependency — Removes datasets dependency and updates benchmarks/registry.
- PR [#147](https://github.com/saeyslab/harpy/pull/147) — 2025-09-09 — SC_66 preprocess — Preprocess updates and tests.
- PR [#146](https://github.com/saeyslab/harpy/pull/146) — 2025-09-05 — Sc 64 fix unit tests — Fixes segmentation/table unit tests.
- PR [#145](https://github.com/saeyslab/harpy/pull/145) — 2025-08-20 — update_installation — Updates installation instructions; removes `nptyping`.
- PR [#136](https://github.com/saeyslab/harpy/pull/136) — 2025-08-05 — development merge — Brings in development changes including SNR QC plot fixes and pre-commit cleanups.
- PR [#124](https://github.com/saeyslab/harpy/pull/124) — 2025-09-16 — pre-commit autoupdate — Tooling updates.

</details>
