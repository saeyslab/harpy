# CosMx metadata: one record per mosaic, keyed by its coordinate system

Spec for [#293](https://github.com/saeyslab/harpy/issues/293), started 2026-11-24.

This is a working document: it replaces the issue as the spec while #293 is implemented, and tracks
progress with the checkboxes below. Update it when a decision is taken or a task is done. Delete it in
the last commit of the PR; the docstrings and
[`docs/development/metadata.md`](../../docs/development/metadata.md) then describe the result.

Links to the code as it was before the change are permalinks at 00df4ef. Links in the tasks are
relative paths without line numbers, as the code moves while it changes.

## Decisions

- [x] All of #293 is implemented in one PR. Nothing has to keep working in between.
- [x] No backward compatibility. Stores written before must be written again with the reader. No
      migration, no detection of the old layout, no fallback reads, no deprecation period.
- [x] `metadata_version` stays 1.
- [x] The registry is `sdata.attrs["harpy"]["coordinate_systems"]`, keyed by the name of a coordinate
      system. A record describes its coordinate system, whatever its unit; `microns_per_unit` gives that
      unit. The CosMx reader registers one coordinate system per mosaic, in full-resolution pixels.
- [x] `fov_bounds_px` becomes `fov_bounds`, in units of the coordinate system. `source_origin_px` and
      `adjacency_tolerance_px` are in pixels of the CosMx source and keep their names.
- [x] The reader writes no micron coordinate systems.
- [x] Harpy writes no records on derived elements: a derived element lies in its source's coordinate
      systems.
- [x] #276 lands in this PR: the FOV column of `assign_cosmx_fovs` becomes a categorical, with the
      same convention as the mosaic column.
- [x] Two public functions, at the top level: `hp.get_coordinate_system_metadata` and
      `hp.get_element_coordinate_system`, as SpatialData has `sd.get_element_annotators`. They live
      in a new package `src/harpy/metadata/`, which replaces `src/harpy/_metadata.py` as the single
      home of Harpy's metadata code. No public key constants: the documented columns of the DataFrame
      are the contract, and the functions supersede #292.
- [x] The expression class is not recorded in the feature panel; the former section 5 of #293 is
      dropped. It stays a choice of the analysis, made when calling `hp.tb.aggregate_points`, as the
      docstrings of `hp.io.cosmx`, `hp.pt.add_feature_panel` and `aggregate_points` document now.
      Feature panels and `aggregate_points` are unchanged.
- [x] This document is deleted before merging.

## Open

None. Add questions that come up during implementation here.

## Background

A _mosaic_ is a group of adjacent FOVs that the reader stitches into one image. In practice, one
mosaic is one tissue section, and downstream analyses treat it as one biological sample: a pipeline
segments each mosaic, aggregates each mosaic into a table of its own, and runs QC per mosaic. For a
table or a labels element, it needs the `sample_id` and mosaic number, the pixel size (for areas in
µm²), and the mosaic's morphology image (for images in reports).

### What the reader records today

Every element that the reader writes gets a record in
`sdata.attrs["harpy"]["images" | "labels" | "points"][<element name>]`, built by
[`_common_element_record`](https://github.com/saeyslab/harpy/blob/00df4ef67e464f2c8ad43b6cf045683431b0e3f3/src/harpy/io/_cosmx/_metadata.py#L19-L43),
which repeats the complete description of its mosaic and run:

```python
{
    "sample_id": "s1",
    "mosaic": {"mode": "spatial_groups", "adjacency_tolerance_px": 85, "number": 1},
    "fovs": [1, 2, 3, ...],
    "fov_bounds_px": {"1": {"y0": ..., "x0": ..., "y1": ..., "x1": ...}, ...},
    "source_origin_px": {"x": 0, "y": 0},
    "orientation": {"flip_x": True, "flip_y": False},
    "pixel_size_um": 0.120280945,
    "acquisition_timestamp": "...",
    # images: "channels"; labels: "instance_id_encoding" or "categories"; points: "feature_panel"
}
```

Every mosaic is described once per element, and
[`_validate_mosaic_consistency`](https://github.com/saeyslab/harpy/blob/00df4ef67e464f2c8ad43b6cf045683431b0e3f3/src/harpy/io/_cosmx/_validation.py#L164)
checks that the copies agree. Each mosaic gets a pixel coordinate system (`Identity`) and a micron one
(`Scale` by the pixel size), named by
[`_names.py`](https://github.com/saeyslab/harpy/blob/00df4ef67e464f2c8ad43b6cf045683431b0e3f3/src/harpy/io/_cosmx/_names.py#L31-L39)
but not recorded. Derived labels, such as those of `hp.im.segment`, get no record.

### What downstream code has to do today

| Needed                                               | How it is reconstructed today                                                                                                        |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Mosaic of a table, of its rows, or of derived labels | table region → points element in `uns["point_aggregation"]["regions"]` → that points element's record → `(sample_id, mosaic.number)` |
| Morphology image of a mosaic                         | scan all image records for the same `(sample_id, mosaic.number)`                                                                     |
| Pixel coordinate system                              | a config template that repeats the reader's base name, or a search for the coordinate system in which an element has an `Identity`   |
| Micron coordinate system                             | append `_micron`, importing the private `_micron_coordinate_system`                                                                  |
| Areas in µm²                                         | the determinant of the transformation to the micron coordinate system                                                                |
| The mosaics of a store                               | regroup all element records by `(sample_id, mosaic.number)`                                                                          |
| The records themselves                               | private constants, and the literals `"mosaic"` and `"number"` (#292)                                                                 |
| Expression class for `hp.tb.aggregate_points`        | a constant `"Endogenous"`                                                                                                            |

### What Harpy already provides

- Harpy's functions that derive an element copy all transformations of their source, with
  `get_transformation(..., get_all=True)`: `hp.im.segment` (also on a crop), `filter_labels`,
  `map_labels` (and `expand_labels`, `align_labels`, `merge_labels_nuclei`), `merge_labels`,
  `rasterize`, `hp.utils.bounding_box_query`, `flowsom`, normalization. A derived element lies in its
  source's coordinate systems.
- Tables record the coordinate system of each region, in `uns["point_aggregation"]["regions"]` and in
  the `regions` of `uns["feature_matrices"]`.
- `hp.qc.summarize_points` takes `microns_per_unit`, the micrometers per unit of
  `to_coordinate_system`.

## Target layout

```python
sdata.attrs["harpy"]["coordinate_systems"]["s1_global_1"] = {
    "sample_id": "s1",
    "mosaic": 1,
    "microns_per_unit": 0.120280945,
    "fovs": [1, 2, 3, ...],
    "fov_bounds": {"1": {"y0": ..., "x0": ..., "y1": ..., "x1": ...}, ...},
    "source_origin_px": {"x": 0, "y": 0},
    "orientation": {"flip_x": True, "flip_y": False},
    "mosaic_grouping": {"mode": "spatial_groups", "adjacency_tolerance_px": 85},
    "acquisition_timestamp": "...",  # optional
}
sdata.attrs["harpy"]["images"]["s1_morphology_image_mosaic_1"] = {"channels": [...]}
sdata.attrs["harpy"]["labels"]["s1_instance_labels_mosaic_1"] = {"instance_id_encoding": {...}}
sdata.attrs["harpy"]["labels"]["s1_compartment_labels_mosaic_1"] = {"categories": {...}}
sdata.attrs["harpy"]["points"]["s1_transcripts_mosaic_1"] = {"feature_panel": "feature_panel_..."}
```

`sdata.attrs["harpy"]["feature_panels"]` is unchanged.

The mosaic of any element, read or derived, is the registered coordinate system it has a
transformation to. The facts of the run (`microns_per_unit`, `orientation`, `mosaic_grouping`,
`acquisition_timestamp`) are repeated in each coordinate system of a sample; a `samples` level for
them is out of scope.

## 1. A registry of the reader's coordinate systems

Files: [`harpy/_metadata.py`](../../src/harpy/_metadata.py) (moves to
`harpy/metadata/_registry.py` in [section 4](#4-public-functions-to-read-the-registry)),
[`_cosmx/_metadata.py`](../../src/harpy/io/_cosmx/_metadata.py),
[`_constants.py`](../../src/harpy/io/_cosmx/_constants.py),
[`_images.py`](../../src/harpy/io/_cosmx/_images.py),
[`_labels.py`](../../src/harpy/io/_cosmx/_labels.py),
[`_transcripts.py`](../../src/harpy/io/_cosmx/_transcripts.py),
[`_reader.py`](../../src/harpy/io/_cosmx/_reader.py),
[`_validation.py`](../../src/harpy/io/_cosmx/_validation.py),
[`qc/points/`](../../src/harpy/qc/points/).

Records:

- [ ] Add `coordinate_systems` to `_METADATA_REGISTRIES`.
- [ ] Replace `_common_element_record` with a builder of the coordinate-system record of one mosaic,
      with the fields of the [target layout](#target-layout).
- [ ] Reduce element records to their own fields: images `channels`; labels `instance_id_encoding` or
      `categories`; points `feature_panel`. Without a feature panel, the reader writes no points record.
- [ ] Rename `_FOV_BOUNDS_KEY` to `"fov_bounds"`.

Writing:

- [ ] `_commit_element_metadata` writes the mosaic's coordinate-system record in the same attributes
      write as the element. It inserts the record if absent, and raises if a different one is present.
      Its element record becomes optional, for points without a feature panel.
- [ ] `add_cosmx_samples` checks new sample IDs against the `sample_id`s of the registry.

Validation, replacing `_validate_common_element_record`, `_validate_mosaic_consistency` and
`_MOSAIC_FIELDS`:

- [ ] At least one registered coordinate system, replacing the rule of at least one element record. A
      store without the registry, such as one written before this change, fails with a message to write
      it again with the current reader.
- [ ] The fields of each coordinate-system record, as the per-record checks do now.
- [ ] Each registered coordinate system exists in `sdata.coordinate_systems`.
- [ ] `(sample_id, mosaic)` is unique across the registry.
- [ ] The facts of the run agree across the coordinate systems of a sample.
- [ ] Each image and labels element with a record has an `Identity` to exactly one registered
      coordinate system and no transformation to another; `fov_bounds` lie within its raster.
- [ ] Points records: the feature-panel checks, as now. A points record needs no coordinate system, so
      the record that `hp.pt.add_feature_panel` writes for another points element of the store is valid.

Other readers of the moved fields:

- [ ] `hp.qc.summarize_points` takes `PointsSummaryMetadata.sample_id` from the registered coordinate
      system of the points element (`None` without one), no longer from the points record
      (`_points_reduction.py`). `summarize_points_by_feature` compares against the same source.

Documentation:

- [ ] The docstring of `hp.io.cosmx`: the tree, the note that elements are paired by `mosaic.number`,
      and the note that coordinate systems are not part of the root metadata.
- [ ] The `pixel_size_um` bullets in the docstrings of `_add_morphology_images`, the labels writers
      and `_add_transcript_points`.
- [ ] [`docs/development/metadata.md`](../../docs/development/metadata.md).

Acceptance:

- [ ] A store written by the reader holds one coordinate-system record per mosaic, and element records
      hold only their own fields.
- [ ] The validator passes on it, also after `hp.pt.add_feature_panel` registered another points
      element, and `add_cosmx_samples` accepts that store.
- [ ] The validator fails with a clear message on a store without the registry, on a registered
      coordinate system that no element has, on two records with the same `(sample_id, mosaic)`, and on a
      recorded image or labels element without an `Identity` to exactly one registered coordinate system.
- [ ] `hp.qc.summarize_points` reports the sample of CosMx points.

## 2. No micron coordinate systems

- [ ] `_images.py`, `_labels.py`, `_transcripts.py`: write each element only in its mosaic's pixel
      coordinate system, with an `Identity`.
- [ ] `_names.py`: remove `_micron_coordinate_system`, and the micron name from the module docstring.
- [ ] `_reader.py`: plan only the pixel coordinate system in `_validate_planned_names`; update the
      docstrings that mention micron coordinate systems.
- [ ] `microns_per_unit` replaces `pixel_size_um` in the records. The run manifest's internal
      `pixel_size_um` can stay.

Acceptance:

- [ ] The reader writes no `_micron` coordinate system, and `_micron_coordinate_system` is gone.

## 3. `assign_cosmx_fovs` reads the registry and writes the mosaic of each row

File: [`_fovs.py`](../../src/harpy/io/_cosmx/_fovs.py). Includes #276, the FOV column as a categorical.

- [ ] Find the registered coordinate system of each labels element of the table, with the resolver of
      [section 4](#4-public-functions-to-read-the-registry); a labels element in none, or in several,
      raises.
- [ ] Transform the canonical centers from the labels element into that coordinate system, and read
      the bounds, the sample and the mosaic from its record.
- [ ] Remove `_cosmx_image_of` and the requirement of a morphology image.
- [ ] Add `mosaic_key: str = "mosaic"`, next to `fov_key` and `sample_key`. Write the mosaic number of
      every row, also of rows between FOVs, as a categorical with integer categories sorted numerically:
      the mosaics of the table's regions.
- [ ] Apply the existing checks to `mosaic_key`: it must not replace the region or instance column,
      and the three keys must differ.
- [ ] Write the FOV column as a categorical instead of `Int64` (#276): integer categories sorted
      numerically, so that FOVs order as 1, 2, 10; every FOV of the matched mosaics as a category, from
      their `fov_bounds`, also FOVs without rows; a missing value for rows between FOVs or outside the
      mosaic.
- [ ] Update the docstring: the categorical FOV column, the new mosaic column, and the lookup through
      the registry instead of the image.
- [ ] Close #276 with the PR.

Acceptance:

- [ ] A table over two mosaics of one sample gets the right mosaic per row, as a categorical, also for
      labels segmented on a crop.
- [ ] Rows outside every FOV keep their sample and mosaic.
- [ ] A store read with `images=False` works.
- [ ] The FOV column is a categorical with every FOV of the matched mosaics as a category, in numeric
      order, including FOVs without rows; rows outside every FOV have a missing value.
- [ ] The FOV and mosaic columns keep their categories and missing values through a Zarr round trip.

## 4. Public functions to read the registry

Package: `src/harpy/metadata/`, the single home of Harpy's metadata code.

- [ ] Move the content of [`src/harpy/_metadata.py`](../../src/harpy/_metadata.py) (the registry
      keys, the version, `_metadata_registry`, `_harpy_metadata`) to `src/harpy/metadata/_registry.py`,
      remove `_metadata.py`, and update its importers.
- [ ] `src/harpy/metadata/_coordinate_systems.py`: a private resolver from an element to its
      coordinate system with a Harpy record, used by sections 1 and 3: the one it has a transformation
      to; `None` if none; an error naming them if several.
- [ ] `hp.get_element_coordinate_system(sdata, element_name) -> str | None`, on that resolver.
- [ ] `hp.get_coordinate_system_metadata(sdata) -> pandas.DataFrame`: one row per coordinate system
      with a Harpy record, indexed by its name, with the columns `sample_id`, `mosaic` and
      `microns_per_unit`, then the other fields of the record. The docstring documents the columns as
      the contract, and the CosMx fields as written by `hp.io.cosmx`.
- [ ] The package imports nothing from Harpy's other subpackages, only `spatialdata` and `pandas`, so
      that `harpy/__init__.py` can import it before them without circular imports.
- [ ] Export both functions from `harpy/__init__.py`, in `__all__`, and add them to the API
      documentation.
- [ ] Public docstrings say "coordinate systems with Harpy metadata", not "registered", which reads
      as image registration.
- [ ] Close #292: the functions supersede its constants.

Acceptance:

- [ ] Downstream code finds a table's mosaic, the mosaic's morphology image and its `microns_per_unit`
      with `hp.get_element_coordinate_system` and `hp.get_coordinate_system_metadata`, without private
      imports, naming conventions or transformation searches of its own.
- [ ] No module imports `harpy._metadata`.
- [ ] The functions are documented and tested, including an element in no and in two coordinate
      systems with Harpy metadata.

## Tests

- [ ] `src/harpy/_tests/test_io/test_cosmx/`: the modules that assert on the record layout, the
      micron coordinate systems or `fov_bounds_px`.
- [ ] `src/harpy/_tests/test_qc/test_points/test_summarize_points.py` and
      `test_summarize_points_by_feature.py`: the sample from the registry.
- [ ] Each acceptance criterion above has a test.

## Handover to downstream code

When all of the above is done, downstream code replaces its workarounds:

| Today                                                           | After #293                                                                                                                                                             |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mosaic of a table's rows through the points element's record    | `obs[sample_key]` and `obs[mosaic_key]` from `hp.io.assign_cosmx_fovs`                                                                                                 |
| Casting the FOV column to a categorical after assigning         | `obs[fov_key]` is already categorical                                                                                                                                  |
| Mosaic of a table or labels element                             | `hp.get_element_coordinate_system`, or the `coordinate_system` that `uns["point_aggregation"]` records per region; then its row of `hp.get_coordinate_system_metadata` |
| Morphology image of a mosaic                                    | the image with a record in `images` and a transformation to the mosaic's coordinate system                                                                             |
| Pixel coordinate system from a template or an `Identity` search | the index of `hp.get_coordinate_system_metadata`                                                                                                                       |
| Micron coordinate system, `_micron`                             | `microns_per_unit` of the record                                                                                                                                       |
| Areas in µm² from the micron transformation                     | area in pixels × determinant of the transformation to the coordinate system × `microns_per_unit`²                                                                      |
| Point summaries in µm                                           | pass the record's `microns_per_unit` to `hp.qc.summarize_points`                                                                                                       |
| Regrouping element records into mosaics                         | `hp.get_coordinate_system_metadata`                                                                                                                                    |
| Private constants and literals                                  | the two functions; the columns of the DataFrame are the contract                                                                                                       |
| `"Endogenous"` as a constant                                    | unchanged: it stays in the downstream configuration, passed as `expression_class=` to `hp.tb.aggregate_points`                                                         |

Stores written before this change must be written again with the reader.

## Limitations

- Coordinate-system keys go stale when a coordinate system is renamed with
  `SpatialData.rename_coordinate_systems`, or when its last element is deleted. The validator reports
  both.
- An element placed in two registered coordinate systems, for example to align two mosaics, is
  ambiguous, and the lookups raise rather than guess. To align mosaics, use a coordinate system of
  their own.

## Out of scope

- The names of the QC summary columns (#278).
- `uns["point_aggregation"]`: it records the coordinate system of each region, which reaches the
  registry.
- A helper that adds a micron coordinate system from `microns_per_unit`.
- The micron coordinate systems of the Xenium reader and the dataset loaders.
- A `samples` level for the facts of the run.
- The expression class in the feature panel, the former section 5 of #293. Which class goes into `X`
  is a choice of the analysis, not a fact of the assay, and the current design chooses it when calling
  `hp.tb.aggregate_points`. Recording it would change the content-hashed panel contract, and so every
  panel key, to save downstream code one string constant.

## Log

- 2026-11-24: spec written from #293, with the decisions above.
- 2026-11-24: decided to land #276 in this PR; tasks added to section 3.
- 2026-11-24: decided the public functions and their package; section 4 rewritten, public key
  constants dropped.
- 2026-11-24: decided `hp.get_expression_class` instead of an option of `aggregate_points`; section 5
  updated.
- 2026-11-24: decided the reader's behaviour for a plex without `Endogenous`; corrected the fix that
  `hp.get_expression_class` names for a panel without an expression class. No open items left.
- 2026-11-24: dropped section 5, the expression class in the feature panel, with the three decisions
  above about it; see Out of scope. #293 updated to match.
- 2026-11-24: #293 brought up to date with this spec: scope and compatibility, #276 in section 3, the
  two `hp.get_*` functions and their package in section 4, and the handover table.
