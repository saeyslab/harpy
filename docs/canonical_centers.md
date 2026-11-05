# Canonical-center interoperability

This page documents the contract for exchanging label-instance centers between
Harpy and downstream packages such as spatiato. It is intended for integration
authors. Most Harpy users obtain canonical centers through
`harpy.tb.aggregate_points`, or add them to an existing table that annotates
labels elements with `harpy.tb.add_canonical_centers`, and do not need the
validation API, types and constants of `harpy.table.canonical_centers`.

Canonical centers are stored as two AnnData components:

```text
adata.obsm["spatial_canonical"]
    dense float64 coordinates in fixed (z, y, x) order, one row per observation

adata.uns["spatial_coordinates"]["spatial_canonical"]
    the record: schema version, axes and, per labels element, the frame,
    algorithm version and source
```

They are all or nothing: a table has a finite center for every row, and a
record entry for every labels element it annotates, or it has neither
component.

The public `harpy.table.canonical_centers` package is the supported import
boundary. Integration code should not import its private `_models`, `_schema`
or `_calculation` modules.

## The record

Schema version 2, for a table that annotates one 2D labels element:

```python
{
    "schema_version": 2,
    "axes": ["z", "y", "x"],
    "regions": {
        "cells": {
            "source_element_type": "labels",
            "source_scale": "scale0",
            "coordinate_frame": {
                "type": "element_intrinsic",
                "pixel_coordinate_convention": "integer_indices_are_pixel_centers",
            },
            "algorithm_version": 1,
            "source": {"dims": ["y", "x"], "shape": [4000, 4000], "dtype": "uint32"},
        },
    },
}
```

- `schema_version`: `2`. A version-1 record, as written by Harpy 0.5, fails
  validation; rebuild it with
  `hp.tb.add_canonical_centers(sdata, table_name=..., overwrite=True)`.
- `axes`: the order of the matrix columns, always `["z", "y", "x"]`.
- `regions`: one entry per labels element, under its name. It may also hold
  entries for labels elements the table no longer annotates; see
  [Subsetting and concatenating tables](#subsetting-and-concatenating-tables).
  Each entry has:
  - `source_element_type`: `"labels"`.
  - `source_scale`: `"scale0"`: the centers are computed at full resolution.
  - `coordinate_frame.type`: `"element_intrinsic"`: the centers are pixel
    coordinates of the labels element's own `scale0` array, not coordinates
    in a SpatialData coordinate system. Different labels elements need not
    share a frame.
  - `coordinate_frame.pixel_coordinate_convention`:
    `"integer_indices_are_pixel_centers"`: the pixel at index `(i, j)` has its
    center at `(i, j)`. SpatialData places pixel centers at half-integers
    (`get_centroids` gives `(0.5, 0.5)` for a label on pixel `(0, 0)`), so add
    `0.5` to `y` and `x`, and to `z` for 3D labels, before mapping a center
    through the labels element's transformation.
  - `algorithm_version`: how the centers were computed; see
    [Algorithm version 1](#algorithm-version-1).
  - `source`: the `dims`, `shape` and `dtype` of the labels element's `scale0`
    array. 2D labels have dims `["y", "x"]` and `z = 0` in the matrix.

Every field except `algorithm_version` and `source` is a constant that the
parser requires, so a change of meaning needs a new schema version.

## Algorithm version 1

The center of a row is the center of mass of the pixels of its instance in the
labels element's `scale0` array, with every pixel weighted equally. Background
`0` is never an instance. For 2D labels, `z = 0`. A row whose instance has no
pixels in its labels element has no center, and writing the centers raises.

## Validation

`validate_canonical_centers(sdata, table_name=...)` returns the metadata of
the labels elements the table annotates, returns `None` when the table has no
canonical centers, and raises otherwise. It checks:

- both components, or neither, and a SpatialData annotation;
- the matrix: dense, shape `(n_obs, 3)`, `float64`, finite;
- the record: it parses strictly, in schema version 2, with an entry for every
  annotated labels element;
- per annotated labels element: the algorithm version is supported, `z = 0`
  for 2D labels, and the stored `dims`, `shape` and `dtype` equal those of
  the labels element's `scale0` array.

It reads no pixels. The last check detects a labels element replaced by one of
another shape, dimensions or dtype, such as another sample's labels under the
same name. It does not detect a re-segmentation of the same image with the same
shape and dtype: rebuild the centers after re-segmenting. `hp.tb.validate_table`
runs the same checks after its other table checks.

No check ties a center to its row. Keeping `obs` and `obsm` aligned is
AnnData's responsibility: replacing `adata.obs` with a reordered copy, without
reordering `adata.obsm`, goes undetected.

## Subsetting and concatenating tables

Canonical centers stay valid when a table is filtered, for example by per-cell
quality control before tables are concatenated, so downstream packages need not
rebuild them:

- A row subset or reorder keeps them valid. AnnData subsets `obs` and
  `obsm["spatial_canonical"]` together.
- A subset that drops a labels element entirely must also drop it from the
  annotation, as SpatialData requires; otherwise neither SpatialData nor
  `hp.tb.validate_table` accepts the table:

  ```python
  subset = adata[adata.obs["region"] == "cells_a"].copy()
  subset.uns["spatialdata_attrs"]["region"] = ["cells_a"]
  ```

- The record may keep the dropped labels element's entry. Validation ignores
  entries of labels elements the table does not annotate.

## Writing

- `hp.tb.aggregate_points` writes the centers together with the table.
- `hp.tb.add_canonical_centers(sdata, table_name=..., overwrite=...)` adds
  them to an existing table that annotates labels elements, in memory and,
  when `sdata` is backed, in its store. It computes the centers for every
  labels element the table annotates. It does not validate existing canonical
  centers, so with `overwrite=True` it also replaces stale or invalid ones.

## API

The functions, types and constants are listed in the
[API reference](api.md#canonical-centers).
