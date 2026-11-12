"""Concatenate Harpy tables along their rows, keeping Harpy's metadata valid."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from anndata.abc import CSCDataset, CSRDataset
from loguru import logger as log
from spatialdata.models import TableModel

from harpy._storage._anndata import _decode_anndata_element
from harpy.table._metadata import (
    _POINT_AGGREGATION_KEY,
    _comparable_metadata,
    _merged_metadata,
    _metadata_difference,
)
from harpy.table._validation import _validate_feature_matrices, _validate_table_annotation_without_sdata
from harpy.table.canonical_centers._models import CanonicalMetadata
from harpy.table.canonical_centers._schema import (
    CANONICAL_OBSM_KEY,
    SPATIAL_COORDINATES_KEY,
    _validate_canonical_centers_without_sources,
)
from harpy.table.io._write_by_region import _matrix_format
from harpy.utils._keys import _FEATURE_MATRICES_KEY

_UNS_MERGE_STRATEGIES = (None, "same", "unique", "first", "only")
_FORMAT_NAMES = {"dense": "dense", "csr": "CSR", "csc": "CSC", "DataFrame": "a DataFrame", None: "absent"}
_CANONICAL_MATRIX = f"obsm[{CANONICAL_OBSM_KEY!r}]"
_CANONICAL_METADATA = f"uns[{SPATIAL_COORDINATES_KEY!r}][{CANONICAL_OBSM_KEY!r}]"


@dataclass(frozen=True)
class _Table:
    """A table to concatenate, with the annotation and metadata its checks return."""

    name: str  # how errors name the table: tables['s1'] or tables[0]
    adata: AnnData
    region_key: str
    instance_key: str
    regions: tuple[str, ...]
    feature_matrices_metadata: Mapping[str, Mapping]  # keyed by obsm key
    canonical_metadata: CanonicalMetadata | None


def concat(
    tables: Sequence[AnnData] | Mapping[str, AnnData],
    *,
    table_key: str | None = None,
    uns_merge: Literal["same", "unique", "first", "only"] | None = None,
    feature_matrices_key: str = _FEATURE_MATRICES_KEY,
    drop_mismatched_feature_matrices: bool = False,
) -> AnnData:
    """Concatenate Harpy tables along their rows, keeping Harpy's metadata valid.

    Use it to combine the tables of several samples into one, for example after
    per-sample quality control. :func:`anndata.concat` drops ``uns`` by
    default, and :func:`spatialdata.concatenate` rebuilds only the SpatialData
    annotation. ``concat`` also merges Harpy's metadata in ``uns``: of the
    feature matrices, ``uns["feature_matrices"]``; of the canonical centers,
    ``uns["spatial_coordinates"]["spatial_canonical"]``; and of
    :func:`harpy.tb.aggregate_points`, ``uns["point_aggregation"]``. It also
    merges the colors of categorical ``obs`` columns.

    The tables must have the same structure. Otherwise ``concat`` raises, and
    the error says what differs and how to fix it:

    - every table is annotated, with the same region and instance columns, and
      no region is in two tables;
    - ``obs_names`` are unique across the tables;
    - the tables have the same ``obs`` columns, each with one dtype in every
      table. The categories of a categorical column may differ: the result
      holds them all;
    - the tables have the same features, ``var_names``, in the same order;
    - the tables have the same layers and ``obsm`` keys. Each of ``X``, a layer
      or an ``obsm`` matrix has one format in every table, dense, CSR or CSC,
      and is lazy in every table or in memory in every table. Backed matrices
      count as lazy: ``concat`` turns them into Dask arrays without reading
      them. It loads no matrix into memory and changes no format;
    - a feature matrix described in ``uns[feature_matrices_key]`` is
      described in every table, with the same metadata (see
      ``drop_mismatched_feature_matrices``);
    - canonical centers, added with :func:`harpy.tb.add_canonical_centers`,
      are in every table or in none;
    - the point-aggregation metadata is in every table or in none. If
      present, the tables were aggregated the same way: same points column,
      same mode (ordinary or class-aware).

    ``concat`` drops ``obsp``, ``varp``, ``varm`` and ``raw``, and the ``var``
    columns that differ between tables, with a warning: they describe one
    table alone. The inputs are not modified.

    Parameters
    ----------
    tables
        The tables to concatenate, in order: a list, or a mapping from a name
        to each table. The names, or the positions in a list, identify the
        tables in error messages, such as ``tables['table_a']`` or
        ``tables[0]``, and in the ``table_key`` column. To use the names of the
        tables in ``sdata``, pass
        ``{name: sdata.tables[name] for name in ["table_a", "table_b"]}``.
    table_key
        Name of a new ``obs`` column that records the table each row came
        from: its name in ``tables`` if that is a mapping, or its position as
        a string if it is a list: ``"0"``, ``"1"``, ... No column is added by
        default. It must not be an ``obs`` column of any table.
    uns_merge
        How :func:`anndata.concat` merges the ``uns`` entries that ``concat``
        does not merge itself. ``None``, the default as in AnnData, drops them;
        ``"same"`` keeps entries equal in every table; ``"unique"`` keeps
        entries on which the tables that hold them agree, such as
        ``uns["qc"][sample]``; ``"first"`` keeps each entry from the first
        table that has it; ``"only"`` keeps entries found in one table only.
        Nested entries are merged key by key. Dropped top-level entries are
        logged.
    feature_matrices_key
        Key in ``uns`` of the feature-matrix metadata, as in
        :func:`harpy.tb.validate_table`.
    drop_mismatched_feature_matrices
        By default, a feature matrix that is missing from a table, or whose
        metadata differs between tables, raises. With ``True``, such matrices
        and their metadata are dropped instead, with a log message. The
        metadata is compared whole, except its ``regions``, the metadata of
        each region, which are merged.

    Returns
    -------
    The concatenated table, lazy or in memory as its inputs are. Store it with
    :func:`harpy.tb.io.add_table`, then check it with
    :func:`harpy.tb.validate_table`.

    Raises
    ------
    TypeError
        If ``tables`` is not a sequence or mapping of AnnData tables.
    ValueError
        If the tables do not have the same structure, as listed above.

    Notes
    -----
    To remove whole regions from a table, for example a failed field of view
    before concatenating or one sample from the result, subset its rows and
    set its declared regions, ``uns["spatialdata_attrs"]["region"]``, to the
    regions that are left. Subsetting does not update them, and ``concat``
    and validation raise while they list a region without rows::

        subset = adata[adata.obs["region"] == "cells_a"].copy()
        subset.uns["spatialdata_attrs"]["region"] = ["cells_a"]

    or, when storing the subset, pass them to :func:`harpy.tb.io.add_table`
    instead::

        hp.tb.io.add_table(
            sdata,
            subset,
            output_table_name="table_a",
            region=["cells_a"],
            region_key="region",
            instance_key="cell_ID",
        )

    Removing cells within a region, such as per-cell quality control, needs
    no such step.

    A subset keeps the metadata in ``uns`` of the regions it removed, for
    example their entries in ``uns["point_aggregation"]["regions"]``.
    ``concat`` takes from each table only the metadata of its declared
    regions, and logs what it leaves out.

    The colors of a categorical ``obs`` column, ``uns[f"{column}_colors"]``,
    are merged by category name, not by ``uns_merge``. A category with
    different colors in different tables keeps the first table's color. If a
    category has no color in any table, the colors are dropped, and scanpy
    assigns its default colors.

    ``obs`` columns and ``obsm`` matrices computed per table, such as a
    ``leiden`` clustering, ``X_pca`` or ``X_umap``, are kept, and so are
    ``uns`` entries such as ``uns["neighbors"]`` if ``uns_merge`` keeps them.
    Their values do not compare across tables: recompute them after
    concatenating.

    Examples
    --------
    .. code-block:: python

        tables = {sample: sdata.tables[f"table_{sample}"] for sample in samples}
        adata = hp.tb.concat(tables, table_key="sample", uns_merge="unique")
        sdata = hp.tb.io.add_table(
            sdata,
            adata,
            output_table_name="table",
            region=adata.uns["spatialdata_attrs"]["region"],
        )
        hp.tb.validate_table(sdata, "table")
    """
    _check_parameters(table_key=table_key, uns_merge=uns_merge, feature_matrices_key=feature_matrices_key)
    keys, adatas = _normalize_tables(tables)
    parsed = [
        _parse_table(f"tables[{key!r}]", adata, feature_matrices_key=feature_matrices_key)
        for key, adata in zip(keys, adatas, strict=True)
    ]

    _check_annotation_keys(parsed)
    _check_regions(parsed)
    _check_obs_names(parsed)
    _check_obs(parsed, table_key=table_key)
    _check_var_names(parsed)
    region_order = [region for table in parsed for region in table.regions]
    canonical_metadata = _merged_canonical_metadata(parsed, region_order=region_order)
    point_aggregation_metadata = _merged_point_aggregation_metadata(parsed, region_order=region_order)
    mismatched_feature_matrices = _feature_matrices_with_mismatched_metadata(parsed)
    if mismatched_feature_matrices and not drop_mismatched_feature_matrices:
        raise ValueError(
            f"A feature matrix described in uns[{feature_matrices_key!r}] must be described in every table, with "
            f"the same metadata: {_describe(mismatched_feature_matrices)}. Compute these matrices the same way for "
            "every table, or pass drop_mismatched_feature_matrices=True to drop them and their metadata."
        )
    _check_matrices(parsed, dropped_feature_matrices=mismatched_feature_matrices)
    feature_matrices_metadata = _merged_feature_matrices_metadata(
        parsed, dropped_feature_matrices=mismatched_feature_matrices, region_order=region_order
    )

    if mismatched_feature_matrices:
        log.info(
            f"concat drops the feature matrices {list(mismatched_feature_matrices)} and their metadata, as "
            f"drop_mismatched_feature_matrices=True: {_describe(mismatched_feature_matrices)}."
        )
    _log_dropped_entries(parsed)
    _log_left_out_regions(
        parsed, feature_matrices_key=feature_matrices_key, dropped_feature_matrices=mismatched_feature_matrices
    )
    palette_columns = [
        column
        for column in parsed[0].adata.obs.columns
        if isinstance(parsed[0].adata.obs[column].dtype, pd.CategoricalDtype)
    ]
    owned_uns_keys = {
        TableModel.ATTRS_KEY,
        feature_matrices_key,
        _POINT_AGGREGATION_KEY,
        SPATIAL_COORDINATES_KEY,
        *(f"{column}_colors" for column in palette_columns),
    }
    prepared = [
        _prepared_table(table, owned_uns_keys=owned_uns_keys, dropped_feature_matrices=mismatched_feature_matrices)
        for table in parsed
    ]
    result = ad.concat(
        dict(zip(keys, prepared, strict=True)) if isinstance(tables, Mapping) else prepared,
        join="inner",
        merge="same",
        uns_merge=uns_merge,
        label=table_key,
    )
    _log_dropped_var_columns(parsed, result)
    _log_dropped_uns_entries(prepared, result, uns_merge=uns_merge)

    result.uns[TableModel.ATTRS_KEY] = {
        TableModel.REGION_KEY: region_order,
        TableModel.REGION_KEY_KEY: parsed[0].region_key,
        TableModel.INSTANCE_KEY: parsed[0].instance_key,
    }
    if feature_matrices_metadata:
        result.uns[feature_matrices_key] = feature_matrices_metadata
    if point_aggregation_metadata is not None:
        result.uns[_POINT_AGGREGATION_KEY] = point_aggregation_metadata
    if canonical_metadata is not None:
        result.uns[SPATIAL_COORDINATES_KEY] = {CANONICAL_OBSM_KEY: canonical_metadata}
    for column in palette_columns:
        _merge_palette(result, parsed, column=column)
    TableModel.validate(result)
    return result


def _check_parameters(*, table_key: object, uns_merge: object, feature_matrices_key: object) -> None:
    if table_key is not None and (not isinstance(table_key, str) or not table_key):
        raise ValueError(f"table_key must be None or a non-empty string, found {table_key!r}.")
    if uns_merge not in _UNS_MERGE_STRATEGIES:
        raise ValueError(f"uns_merge must be None, 'same', 'unique', 'first' or 'only', found {uns_merge!r}.")
    if not isinstance(feature_matrices_key, str) or not feature_matrices_key:
        raise ValueError(f"feature_matrices_key must be a non-empty string, found {feature_matrices_key!r}.")


def _normalize_tables(tables: object) -> tuple[list, list[AnnData]]:
    """Return the keys of the tables, their positions for a sequence, and the tables themselves."""
    if isinstance(tables, Mapping):
        keys, adatas = list(tables.keys()), list(tables.values())
    elif isinstance(tables, Sequence) and not isinstance(tables, (str, bytes)):
        keys, adatas = list(range(len(tables))), list(tables)
    else:
        raise TypeError(
            f"tables must be a sequence or a mapping of AnnData tables, found {type(tables).__name__}; pass "
            "[adata_1, adata_2] or {'s1': adata_1, 's2': adata_2}."
        )
    if not adatas:
        raise ValueError("tables must hold at least one table.")
    for key, adata in zip(keys, adatas, strict=True):
        if not isinstance(adata, AnnData):
            raise TypeError(
                f"tables[{key!r}] is a {type(adata).__name__}, not an AnnData; pass tables such as "
                "sdata.tables['table']."
            )
    return keys, adatas


def _parse_table(name: str, adata: AnnData, *, feature_matrices_key: str) -> _Table:
    """Run the checks of one table that need no other table and no SpatialData object."""
    annotation = _validate_table_annotation_without_sdata(adata, name=name, path=name)
    if annotation is None:
        raise ValueError(
            f"{name} has no SpatialData annotation, uns[{TableModel.ATTRS_KEY!r}], which concat needs to merge "
            "the metadata of each region. Annotate it before concatenating, for example with "
            "spatialdata.models.TableModel.parse(adata, region=..., region_key=..., instance_key=...)."
        )
    region_key, instance_key, regions = annotation
    # These checks are shared with validate_table and do not know the table: name it in their errors.
    try:
        feature_matrices_metadata = _validate_feature_matrices(adata, feature_matrices_key=feature_matrices_key)
        canonical_metadata = _validate_canonical_centers_without_sources(adata)
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from error
    for matrix, metadata in feature_matrices_metadata.items():
        if "regions" in metadata and not isinstance(metadata["regions"], Mapping):
            raise ValueError(
                f"{name}: uns[{feature_matrices_key!r}][{matrix!r}]['regions'] is a "
                f"{type(metadata['regions']).__name__}, not a mapping. Harpy keeps the metadata of each region there, "
                "keyed by labels element name; rename the field before concatenating."
            )
    return _Table(
        name=name,
        adata=adata,
        region_key=region_key,
        instance_key=instance_key,
        regions=regions,
        feature_matrices_metadata=feature_matrices_metadata,
        canonical_metadata=canonical_metadata,
    )


def _check_annotation_keys(tables: Sequence[_Table]) -> None:
    if len({(table.region_key, table.instance_key) for table in tables}) > 1:
        details = "; ".join(
            f"{table.name} region_key {table.region_key!r} and instance_key {table.instance_key!r}" for table in tables
        )
        raise ValueError(
            f"The tables name their region and instance columns differently: {details}. Rename the columns to the "
            f"same names in every table, and set uns[{TableModel.ATTRS_KEY!r}]['region_key'] and ['instance_key'] "
            "to match, before concatenating."
        )


def _check_regions(tables: Sequence[_Table]) -> None:
    holders: dict[str, list[str]] = {}
    for table in tables:
        for region in table.regions:
            holders.setdefault(region, []).append(table.name)
    shared = {region: names for region, names in holders.items() if len(names) > 1}
    if shared:
        details = "; ".join(f"{region!r} in {', '.join(names)}" for region, names in shared.items())
        raise ValueError(
            f"Each region must be in one table only, as each table holds its own metadata of its regions: {details}. "
            "Check that no table is passed twice, or keep the rows of each region in one table and update the "
            f"others' uns[{TableModel.ATTRS_KEY!r}]['region'], before concatenating."
        )


def _check_obs_names(tables: Sequence[_Table]) -> None:
    index = pd.Index(np.concatenate([table.adata.obs_names.to_numpy() for table in tables]))
    if index.is_unique:
        return
    duplicated = index.duplicated(keep=False)
    positions = np.repeat(np.arange(len(tables)), [table.adata.n_obs for table in tables])
    holders = [tables[position].name for position in np.unique(positions[duplicated])]
    raise ValueError(
        f"obs_names must be unique across the tables: {_preview(pd.unique(index[duplicated]))} occur more than "
        f"once, in {', '.join(holders)}; anndata.concat would keep the duplicates. Make them unique before "
        "concatenating, for example with adata.obs_names = adata.obs_names + '_s1', with one suffix per table."
    )


def _check_obs(tables: Sequence[_Table], *, table_key: str | None) -> None:
    _require_same_keys(
        "obs columns",
        [table.adata.obs.columns for table in tables],
        tables,
        fix="Add or delete these columns so that every table has the same ones before concatenating.",
    )
    for column in tables[0].adata.obs.columns:
        by_dtype = _group({table.name: _dtype_name(table.adata.obs[column].dtype) for table in tables})
        if len(by_dtype) > 1:
            (target, _), (_, others) = list(by_dtype.items())[:2]
            example = f"{others[0]}.obs[{column!r}]"
            raise ValueError(
                f"obs column {column!r} is {_describe_groups(by_dtype)}; cast it to one dtype in every table before "
                f"concatenating, for example with {example} = {example}.astype({target!r}). Categorical columns "
                "may differ in their categories."
            )
    if table_key is not None:
        holders = [table.name for table in tables if table_key in table.adata.obs.columns]
        if holders:
            raise ValueError(
                f"table_key {table_key!r} is an obs column of {', '.join(holders)}, which anndata.concat would "
                "overwrite. Pass another table_key, or rename the column."
            )


def _check_var_names(tables: Sequence[_Table]) -> None:
    first = tables[0]
    for table in tables[1:]:
        if table.adata.var_names.equals(first.adata.var_names):
            continue
        only_first = first.adata.var_names.difference(table.adata.var_names, sort=False)
        only_other = table.adata.var_names.difference(first.adata.var_names, sort=False)
        if only_first.empty and only_other.empty:
            raise ValueError(
                f"{table.name} has the features of {first.name} in another order; concat requires the same "
                f"var_names in the same order. Reorder them before concatenating, for example with "
                f"{table.name} = {table.name}[:, {first.name}.var_names]."
            )
        details = [
            f"{owner.name} has {_preview(names)} that {other.name} lacks"
            for owner, other, names in ((first, table, only_first), (table, first, only_other))
            if not names.empty
        ]
        raise ValueError(
            f"var_names differ: {' and '.join(details)}. concat requires the same features in every table, in the "
            "same order: an outer join would invent zero counts for features a table did not measure. "
            "hp.tb.aggregate_points in ordinary mode keeps only the features observed in the points it aggregates, "
            "so tables of different samples can differ; in class-aware mode it keeps every feature of the feature "
            "panel, observed or not. Aggregate in class-aware mode, after registering a panel with "
            "hp.pt.add_feature_panel if the points have none, or subset the tables to their shared features, for "
            "example with adata[:, shared].copy(), before concatenating."
        )


def _merged_canonical_metadata(tables: Sequence[_Table], *, region_order: Sequence[str]) -> dict | None:
    """Return the canonical-center metadata of the result, from the entries of each table's declared regions."""
    holders = [table.name for table in tables if table.canonical_metadata is not None]
    if not holders:
        return None
    if len(holders) != len(tables):
        lacking = [table.name for table in tables if table.canonical_metadata is None]
        raise ValueError(
            f"Canonical centers are in {', '.join(holders)} but not in {', '.join(lacking)}; every table or none "
            "must have them. Add them to every table with hp.tb.add_canonical_centers(sdata, table_name=...), or "
            f"delete {_CANONICAL_MATRIX} and {_CANONICAL_METADATA} from every table, before concatenating."
        )
    # No _metadata_difference check: _parse_table validated each table's canonical metadata, which allows
    # one value only for every field outside regions (schema_version 2, axes ["z", "y", "x"]).
    return _merged_metadata(
        [table.adata.uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY] for table in tables],
        declared_regions=[table.regions for table in tables],
        region_order=region_order,
    )


def _merged_point_aggregation_metadata(tables: Sequence[_Table], *, region_order: Sequence[str]) -> dict | None:
    """Return the point-aggregation metadata of the result, from the entries of each table's declared regions."""
    key = f"uns[{_POINT_AGGREGATION_KEY!r}]"
    per_table = [table.adata.uns.get(_POINT_AGGREGATION_KEY) for table in tables]
    holders = [table.name for table, metadata in zip(tables, per_table, strict=True) if metadata is not None]
    if not holders:
        return None
    if len(holders) != len(tables):
        lacking = [table.name for table, metadata in zip(tables, per_table, strict=True) if metadata is None]
        raise ValueError(
            f"{key} is in {', '.join(holders)} but not in {', '.join(lacking)}; every table or none must have it. "
            "Concatenate tables of hp.tb.aggregate_points with each other, or delete it from every table, for "
            f"example with del adata.uns[{_POINT_AGGREGATION_KEY!r}], before concatenating."
        )
    for table, metadata in zip(tables, per_table, strict=True):
        if not isinstance(metadata, Mapping) or not isinstance(metadata.get("regions"), Mapping):
            raise ValueError(
                f"{key} of {table.name} must be a mapping with a 'regions' mapping, as hp.tb.aggregate_points "
                f"writes it. Aggregate the table again, or delete {key} from every table, before concatenating."
            )
    difference = _metadata_difference(per_table)
    if difference is not None:
        position, fields = difference
        first, other = _comparable_metadata(per_table[0]), _comparable_metadata(per_table[position])
        raise ValueError(
            f"{key} differs in {fields!r} between {tables[0].name} and {tables[position].name}: "
            f"{ {field: first.get(field) for field in fields} } and "
            f"{ {field: other.get(field) for field in fields} }. Tables aggregated in ordinary and class-aware "
            "mode, or from different points columns, do not concatenate; aggregate every table the same way "
            "with hp.tb.aggregate_points."
        )
    return _merged_metadata(per_table, declared_regions=[table.regions for table in tables], region_order=region_order)


def _feature_matrices_with_mismatched_metadata(tables: Sequence[_Table]) -> dict[str, str]:
    """Return the described feature matrices whose metadata is missing from a table or differs between tables, with why."""
    reasons = {}
    for matrix in dict.fromkeys(matrix for table in tables for matrix in table.feature_matrices_metadata):
        lacking = [table.name for table in tables if matrix not in table.feature_matrices_metadata]
        if lacking:
            reasons[matrix] = f"has no metadata in {', '.join(lacking)}"
            continue
        difference = _metadata_difference([table.feature_matrices_metadata[matrix] for table in tables])
        if difference is not None:
            position, fields = difference
            reasons[matrix] = (
                f"has metadata that differs in {fields!r} between {tables[0].name} and {tables[position].name}"
            )
    return reasons


def _merged_feature_matrices_metadata(
    tables: Sequence[_Table], *, dropped_feature_matrices: Collection[str], region_order: Sequence[str]
) -> dict:
    """Return the feature-matrix metadata of the result, without that of the dropped feature matrices."""
    return {
        matrix: _merged_metadata(
            [table.feature_matrices_metadata[matrix] for table in tables],
            declared_regions=[table.regions for table in tables],
            region_order=region_order,
        )
        for matrix in tables[0].feature_matrices_metadata
        if matrix not in dropped_feature_matrices
    }


def _check_matrices(tables: Sequence[_Table], *, dropped_feature_matrices: Collection[str]) -> None:
    """Check that ``X``, the layers and the ``obsm`` matrices fit together across the tables.

    The tables need the same layer and ``obsm`` keys, the same ``obsm`` widths,
    and one format (dense, CSR, CSC or DataFrame) and one laziness per matrix.
    """
    layer_keys = [[key for key in table.adata.layers.keys() if key is not None] for table in tables]
    _require_same_keys(
        "layers",
        layer_keys,
        tables,
        fix="Add or delete these layers so that every table has the same ones before concatenating.",
    )
    obsm_keys = [[key for key in table.adata.obsm.keys() if key not in dropped_feature_matrices] for table in tables]
    _require_same_keys(
        "obsm keys",
        obsm_keys,
        tables,
        fix=(
            "Delete these matrices from the tables that have them, for example with del adata.obsm['X_pca'], "
            "and compute them after concatenating."
        ),
    )
    for key in obsm_keys[0]:
        by_width = _group({table.name: table.adata.obsm[key].shape[1:] for table in tables})
        if len(by_width) > 1:
            details = " and ".join(f"shape[1:] {width} in {', '.join(names)}" for width, names in by_width.items())
            raise ValueError(
                f"obsm[{key!r}] has {details}; concat stacks the rows of a matrix, so it needs the same columns in "
                f"every table. Delete it from every table, for example with del adata.obsm[{key!r}], and compute it "
                "after concatenating."
            )
    paths = [("X", None), *(("layers", key) for key in layer_keys[0]), *(("obsm", key) for key in obsm_keys[0])]
    for slot, key in paths:
        component = slot if key is None else f"{slot}[{key!r}]"
        values = {table.name: table.adata.X if key is None else getattr(table.adata, slot)[key] for table in tables}
        by_format = _group(
            {name: _checked_matrix_format(value, component=component, table=name) for name, value in values.items()}
        )
        if len(by_format) > 1:
            raise ValueError(
                f"{component} is {_describe_groups(by_format, names=_FORMAT_NAMES)}; concat requires one format per "
                "matrix, as anndata.concat converts mixed formats, lazily to COO, which Harpy rejects. Convert it to "
                "one format in every table before concatenating, for example with .tocsr(), .tocsc() or .toarray() "
                "on a matrix in memory."
            )
        if None in by_format:
            continue
        by_laziness = _group({name: _is_lazy(value) for name, value in values.items()})
        if len(by_laziness) > 1:
            lazy = by_laziness[True]
            raise ValueError(
                f"{component} is lazy in {', '.join(lazy)} and in memory in {', '.join(by_laziness[False])}; a backed "
                "matrix counts as lazy, and concat converts no matrix. Load it into memory in every table, for "
                f"example with {lazy[0]} = {lazy[0]}.to_memory(), or make it lazy in every table, for example by "
                "reading the tables with hp.tb.io.read_table(..., mode='lazy'). Subsetting with adata[mask].copy() "
                "loads backed matrices into memory, and keeps lazy ones lazy."
            )


def _checked_matrix_format(value: object, *, component: str, table: str) -> str | None:
    """Return the format of a matrix, ``"DataFrame"`` for a DataFrame, or ``None`` when absent.

    Raise for a matrix that concat cannot take, such as a Zarr array without AnnData encoding.
    """
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return "DataFrame"
    if isinstance(value, zarr.Array) and value.attrs.get("encoding-type") is None:
        raise ValueError(
            f"{component} of {table} is a Zarr array without AnnData encoding, which concat cannot read lazily. "
            "Wrap it with dask.array.from_zarr before concatenating."
        )
    try:
        return _matrix_format(value, label=f"{component} of {table}")
    except TypeError:
        kind = type(value._meta if isinstance(value, da.Array) else value).__name__
        dtype = getattr(value, "dtype", None)
        raise ValueError(
            f"{component} of {table} is a {kind}{'' if dtype is None else f' of dtype {dtype}'}; concat takes "
            "numeric dense, CSR or CSC matrices, and DataFrames in obsm. Convert it, for example with .tocsr() "
            "for another SciPy sparse format, or delete it from every table, before concatenating."
        ) from None
    except ValueError as error:
        raise ValueError(
            f"{error} Compute the shape and chunk sizes, for example with dask's compute_chunk_sizes(), before "
            "concatenating."
        ) from error


def _is_lazy(value: object) -> bool:
    """Return whether a matrix is lazy; a backed matrix counts as lazy."""
    return isinstance(value, (da.Array, zarr.Array, CSRDataset, CSCDataset))


def _lazy_if_backed(value: object) -> object:
    """Return a backed matrix as the Dask array that a lazy read gives, any other value as it is."""
    if isinstance(value, (CSRDataset, CSCDataset)):
        return _decode_anndata_element(value.group, mode="lazy")
    if isinstance(value, zarr.Array):
        return _decode_anndata_element(value, mode="lazy")
    return value


def _prepared_table(
    table: _Table, *, owned_uns_keys: Collection[str], dropped_feature_matrices: Collection[str]
) -> AnnData:
    """Return the table for anndata.concat: no metadata that concat merges itself, no dropped entries.

    Matrices are not copied, and backed ones are read lazily. ``obsp``, ``varp``,
    ``varm`` and ``raw`` are left out.
    """
    adata = table.adata
    uns = {key: deepcopy(value) for key, value in adata.uns.items() if key not in owned_uns_keys}
    # adata.layers holds X under the key None (anndata >= 0.13); AnnData sets X from it.
    return AnnData(
        obs=adata.obs.copy(),
        var=adata.var.copy(),
        uns=uns,
        layers={key: _lazy_if_backed(value) for key, value in adata.layers.items()},
        obsm={key: _lazy_if_backed(value) for key, value in adata.obsm.items() if key not in dropped_feature_matrices},
    )


def _merge_palette(result: AnnData, tables: Sequence[_Table], *, column: str) -> None:
    """Set the colors of a categorical ``obs`` column of the result, merged by category name."""
    key = f"{column}_colors"
    colors: dict[str, tuple[str, str]] = {}
    conflicts: list[str] = []
    merged_from = []
    for table in tables:
        if key not in table.adata.uns:
            continue
        palette = [str(color) for color in table.adata.uns[key]]
        categories = list(table.adata.obs[column].cat.categories)
        if len(palette) != len(categories):
            log.warning(
                f"concat ignores uns[{key!r}] of {table.name}: it holds {len(palette)} colors for the "
                f"{len(categories)} categories of obs[{column!r}], so its colors cannot be matched to them. Give it "
                "one color per category in .cat.categories, or delete it."
            )
            continue
        merged_from.append(table.name)
        for category, color in zip(categories, palette, strict=True):
            if category not in colors:
                colors[category] = (color, table.name)
            elif colors[category][0] != color:
                first_color, first_table = colors[category]
                conflicts.append(f"{category!r} is {first_color!r} in {first_table} and {color!r} in {table.name}")
    if not merged_from:
        return
    if conflicts:
        log.warning(
            f"Categories of obs[{column!r}] have different colors in different tables: {'; '.join(conflicts)}. "
            "concat keeps the color of the first table."
        )
    categories = list(result.obs[column].cat.categories)
    without = [category for category in categories if category not in colors]
    if without:
        log.warning(
            f"concat drops uns[{key!r}]: the categories {without!r} of obs[{column!r}] have no color in any table, "
            "so the colors would not match the categories, and scanpy assigns its default colors. To keep them, "
            "give every table one color per category in .cat.categories before concatenating."
        )
        return
    result.uns[key] = np.array([colors[category][0] for category in categories], dtype=object)
    log.info(f"concat merges uns[{key!r}] of {', '.join(merged_from)} by category name.")


def _log_dropped_entries(tables: Sequence[_Table]) -> None:
    dropped: dict[str, list[str]] = {}
    for table in tables:
        for slot in ("obsp", "varp", "varm"):
            for key in getattr(table.adata, slot).keys():
                dropped.setdefault(f"{slot}[{key!r}]", []).append(table.name)
        if table.adata.raw is not None:
            dropped.setdefault("raw", []).append(table.name)
    if dropped:
        details = ", ".join(f"{entry} of {', '.join(names)}" for entry, names in dropped.items())
        log.warning(
            f"concat drops {details}: the graphs (obsp), feature loadings (varp, varm) and snapshot before "
            "normalization (raw) of a table describe that table alone, and mean nothing for the combined table. "
            "Concatenate count tables, and compute these afterwards."
        )


def _log_left_out_regions(
    tables: Sequence[_Table], *, feature_matrices_key: str, dropped_feature_matrices: Collection[str]
) -> None:
    """Log the entries that a table's metadata holds for regions it does not declare, which concat leaves out."""
    for table in tables:
        by_path: dict[str, Mapping] = {}
        if table.canonical_metadata is not None:
            by_path[_CANONICAL_METADATA] = table.adata.uns[SPATIAL_COORDINATES_KEY][CANONICAL_OBSM_KEY]
        point_aggregation_metadata = table.adata.uns.get(_POINT_AGGREGATION_KEY)
        if point_aggregation_metadata is not None:
            by_path[f"uns[{_POINT_AGGREGATION_KEY!r}]"] = point_aggregation_metadata
        for matrix, metadata in table.feature_matrices_metadata.items():
            if matrix not in dropped_feature_matrices:
                by_path[f"uns[{feature_matrices_key!r}][{matrix!r}]"] = metadata
        left_out = {
            path: [region for region in metadata.get("regions", {}) if region not in table.regions]
            for path, metadata in by_path.items()
        }
        left_out = {path: regions for path, regions in left_out.items() if regions}
        if left_out:
            regions = list(dict.fromkeys(region for path_regions in left_out.values() for region in path_regions))
            log.info(
                f"concat leaves out the entries of the regions {regions!r} in {', '.join(left_out)} of {table.name}: "
                "the table does not declare these regions, for example because a subset removed them."
            )


def _log_dropped_var_columns(tables: Sequence[_Table], result: AnnData) -> None:
    columns = dict.fromkeys(column for table in tables for column in table.adata.var.columns)
    dropped = [column for column in columns if column not in result.var.columns]
    if dropped:
        log.warning(
            f"concat drops the var columns {dropped!r}, which are not equal in every table: feature statistics, "
            "such as those of hp.tb.pp.calculate_qc_metrics, describe one table. Compute them on the combined table."
        )


def _log_dropped_uns_entries(prepared: Sequence[AnnData], result: AnnData, *, uns_merge: str | None) -> None:
    keys = dict.fromkeys(key for adata in prepared for key in adata.uns)
    dropped = [key for key in keys if key not in result.uns]
    if dropped:
        log.warning(
            f"concat drops the uns entries {dropped!r}, which uns_merge={uns_merge!r} does not keep. concat merges "
            "Harpy's metadata itself and passes the other entries to anndata.concat; pass uns_merge='unique', "
            "'same' or 'first' to keep them."
        )


def _require_same_keys(what: str, keys: Sequence[Collection], tables: Sequence[_Table], *, fix: str) -> None:
    union = dict.fromkeys(key for table_keys in keys for key in table_keys)
    lacking = [
        f"{table.name} lacks {[key for key in union if key not in set(table_keys)]!r}"
        for table, table_keys in zip(tables, keys, strict=True)
        if any(key not in set(table_keys) for key in union)
    ]
    if lacking:
        raise ValueError(f"The {what} must be the same in every table: {'; '.join(lacking)}. {fix}")


def _group(values: Mapping[str, object]) -> dict[object, list[str]]:
    """Group the names of the tables by their values, in table order."""
    groups: dict[object, list[str]] = {}
    for name, value in values.items():
        groups.setdefault(value, []).append(name)
    return groups


def _describe_groups(groups: Mapping[object, Sequence[str]], *, names: Mapping[object, str] | None = None) -> str:
    """Describe groups of tables, such as "int64 in tables[0] and float64 in tables[1]"."""
    return " and ".join(
        f"{value if names is None else names[value]} in {', '.join(tables)}" for value, tables in groups.items()
    )


def _describe(reasons: Mapping[str, str]) -> str:
    return "; ".join(f"{name!r} {reason}" for name, reason in reasons.items())


def _dtype_name(dtype: object) -> str:
    return "category" if isinstance(dtype, pd.CategoricalDtype) else str(dtype)


def _preview(values: Sequence, limit: int = 10) -> str:
    values = list(values)
    shown = repr(values[:limit])
    return shown if len(values) <= limit else f"{shown[:-1]}, ...] ({len(values)} in all)"
