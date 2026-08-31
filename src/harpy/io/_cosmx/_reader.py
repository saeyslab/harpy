from __future__ import annotations

import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from loguru import logger as log
from spatialdata import SpatialData, read_zarr
from spatialdata.models.models import ScaleFactors_t

from harpy import __version__
from harpy._metadata import _FEATURE_PANELS_METADATA_KEY, _HARPY_METADATA_KEY, _PROVENANCE_METADATA_KEY
from harpy.io._cosmx._discovery import _discover_cosmx
from harpy.io._cosmx._images import _add_morphology_images, _select_channels
from harpy.io._cosmx._labels import _add_compartment_labels, _add_instance_labels
from harpy.io._cosmx._models import (
    _COMPARTMENT_LABELS_PRODUCT,
    _INSTANCE_LABELS_PRODUCT,
    _MORPHOLOGY_PRODUCT,
    _TRANSCRIPTS_PRODUCT,
    CosmxSample,
    _CosmxPreview,
    _validate_identifier,
)
from harpy.io._cosmx._preview import _preview_cosmx
from harpy.io._cosmx._transcripts import (
    _add_transcript_points,
    _feature_panel_metadata,
    _feature_panel_name,
)
from harpy.io._cosmx._validation import _validate_cosmx_sdata


@dataclass(frozen=True)
class _PreparedCosmxSample:
    sample_id: str
    config: CosmxSample
    preview: _CosmxPreview


def cosmx(
    samples: Mapping[str, CosmxSample],
    output: str | Path,
    *,
    images: bool = True,
    instance_labels: bool = True,
    compartment_labels: bool = True,
    points: bool = True,
    output_image_name: str = "morphology_image",
    output_instance_labels_name: str = "instance_labels",
    output_compartment_labels_name: str = "compartment_labels",
    output_points_name: str = "transcripts",
    image_chunks: tuple[int, int, int] = (1, 1024, 1024),
    labels_chunks: tuple[int, int] = (1024, 1024),
    raster_scale_factors: ScaleFactors_t | None = None,
    points_blocksize: str | int = "64MB",
    overwrite: bool = False,
) -> SpatialData:
    """Read one or more decoded CosMx samples into a backed SpatialData store.

    Each mapping entry is discovered and planned before any payload is decoded.
    For a sample, positioned FOV availability is intersected only across the
    enabled modalities. Those modalities then share the sample's FOV selection,
    mosaics, orientation, and independent pixel and micron coordinate systems.
    Every generated element and coordinate system is prefixed by its sample ID,
    including for a one-entry mapping.

    The complete output is transactional. Data are written directly into a
    temporary sibling Zarr store and published only after that store can be
    reopened. ``overwrite=True`` replaces a complete CosMx store; it never
    merges elements into an existing SpatialData object.

    Parameters
    ----------
    samples
        Non-empty mapping from exact sample identifiers to immutable
        :class:`CosmxSample` configurations. Identifiers must match
        ``^[A-Za-z][A-Za-z0-9_]*$``.
    output
        Destination for the backed SpatialData Zarr store. In-memory output is
        not supported.
    images
        Whether to ingest morphology images.
    instance_labels
        Whether to ingest instance-label rasters with globally unique
        ``uint32`` IDs.
    compartment_labels
        Whether to ingest semantic compartment-label rasters.
    points
        Whether to ingest out-of-core transcript points.
    output_image_name
        Base name for morphology elements.
    output_instance_labels_name
        Base name for instance-label elements.
    output_compartment_labels_name
        Base name for compartment-label elements.
    output_points_name
        Base name for transcript-points elements.
    image_chunks
        Final morphology chunks in ``(c, y, x)`` order.
    labels_chunks
        Final instance- and compartment-label chunks in ``(y, x)`` order.
    raster_scale_factors
        Optional shared relative scale factors for image and label pyramids.
    points_blocksize
        Approximate byte block used to partition source transcript CSVs into
        the lazy points dataframe.
    overwrite
        Replace an existing output only if it is a readable SpatialData store
        created by this CosMx reader.

    Returns
    -------
    Reopened SpatialData object backed by ``output``.

    Metadata
    --------
    Reader metadata is stored under ``sdata.attrs["harpy"]``. This is a
    versioned Harpy convention, not part of the SpatialData standard. Its
    structure is::

        sdata.attrs["harpy"]
        ├── metadata_version: 1
        │
        ├── provenance
        │   ├── reader: "cosmx"
        │   └── reader_version: <Harpy version>
        │
        ├── images
        │   └── <image element name>
        │       ├── sample_id: <sample identifier>
        │       ├── fovs: [<source FOVs>]
        │       ├── mosaic
        │       │   ├── mode: "spatial_groups" | "single"
        │       │   └── adjacency_tolerance_px: <int> | None
        │       ├── source_origin_px
        │       │   ├── x: <int>
        │       │   └── y: <int>
        │       ├── orientation
        │       │   ├── flip_x: <bool>
        │       │   └── flip_y: <bool>
        │       ├── pixel_size_um: <float>
        │       ├── acquisition_timestamp: <source OrigTimeStamp>  [optional]
        │       └── channels
        │           └── [<channel record>, ...]
        │               ├── channel_id: <str>
        │               ├── name: <str>
        │               ├── source_plane: <int>
        │               └── output_coordinate: <str>
        │
        ├── labels
        │   ├── <instance-label element name>
        │   │   ├── sample_id, fovs, mosaic, source_origin_px,
        │   │   │   orientation, pixel_size_um, acquisition_timestamp [optional]
        │   │   └── instance_id_encoding
        │   │       ├── background: 0
        │   │       ├── base: <number of source-dtype values>
        │   │       └── formula:
        │   │           "global_id = (fov - 1) * base + local_id"
        │   │
        │   └── <compartment-label element name>
        │       ├── sample_id, fovs, mosaic, source_origin_px,
        │       │   orientation, pixel_size_um, acquisition_timestamp [optional]
        │       └── categories
        │           ├── 0: "background"
        │           ├── 1: "nuclear"
        │           ├── 2: "membrane"
        │           └── 3: "cytoplasmic"
        │
        ├── points
        │   └── <transcript points element name>
        │       ├── sample_id, fovs, mosaic, source_origin_px,
        │       │   orientation, pixel_size_um, acquisition_timestamp [optional]
        │       └── feature_panel: "feature_panel_<content hash>"  [optional]
        │
        └── feature_panels                                      [optional]
            └── feature_panel_<content hash>
                ├── feature_column: "gene"
                ├── class_column: "code_class"
                ├── categories: [<ordered feature classes>]
                └── targets_by_class
                    └── <feature class>: [<authoritative targets>]

    ``provenance`` records only the reader and Harpy version. The ``images``,
    ``labels``, and ``points`` mappings are keyed by exact SpatialData element
    names. Disabled modality registries are omitted. SpatialData coordinate
    systems and transformations are stored with the spatial elements and are
    not part of this root Harpy metadata tree. ``acquisition_timestamp`` is the
    source ``OrigTimeStamp`` preserved verbatim when it is non-empty and
    consistent across all morphology TIFFs for the sample; otherwise the field
    is omitted.

    When an authoritative run-level plex is available, ``feature_panel`` is a
    points-record reference whose value is the key of a content-addressed
    record in the shared ``feature_panels`` registry. Identical panels across
    samples are stored once. For example, two transcript points elements can
    reference the same shared panel record::

        sdata.attrs["harpy"]
        ├── points
        │   ├── sample_a_transcripts_mosaic_1
        │   │   └── feature_panel: feature_panel_8a31b240c75e1234
        │   └── sample_b_transcripts_mosaic_1
        │       └── feature_panel: feature_panel_8a31b240c75e1234
        │
        └── feature_panels
            └── feature_panel_8a31b240c75e1234
                ├── feature_column: gene
                ├── class_column: code_class
                ├── categories: [...]
                └── targets_by_class: {...}

    The relationship can be resolved exactly as follows::

        panel_key = sdata.attrs["harpy"]["points"][points_name]["feature_panel"]
        panel = sdata.attrs["harpy"]["feature_panels"][panel_key]

    If no plex is available, transcript ingestion still succeeds and neither
    the shared panel nor its points reference is written.

    When a feature panel is available, every target represented in a transcript
    points element must occur in the shared feature-panel record referenced by
    that element. Its observed feature class must also match the class assigned
    to that target by the panel. Panel targets may have zero detections. These
    row-level checks run when the lazy transcript partitions are materialized
    during writing. Without a feature panel, this cross-validation is skipped
    and the transcript-provided targets and classes are retained.

    Notes
    -----
    FOV selection is permissive for known FOVs. For each sample, the reader
    intersects requested and positioned FOVs with availability across all
    enabled modalities. Known requested FOVs that do not satisfy that
    intersection are excluded, reported through logging, and not persisted as
    exclusion metadata. An ID absent from the manifest raises immediately, and
    a sample for which no requested FOV remains usable also raises.

    This reader does not create cell-boundary shapes, tables, cell statistics,
    or expression aggregates. Instances meeting at FOV boundaries are not
    merged.
    """
    enabled_product_output_names = _validate_enabled_product_output_names(
        images=images,
        output_image_name=output_image_name,
        instance_labels=instance_labels,
        output_instance_labels_name=output_instance_labels_name,
        compartment_labels=compartment_labels,
        output_compartment_labels_name=output_compartment_labels_name,
        points=points,
        output_points_name=output_points_name,
    )

    output_path = Path(output).expanduser().resolve()
    output_exists = output_path.exists()
    if output_exists and not overwrite:
        raise FileExistsError(f"CosMx output already exists: {output_path}")
    if output_exists:
        _validate_replaceable_output(output_path)

    prepared = _prepare_cosmx_samples(
        samples,
        output=output_path,
        products=tuple(enabled_product_output_names),
    )
    _validate_planned_names(prepared, enabled_product_output_names=enabled_product_output_names)

    _log_prepared_samples(prepared)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = _unique_sibling(output_path, purpose="staging")
    try:
        SpatialData().write(staging)
        sdata = read_zarr(staging)

        for sample in prepared:
            sdata = _write_cosmx_sample(
                sdata,
                sample,
                enabled_product_output_names=enabled_product_output_names,
                image_chunks=image_chunks,
                labels_chunks=labels_chunks,
                raster_scale_factors=raster_scale_factors,
                points_blocksize=points_blocksize,
            )

        # Reopening validates the final-format staging store before publication.
        read_zarr(staging)
        return _publish_staging_store(staging, output_path, replace_existing=output_exists)
    except Exception:
        _remove_generated_path(staging)
        raise


def add_cosmx_samples(
    output: str | Path,
    samples: Mapping[str, CosmxSample],
    *,
    images: bool = True,
    instance_labels: bool = True,
    compartment_labels: bool = True,
    points: bool = True,
    output_image_name: str = "morphology_image",
    output_instance_labels_name: str = "instance_labels",
    output_compartment_labels_name: str = "compartment_labels",
    output_points_name: str = "transcripts",
    image_chunks: tuple[int, int, int] = (1, 1024, 1024),
    labels_chunks: tuple[int, int] = (1024, 1024),
    raster_scale_factors: ScaleFactors_t | None = None,
    points_blocksize: str | int = "64MB",
) -> SpatialData:
    """Add independently named CosMx samples to an existing reader-created store.

    The destination is validated before any requested source is discovered.
    This includes scanning the panel-declared feature and class columns of
    existing transcript points to verify their contents against referenced
    feature panels. All new sample IDs, element names, coordinate systems, and
    feature panels are then preflighted against both the requested batch and
    the existing store before payload decoding begins.

    Addition is strictly incremental: existing samples and elements are never
    replaced, and there is no ``overwrite`` mode. Each new element is a commit
    boundary. Its metadata and the current Harpy reader version are persisted
    only after its payload write succeeds. If that metadata write fails, Harpy
    makes a best-effort attempt to delete only that element and restore the
    preceding root attributes. Elements committed earlier remain in the store.

    Parameters
    ----------
    output
        Existing backed SpatialData Zarr store created by :func:`cosmx`.
    samples
        Non-empty mapping from new sample identifiers to :class:`CosmxSample`
        configurations.
    images, instance_labels, compartment_labels, points
        Modalities to ingest for every requested sample. At least one must be
        enabled.
    output_image_name, output_instance_labels_name, output_compartment_labels_name, output_points_name
        Output base names. Every generated element remains prefixed by its
        sample identifier and suffixed by its mosaic number.
    image_chunks, labels_chunks
        Final chunks for morphology and label rasters respectively.
    raster_scale_factors
        Optional shared relative scale factors for image and label pyramids.
    points_blocksize
        Approximate byte block used to partition source transcript CSVs into
        the lazy points dataframe.

    Returns
    -------
    SpatialData
        The incrementally extended object backed by ``output``.

    Notes
    -----
    This operation is not sample-level transactional. A failure leaves elements
    committed earlier in the call intact. The caller must ensure that no other
    process writes to the destination concurrently.
    """
    enabled_product_output_names = _validate_enabled_product_output_names(
        images=images,
        output_image_name=output_image_name,
        instance_labels=instance_labels,
        output_instance_labels_name=output_instance_labels_name,
        compartment_labels=compartment_labels,
        output_compartment_labels_name=output_compartment_labels_name,
        points=points,
        output_points_name=output_points_name,
    )
    output_path = Path(output).expanduser().resolve()
    try:
        sdata = read_zarr(output_path)
    except Exception as error:
        raise ValueError(f"Could not read incremental CosMx SpatialData Zarr destination: {output_path}") from error

    existing_sample_ids = _validate_cosmx_sdata(sdata, check_point_contents=True)
    _validate_new_sample_ids(samples, existing_sample_ids=existing_sample_ids)
    prepared = _prepare_cosmx_samples(
        samples,
        output=output_path,
        products=tuple(enabled_product_output_names),
    )

    _validate_planned_names(
        prepared,
        enabled_product_output_names=enabled_product_output_names,
        existing_element_names=tuple(name for _, name, _ in sdata.gen_elements()),
        existing_coordinate_systems=tuple(sdata.coordinate_systems),
    )
    harpy_metadata = sdata.attrs[_HARPY_METADATA_KEY]
    assert isinstance(harpy_metadata, dict)
    existing_panels = harpy_metadata.get(_FEATURE_PANELS_METADATA_KEY, {})
    assert isinstance(existing_panels, dict)
    _validate_planned_panels(prepared, existing_panels=existing_panels)
    _log_prepared_samples(prepared)

    for sample in prepared:
        sdata = _write_cosmx_sample(
            sdata,
            sample,
            enabled_product_output_names=enabled_product_output_names,
            image_chunks=image_chunks,
            labels_chunks=labels_chunks,
            raster_scale_factors=raster_scale_factors,
            points_blocksize=points_blocksize,
        )
    return sdata


def _validate_enabled_product_output_names(
    *,
    images: bool,
    output_image_name: str,
    instance_labels: bool,
    output_instance_labels_name: str,
    compartment_labels: bool,
    output_compartment_labels_name: str,
    points: bool,
    output_points_name: str,
) -> dict[str, str]:
    """Return the output base name for each enabled CosMx product.

    Each public modality flag is paired with the output base name for that
    product. Disabled products are omitted. The resulting mapping is the
    reader's single source of truth for which products discovery must require,
    which element names must be preflighted, and which payloads must be
    written.

    Returns
    -------
    dict[str, str]
        Mapping from an enabled CosMx product identifier to its validated
        output base name, in canonical product order. For example, an
        image-and-transcript request returns ``{"morphology":
        "morphology_image", "transcripts": "transcripts"}``.

    Notes
    -----
    Only names belonging to enabled products are validated, because disabled
    products do not generate SpatialData elements.
    """
    configured_outputs = {
        _MORPHOLOGY_PRODUCT: (images, output_image_name),
        _INSTANCE_LABELS_PRODUCT: (instance_labels, output_instance_labels_name),
        _COMPARTMENT_LABELS_PRODUCT: (compartment_labels, output_compartment_labels_name),
        _TRANSCRIPTS_PRODUCT: (points, output_points_name),
    }
    enabled_product_output_names = {
        product: output_name for product, (enabled, output_name) in configured_outputs.items() if enabled
    }
    if not enabled_product_output_names:
        raise ValueError("CosMx ingestion requires at least one enabled modality.")
    for product, output_name in enabled_product_output_names.items():
        if not isinstance(output_name, str) or not output_name or "/" in output_name:
            raise ValueError(
                f"CosMx {product} output base name must be a non-empty path-safe string, found {output_name!r}."
            )
    return enabled_product_output_names


def _validate_new_sample_ids(
    samples: Mapping[str, CosmxSample],
    *,
    existing_sample_ids: frozenset[str],
) -> None:
    """Reject duplicate sample IDs before discovering any requested source."""
    if not isinstance(samples, Mapping) or not samples:
        raise ValueError("CosMx samples must be a non-empty mapping.")
    for sample_id in samples:
        _validate_identifier(sample_id, name="sample identifier")
    collisions = sorted(set(samples) & existing_sample_ids)
    if collisions:
        raise ValueError(f"CosMx sample identifiers already exist in the destination: {collisions}.")


def _log_prepared_samples(samples: tuple[_PreparedCosmxSample, ...]) -> None:
    for sample in samples:
        for message in sample.preview.diagnostics:
            log.info(f"CosMx sample {sample.sample_id!r}: {message}")
        if sample.preview.excluded_fovs:
            log.warning(f"CosMx sample {sample.sample_id!r} excludes FOVs {list(sample.preview.excluded_fovs)}.")


def _prepare_cosmx_samples(
    samples: Mapping[str, CosmxSample],
    *,
    output: Path,
    products: tuple[str, ...],
) -> tuple[_PreparedCosmxSample, ...]:
    """Discover and validate every sample before any payload is written."""
    if not isinstance(samples, Mapping) or not samples:
        raise ValueError("CosMx samples must be a non-empty mapping.")

    prepared = []
    for sample_id, config in samples.items():
        _validate_identifier(sample_id, name="sample identifier")
        if not isinstance(config, CosmxSample):
            raise TypeError(
                f"CosMx sample {sample_id!r} must be configured with CosmxSample, found {type(config).__name__}."
            )
        manifest = _discover_cosmx(config.path, products=products)
        _validate_source_output_paths(manifest.root, output)
        requested_fovs = manifest.fov_ids if config.fovs is None else tuple(sorted({int(fov) for fov in config.fovs}))
        preview = _preview_cosmx(
            manifest,
            fovs=requested_fovs,
            products=products,
            mosaic_mode=config.mosaic_mode,
            adjacency_tolerance_px=config.adjacency_tolerance_px,
        )
        if not preview.mosaics:
            raise ValueError(
                f"CosMx sample {sample_id!r} has no common positioned FOVs for the requested selection and "
                f"enabled products {list(products)}."
            )
        if _MORPHOLOGY_PRODUCT in products:
            _select_channels(preview, config.channels)
        prepared.append(_PreparedCosmxSample(sample_id=sample_id, config=config, preview=preview))

    result = tuple(prepared)
    _validate_planned_panels(result)
    return result


def _write_cosmx_sample(
    sdata: SpatialData,
    sample: _PreparedCosmxSample,
    *,
    enabled_product_output_names: Mapping[str, str],
    image_chunks: tuple[int, int, int],
    labels_chunks: tuple[int, int],
    raster_scale_factors: ScaleFactors_t | None,
    points_blocksize: str | int,
) -> SpatialData:
    """Write one fully prepared sample into the shared backed store."""
    sample_id = sample.sample_id
    config = sample.config
    preview = sample.preview
    coordinate_system = f"{sample_id}_{config.coordinate_system}"

    if _MORPHOLOGY_PRODUCT in enabled_product_output_names:
        started = perf_counter()
        sdata = _add_morphology_images(
            sdata,
            preview,
            channels=config.channels,
            output_image_name=f"{sample_id}_{enabled_product_output_names[_MORPHOLOGY_PRODUCT]}",
            coordinate_system=coordinate_system,
            flip_x=config.flip_x,
            flip_y=config.flip_y,
            chunks=image_chunks,
            scale_factors=raster_scale_factors,
            sample_id=sample_id,
            reader_version=__version__,
            overwrite=False,
        )
        log.info(f"Wrote CosMx sample {sample_id!r} morphology images in {perf_counter() - started:.2f} seconds.")

    if _INSTANCE_LABELS_PRODUCT in enabled_product_output_names:
        started = perf_counter()
        sdata = _add_instance_labels(
            sdata,
            preview,
            output_labels_name=f"{sample_id}_{enabled_product_output_names[_INSTANCE_LABELS_PRODUCT]}",
            coordinate_system=coordinate_system,
            flip_x=config.flip_x,
            flip_y=config.flip_y,
            chunks=labels_chunks,
            scale_factors=raster_scale_factors,
            sample_id=sample_id,
            reader_version=__version__,
            overwrite=False,
        )
        log.info(f"Wrote CosMx sample {sample_id!r} instance labels in {perf_counter() - started:.2f} seconds.")

    if _COMPARTMENT_LABELS_PRODUCT in enabled_product_output_names:
        started = perf_counter()
        sdata = _add_compartment_labels(
            sdata,
            preview,
            output_labels_name=f"{sample_id}_{enabled_product_output_names[_COMPARTMENT_LABELS_PRODUCT]}",
            coordinate_system=coordinate_system,
            flip_x=config.flip_x,
            flip_y=config.flip_y,
            chunks=labels_chunks,
            scale_factors=raster_scale_factors,
            sample_id=sample_id,
            reader_version=__version__,
            overwrite=False,
        )
        log.info(f"Wrote CosMx sample {sample_id!r} compartment labels in {perf_counter() - started:.2f} seconds.")

    if _TRANSCRIPTS_PRODUCT in enabled_product_output_names:
        started = perf_counter()
        sdata = _add_transcript_points(
            sdata,
            preview,
            output_points_name=f"{sample_id}_{enabled_product_output_names[_TRANSCRIPTS_PRODUCT]}",
            coordinate_system=coordinate_system,
            flip_x=config.flip_x,
            flip_y=config.flip_y,
            blocksize=points_blocksize,
            sample_id=sample_id,
            reader_version=__version__,
            overwrite=False,
        )
        log.info(f"Wrote CosMx sample {sample_id!r} transcript points in {perf_counter() - started:.2f} seconds.")
    return sdata


def _validate_source_output_paths(source: Path, output: Path) -> None:
    """Prevent the reader from writing into, over, or around its source run."""
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError(f"CosMx source and output paths must not equal or contain one another: {source}, {output}")


def _validate_planned_names(
    samples: tuple[_PreparedCosmxSample, ...],
    *,
    enabled_product_output_names: Mapping[str, str],
    existing_element_names: tuple[str, ...] = (),
    existing_coordinate_systems: tuple[str, ...] = (),
) -> None:
    """Validate planned SpatialData element and coordinate-system names.

    Element names must be globally unique. Coordinate-system names may be
    shared by aligned image, labels, and points elements within one sample
    mosaic, but must not collide across different sample mosaics. The reader
    therefore validates the element and coordinate-system namespaces
    separately, rejecting both exact and case-insensitive collisions.

    Within one sample mosaic, all enabled modalities intentionally receive the
    same pixel and micron coordinate systems. Those names are registered once
    per mosaic below. Names belonging to different sample mosaics, however,
    must remain distinct; otherwise SpatialData would accept them as one shared
    coordinate system and incorrectly imply cross-sample alignment.

    Sample prefixes alone do not guarantee uniqueness because identifiers are
    joined with underscores. For example, sample/base pairs ``("a_b", "c")``
    and ``("a", "b_c")`` both generate ``"a_b_c_1"``. The owner registries
    detect such ambiguities before any payload is written. Case-folded keys
    additionally reject names that differ only by letter case.
    """
    element_owners: dict[str, tuple[str, str]] = {}
    coordinate_owners: dict[str, tuple[str, str]] = {}
    for name in existing_element_names:
        _register_planned_name(
            element_owners,
            name,
            owner=("existing store", name),
            kind="element",
        )
    for name in existing_coordinate_systems:
        _register_planned_name(
            coordinate_owners,
            name,
            owner=("existing store", name),
            kind="coordinate system",
        )
    for sample in samples:
        for mosaic in sample.preview.mosaics:
            for product, output_name in enabled_product_output_names.items():
                element_name = f"{sample.sample_id}_{output_name}_mosaic_{mosaic.mosaic}"
                _register_planned_name(element_owners, element_name, owner=(sample.sample_id, product), kind="element")
            pixel_coordinate_system = f"{sample.sample_id}_{sample.config.coordinate_system}_{mosaic.mosaic}"
            _register_planned_name(
                coordinate_owners,
                pixel_coordinate_system,
                owner=(sample.sample_id, "pixel coordinate system"),
                kind="coordinate system",
            )
            _register_planned_name(
                coordinate_owners,
                f"{pixel_coordinate_system}_micron",
                owner=(sample.sample_id, "micron coordinate system"),
                kind="coordinate system",
            )


def _register_planned_name(
    registry: dict[str, tuple[str, str]],
    name: str,
    *,
    owner: tuple[str, str],
    kind: str,
) -> None:
    if not name or "/" in name:
        raise ValueError(f"Invalid CosMx output {kind} name {name!r}.")
    normalized = name.casefold()
    previous = registry.setdefault(normalized, owner)
    if previous != owner:
        raise ValueError(f"CosMx output {kind} {name!r} is planned by both {previous} and {owner}.")


def _validate_planned_panels(
    samples: tuple[_PreparedCosmxSample, ...],
    *,
    existing_panels: Mapping[str, object] | None = None,
) -> None:
    """Resolve planned panels and detect truncated content-hash collisions."""
    panels: dict[str, object] = {} if existing_panels is None else dict(existing_panels)
    for sample in samples:
        if sample.preview.manifest.feature_panel is None:
            continue
        metadata = _feature_panel_metadata(sample.preview)
        name = _feature_panel_name(metadata)
        existing = panels.setdefault(name, metadata)
        if existing != metadata:
            raise ValueError(f"CosMx feature-panel hash collision for {name!r}.")


def _validate_replaceable_output(output: Path) -> None:
    """Require whole-store overwrite targets to originate from this reader."""
    try:
        existing = read_zarr(output)
    except Exception as error:
        raise ValueError(f"Refusing to replace unreadable or non-SpatialData output: {output}") from error
    harpy_metadata = existing.attrs.get(_HARPY_METADATA_KEY)
    provenance = harpy_metadata.get(_PROVENANCE_METADATA_KEY) if isinstance(harpy_metadata, dict) else None
    if not isinstance(provenance, dict) or provenance.get("reader") != "cosmx":
        raise ValueError(f"Refusing to replace SpatialData output not created by the CosMx reader: {output}")


def _unique_sibling(output: Path, *, purpose: str) -> Path:
    """Return a unique generated sibling path for staging or backup state."""
    while True:
        candidate = output.with_name(f".{output.name}.cosmx-{uuid.uuid4().hex}.{purpose}")
        if not candidate.exists():
            return candidate


def _publish_staging_store(staging: Path, output: Path, *, replace_existing: bool) -> SpatialData:
    """Atomically publish a validated staging store and restore old output on failure."""
    if output.exists() != replace_existing:
        raise RuntimeError(f"CosMx output existence changed during ingestion: {output}")

    backup = _unique_sibling(output, purpose="backup") if replace_existing else None
    if backup is not None:
        output.replace(backup)
    try:
        staging.replace(output)
        result = read_zarr(output)
    except Exception:
        if output.exists() and not staging.exists():
            output.replace(staging)
        if backup is not None and backup.exists() and not output.exists():
            backup.replace(output)
        raise
    else:
        if backup is not None:
            _remove_generated_path(backup)
        return result


def _remove_generated_path(path: Path) -> None:
    """Remove a reader-generated staging or backup path if it exists."""
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
