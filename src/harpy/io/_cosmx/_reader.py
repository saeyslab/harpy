from __future__ import annotations

import shutil
import uuid
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from loguru import logger as log
from spatialdata import SpatialData, read_zarr
from spatialdata.models.models import ScaleFactors_t

from harpy import __version__
from harpy._metadata import _HARPY_METADATA_KEY, _PROVENANCE_METADATA_KEY, _harpy_metadata
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


@dataclass(frozen=True)
class _PreparedCosmxSample:
    sample_id: str
    config: CosmxSample
    preview: _CosmxPreview


def cosmx(
    samples: Mapping[str, CosmxSample],
    output: str | Path,
    *,
    morphology: bool = True,
    instance_labels: bool = True,
    compartment_labels: bool = True,
    transcripts: bool = True,
    output_image_name: str = "morphology_image",
    output_instance_labels_name: str = "instance_labels",
    output_compartment_labels_name: str = "compartment_labels",
    output_points_name: str = "transcripts",
    image_chunks: tuple[int, int, int] = (1, 1024, 1024),
    labels_chunks: tuple[int, int] = (1024, 1024),
    raster_scale_factors: ScaleFactors_t | None = None,
    transcript_blocksize: str | int = "64MB",
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
    morphology
        Whether to ingest morphology images.
    instance_labels
        Whether to ingest instance-label rasters with globally unique
        ``uint32`` IDs.
    compartment_labels
        Whether to ingest semantic compartment-label rasters.
    transcripts
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
    transcript_blocksize
        Positive byte count or Dask byte-size string used to partition each
        transcript CSV.
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
    enabled = (
        (_MORPHOLOGY_PRODUCT, morphology),
        (_INSTANCE_LABELS_PRODUCT, instance_labels),
        (_COMPARTMENT_LABELS_PRODUCT, compartment_labels),
        (_TRANSCRIPTS_PRODUCT, transcripts),
    )
    products = tuple(product for product, is_enabled in enabled if is_enabled)
    if not products:
        raise ValueError("CosMx ingestion requires at least one enabled modality.")

    output_path = Path(output).expanduser().resolve()
    prepared = _prepare_cosmx_samples(
        samples,
        products=products,
        output=output_path,
        output_names={
            _MORPHOLOGY_PRODUCT: output_image_name,
            _INSTANCE_LABELS_PRODUCT: output_instance_labels_name,
            _COMPARTMENT_LABELS_PRODUCT: output_compartment_labels_name,
            _TRANSCRIPTS_PRODUCT: output_points_name,
        },
    )

    output_exists = output_path.exists()
    if output_exists and not overwrite:
        raise FileExistsError(f"CosMx output already exists: {output_path}")
    if output_exists:
        _validate_replaceable_output(output_path)

    for sample in prepared:
        for message in sample.preview.diagnostics:
            log.info(f"CosMx sample {sample.sample_id!r}: {message}")
        if sample.preview.excluded_fovs:
            log.warning(f"CosMx sample {sample.sample_id!r} excludes FOVs {list(sample.preview.excluded_fovs)}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = _unique_sibling(output_path, purpose="staging")
    try:
        SpatialData().write(staging)
        sdata = read_zarr(staging)

        for sample in prepared:
            sdata = _write_cosmx_sample(
                sdata,
                sample,
                morphology=morphology,
                instance_labels=instance_labels,
                compartment_labels=compartment_labels,
                transcripts=transcripts,
                output_image_name=output_image_name,
                output_instance_labels_name=output_instance_labels_name,
                output_compartment_labels_name=output_compartment_labels_name,
                output_points_name=output_points_name,
                image_chunks=image_chunks,
                labels_chunks=labels_chunks,
                raster_scale_factors=raster_scale_factors,
                transcript_blocksize=transcript_blocksize,
            )

        attrs = deepcopy(sdata.attrs)
        harpy_metadata = _harpy_metadata(attrs)
        harpy_metadata[_PROVENANCE_METADATA_KEY] = {
            "reader": "cosmx",
            "reader_version": __version__,
        }
        sdata.attrs = attrs
        sdata.write_attrs()

        # Reopening validates the final-format staging store before publication.
        read_zarr(staging)
        return _publish_staging_store(staging, output_path, replace_existing=output_exists)
    except Exception:
        _remove_generated_path(staging)
        raise


def _prepare_cosmx_samples(
    samples: Mapping[str, CosmxSample],
    *,
    products: tuple[str, ...],
    output: Path,
    output_names: Mapping[str, str],
) -> tuple[_PreparedCosmxSample, ...]:
    """Discover and validate every sample before any output is staged."""
    if not isinstance(samples, Mapping) or not samples:
        raise ValueError("CosMx samples must be a non-empty mapping.")
    for product in products:
        base_name = output_names[product]
        if not isinstance(base_name, str) or not base_name or "/" in base_name:
            raise ValueError(f"CosMx {product} output base name must be a non-empty path-safe string, found {base_name!r}.")

    prepared = []
    for sample_id, config in samples.items():
        _validate_identifier(sample_id, name="sample identifier")
        if not isinstance(config, CosmxSample):
            raise TypeError(
                f"CosMx sample {sample_id!r} must be configured with CosmxSample, found {type(config).__name__}."
            )
        manifest = _discover_cosmx(config.path, products=products)
        _validate_source_output_paths(manifest.root, output)
        requested_fovs = (
            manifest.fov_ids if config.fovs is None else tuple(sorted({int(fov) for fov in config.fovs}))
        )
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
    _validate_planned_names(result, products=products, output_names=output_names)
    _validate_planned_panels(result)
    return result


def _write_cosmx_sample(
    sdata: SpatialData,
    sample: _PreparedCosmxSample,
    *,
    morphology: bool,
    instance_labels: bool,
    compartment_labels: bool,
    transcripts: bool,
    output_image_name: str,
    output_instance_labels_name: str,
    output_compartment_labels_name: str,
    output_points_name: str,
    image_chunks: tuple[int, int, int],
    labels_chunks: tuple[int, int],
    raster_scale_factors: ScaleFactors_t | None,
    transcript_blocksize: str | int,
) -> SpatialData:
    """Write one fully prepared sample into the shared staging store."""
    sample_id = sample.sample_id
    config = sample.config
    preview = sample.preview
    coordinate_system = f"{sample_id}_{config.coordinate_system}"

    if morphology:
        started = perf_counter()
        sdata = _add_morphology_images(
            sdata,
            preview,
            channels=config.channels,
            output_image_name=f"{sample_id}_{output_image_name}",
            coordinate_system=coordinate_system,
            flip_x=config.flip_x,
            flip_y=config.flip_y,
            chunks=image_chunks,
            scale_factors=raster_scale_factors,
            sample_id=sample_id,
            overwrite=False,
        )
        log.info(f"Wrote CosMx sample {sample_id!r} morphology images in {perf_counter() - started:.2f} seconds.")

    if instance_labels:
        started = perf_counter()
        sdata = _add_instance_labels(
            sdata,
            preview,
            output_labels_name=f"{sample_id}_{output_instance_labels_name}",
            coordinate_system=coordinate_system,
            flip_x=config.flip_x,
            flip_y=config.flip_y,
            chunks=labels_chunks,
            scale_factors=raster_scale_factors,
            sample_id=sample_id,
            overwrite=False,
        )
        log.info(f"Wrote CosMx sample {sample_id!r} instance labels in {perf_counter() - started:.2f} seconds.")

    if compartment_labels:
        started = perf_counter()
        sdata = _add_compartment_labels(
            sdata,
            preview,
            output_labels_name=f"{sample_id}_{output_compartment_labels_name}",
            coordinate_system=coordinate_system,
            flip_x=config.flip_x,
            flip_y=config.flip_y,
            chunks=labels_chunks,
            scale_factors=raster_scale_factors,
            sample_id=sample_id,
            overwrite=False,
        )
        log.info(
            f"Wrote CosMx sample {sample_id!r} compartment labels in {perf_counter() - started:.2f} seconds."
        )

    if transcripts:
        started = perf_counter()
        sdata = _add_transcript_points(
            sdata,
            preview,
            output_points_name=f"{sample_id}_{output_points_name}",
            coordinate_system=coordinate_system,
            flip_x=config.flip_x,
            flip_y=config.flip_y,
            blocksize=transcript_blocksize,
            sample_id=sample_id,
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
    products: tuple[str, ...],
    output_names: Mapping[str, str],
) -> None:
    """Reject exact and case-insensitive generated-name collisions.

    SpatialData requires element names to be unique, but deliberately permits
    several elements to reference the same coordinate-system name: that is how
    aligned image, labels, and points elements share a spatial frame. This
    reader therefore plans element and coordinate-system namespaces separately.

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
    for sample in samples:
        for mosaic in sample.preview.mosaics:
            for product in products:
                element_name = f"{sample.sample_id}_{output_names[product]}_mosaic_{mosaic.mosaic}"
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


def _validate_planned_panels(samples: tuple[_PreparedCosmxSample, ...]) -> None:
    """Detect a truncated content-hash collision before writing any points."""
    panels: dict[str, dict[str, object]] = {}
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
