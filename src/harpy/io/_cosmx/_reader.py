from __future__ import annotations

import shutil
import uuid
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from loguru import logger as log
from spatialdata import SpatialData, read_zarr
from spatialdata.models.models import ScaleFactors_t

from harpy import __version__
from harpy._metadata import _HARPY_METADATA_KEY, _PROVENANCE_METADATA_KEY, _harpy_metadata
from harpy.io._cosmx._discovery import _discover_cosmx
from harpy.io._cosmx._images import _add_morphology_images
from harpy.io._cosmx._labels import _add_compartment_labels, _add_instance_labels
from harpy.io._cosmx._models import _CosmxPreview, _MosaicMode
from harpy.io._cosmx._preview import _preview_cosmx
from harpy.io._cosmx._transcripts import _add_transcript_points


def cosmx(
    path: str | Path,
    output: str | Path,
    *,
    fovs: Sequence[int] | None = None,
    channels: Sequence[str] | None = None,
    mosaic_mode: _MosaicMode = "spatial_groups",
    adjacency_tolerance_px: int | None = None,
    morphology: bool = True,
    instance_labels: bool = True,
    compartment_labels: bool = True,
    transcripts: bool = True,
    output_image_name: str = "morphology_image",
    output_instance_labels_name: str = "instance_labels",
    output_compartment_labels_name: str = "compartment_labels",
    output_points_name: str = "transcripts",
    coordinate_system: str = "global",
    flip_x: bool = True,
    flip_y: bool = False,
    image_chunks: tuple[int, int, int] = (1, 1024, 1024),
    labels_chunks: tuple[int, int] = (1024, 1024),
    raster_scale_factors: ScaleFactors_t | None = None,
    transcript_blocksize: str | int = "64MB",
    overwrite: bool = False,
) -> SpatialData:
    """Read a decoded CosMx run into a backed SpatialData Zarr store.

    Positioned FOVs for which morphology, instance labels, compartment labels,
    and transcripts are all available are organized according to
    ``mosaic_mode``. Every enabled modality uses the same resulting mosaics,
    orientation, and mosaic-specific pixel and micron coordinate systems.
    Raster mosaics and transcript tables are constructed out of core by their
    modality-specific readers.

    The complete output is transactional. Data are written directly into a
    temporary sibling Zarr store and published only after that store can be
    reopened. ``overwrite=True`` replaces a complete CosMx store; it never
    merges elements into an existing SpatialData object.

    Parameters
    ----------
    path
        Decoded CosMx run directory, or a parent containing exactly one decoded
        run.
    output
        Destination for the backed SpatialData Zarr store. In-memory output is
        not supported.
    fovs
        Optional FOV numbers to consider. Selection is deterministic and still
        requires the common source availability used by every modality.
    channels
        Optional morphology channel IDs or unambiguous biological names. The
        acquisition order is preserved. Ignored when morphology is disabled.
    mosaic_mode
        ``"spatial_groups"`` creates separate adjacency-derived mosaics.
        ``"single"`` deliberately places every included FOV in one potentially
        sparse bounding canvas.
    adjacency_tolerance_px
        Maximum horizontal or vertical FOV gap, in pixels, bridged during
        mosaic grouping. ``None`` uses two percent of the smaller tile
        dimension when ``mosaic_mode="spatial_groups"``. Single-mosaic mode
        does not perform adjacency grouping and therefore requires this value
        to remain ``None``; supplying an integer together with
        ``mosaic_mode="single"`` raises a ``ValueError`` rather than silently
        ignoring it.
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
    coordinate_system
        Base coordinate-system name. Mosaic ``n`` receives ``<base>_n`` and
        ``<base>_n_micron`` coordinate systems.
    flip_x, flip_y
        Dataset-wide local-axis orientation applied consistently to rasters and
        transcript coordinates.
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

        harpy
        ├── metadata_version
        ├── provenance
        ├── images
        │   └── <image element name>
        ├── labels
        │   └── <labels element name>
        ├── points
        │   └── <points element name>
        └── feature_panels
            └── <shared panel name>

    ``provenance`` records the reader and its Harpy version together with the
    mosaic construction settings. Machine-specific source paths and FOV
    selection diagnostics are not persisted. The ``images``, ``labels``, and
    ``points`` mappings are keyed by exact SpatialData element names. Their
    records contain the authoritative source FOV membership, pre-group/source
    origin, orientation, and pixel size. Images additionally describe retained
    channels; instance labels describe their ID encoding; compartment labels
    describe their semantic categories.

    When an authoritative run-level plex is available, ``feature_panels``
    stores its shared target-to-class relation, including panel targets with
    zero detections. Each associated points record references that shared panel
    by name. If no plex is available, transcript ingestion still succeeds and
    neither the shared panel nor its points reference is written.

    Notes
    -----
    This reader does not create cell-boundary shapes, tables, cell statistics,
    or expression aggregates. Instances meeting at FOV boundaries are not
    merged.
    """
    enabled = {
        "morphology": morphology,
        "instance_labels": instance_labels,
        "compartment_labels": compartment_labels,
        "transcripts": transcripts,
    }
    if not any(enabled.values()):
        raise ValueError("CosMx ingestion requires at least one enabled modality.")

    manifest = _discover_cosmx(path)
    requested_fovs = manifest.fov_ids if fovs is None else tuple(sorted({int(fov) for fov in fovs}))
    preview = _preview_cosmx(
        manifest,
        fovs=requested_fovs,
        mosaic_mode=mosaic_mode,
        adjacency_tolerance_px=adjacency_tolerance_px,
    )
    if not preview.mosaics:
        raise ValueError("CosMx ingestion found no common positioned FOVs for the requested selection.")

    output_path = Path(output).expanduser().resolve()
    _validate_source_output_paths(manifest.root, output_path)
    _validate_planned_names(
        preview,
        morphology=(morphology, output_image_name),
        instance_labels=(instance_labels, output_instance_labels_name),
        compartment_labels=(compartment_labels, output_compartment_labels_name),
        transcripts=(transcripts, output_points_name),
    )

    output_exists = output_path.exists()
    if output_exists and not overwrite:
        raise FileExistsError(f"CosMx output already exists: {output_path}")
    if output_exists:
        _validate_replaceable_output(output_path)

    for message in preview.diagnostics:
        log.info(message)
    if preview.excluded_fovs:
        log.warning(f"CosMx output excludes FOVs {list(preview.excluded_fovs)}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = _unique_sibling(output_path, purpose="staging")
    try:
        SpatialData().write(staging)
        sdata = read_zarr(staging)

        if morphology:
            started = perf_counter()
            sdata = _add_morphology_images(
                sdata,
                preview,
                channels=channels,
                output_image_name=output_image_name,
                coordinate_system=coordinate_system,
                flip_x=flip_x,
                flip_y=flip_y,
                chunks=image_chunks,
                scale_factors=raster_scale_factors,
                overwrite=False,
            )
            log.info(f"Wrote CosMx morphology images in {perf_counter() - started:.2f} seconds.")

        if instance_labels:
            started = perf_counter()
            sdata = _add_instance_labels(
                sdata,
                preview,
                output_labels_name=output_instance_labels_name,
                coordinate_system=coordinate_system,
                flip_x=flip_x,
                flip_y=flip_y,
                chunks=labels_chunks,
                scale_factors=raster_scale_factors,
                overwrite=False,
            )
            log.info(f"Wrote CosMx instance labels in {perf_counter() - started:.2f} seconds.")

        if compartment_labels:
            started = perf_counter()
            sdata = _add_compartment_labels(
                sdata,
                preview,
                output_labels_name=output_compartment_labels_name,
                coordinate_system=coordinate_system,
                flip_x=flip_x,
                flip_y=flip_y,
                chunks=labels_chunks,
                scale_factors=raster_scale_factors,
                overwrite=False,
            )
            log.info(f"Wrote CosMx compartment labels in {perf_counter() - started:.2f} seconds.")

        if transcripts:
            started = perf_counter()
            sdata = _add_transcript_points(
                sdata,
                preview,
                output_points_name=output_points_name,
                coordinate_system=coordinate_system,
                flip_x=flip_x,
                flip_y=flip_y,
                blocksize=transcript_blocksize,
                overwrite=False,
            )
            log.info(f"Wrote CosMx transcript points in {perf_counter() - started:.2f} seconds.")

        attrs = deepcopy(sdata.attrs)
        harpy_metadata = _harpy_metadata(attrs)
        harpy_metadata[_PROVENANCE_METADATA_KEY] = {
            "reader": "cosmx",
            "reader_version": __version__,
            "mosaic_mode": preview.mosaic_mode,
            "adjacency_tolerance_px": preview.adjacency_tolerance_px,
        }
        sdata.attrs = attrs
        sdata.write_attrs()

        # Reopening validates the final-format staging store before publication.
        read_zarr(staging)
        return _publish_staging_store(staging, output_path, replace_existing=output_exists)
    except Exception:
        _remove_generated_path(staging)
        raise


def _validate_source_output_paths(source: Path, output: Path) -> None:
    """Prevent the reader from writing into, over, or around its source run."""
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError(f"CosMx source and output paths must not equal or contain one another: {source}, {output}")


def _validate_planned_names(
    preview: _CosmxPreview,
    **modalities: tuple[bool, str],
) -> None:
    """Reject element-name collisions that are visible only across modalities."""
    owners: dict[str, str] = {}
    for modality, (is_enabled, base_name) in modalities.items():
        if not is_enabled:
            continue
        for mosaic in preview.mosaics:
            element_name = f"{base_name}_mosaic_{mosaic.mosaic}"
            previous = owners.setdefault(element_name, modality)
            if previous != modality:
                raise ValueError(f"CosMx output element {element_name!r} is planned by both {previous} and {modality}.")


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
