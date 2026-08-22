from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from harpy.io._cosmx._models import (
    _PRODUCTS,
    _CosmxChannel,
    _CosmxFovFiles,
    _CosmxFovPosition,
    _CosmxManifest,
    _CosmxRunMetadata,
    _MorphologyPosition,
)

_FOV_DIR_RE = re.compile(r"^FOV0*(\d+)$", re.IGNORECASE)
_FOV_FILE_RE = re.compile(r"(?:^|[_-])(?:FOV|F)0*(\d+)(?=[_.-]|$)", re.IGNORECASE)
_MORPHOLOGY_FILE_RE = re.compile(
    r"^\d{8}_\d{6}_S\d+.*_F0*(?P<fov>\d+)\.(?:tif|tiff)$",
    re.IGNORECASE,
)
_TRANSCRIPT_SUFFIX = "target_call_coord.csv"
_DEFAULT_PIXEL_SIZE_UM = 0.120280945


def _is_decoded_cosmx(path: str | Path) -> bool:
    path = Path(path)
    return (path / "CellStatsDir" / "Morphology2D").is_dir() and (path / "AnalysisResults").is_dir()


def _resolve_decoded_cosmx_root(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"CosMx path does not exist: {path}")
    if _is_decoded_cosmx(path):
        return path

    candidates = sorted(
        candidate.parent.parent
        for candidate in path.rglob("Morphology2D")
        if candidate.is_dir()
        and candidate.parent.name == "CellStatsDir"
        and (candidate.parent.parent / "AnalysisResults").is_dir()
    )
    if not candidates:
        raise ValueError(
            f"Could not find a decoded CosMx run below {path}. Expected CellStatsDir/Morphology2D and "
            "AnalysisResults directories."
        )
    if len(candidates) > 1:
        formatted = "\n".join(f"- {candidate}" for candidate in candidates)
        raise ValueError(f"Found multiple decoded CosMx runs below {path}; pass one run explicitly:\n{formatted}")
    return candidates[0]


def _discover_cosmx(path: str | Path) -> _CosmxManifest:
    root = _resolve_decoded_cosmx_root(path)
    discovered: dict[int, dict[str, Path]] = {}

    def assign(fov: int, product: str, file_path: Path) -> None:
        file_path = file_path.resolve()
        existing = discovered.setdefault(fov, {}).get(product)
        if existing is not None and existing != file_path:
            raise ValueError(f"Duplicate {product} files for FOV {fov}: {existing} and {file_path}")
        discovered[fov][product] = file_path

    morphology_dir = root / "CellStatsDir" / "Morphology2D"
    for file_path in sorted(morphology_dir.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        match = _MORPHOLOGY_FILE_RE.match(file_path.name)
        if match is not None:
            assign(int(match.group("fov")), "morphology", file_path)

    cell_stats_dir = root / "CellStatsDir"
    for fov_dir in sorted(cell_stats_dir.iterdir()):
        if not fov_dir.is_dir() or (match := _FOV_DIR_RE.match(fov_dir.name)) is None:
            continue
        directory_fov = int(match.group(1))
        for file_path in sorted(fov_dir.iterdir()):
            if not file_path.is_file():
                continue
            product = _label_product(file_path.name)
            if product is None:
                continue
            filename_fov = _fov_from_name(file_path.name)
            if filename_fov is None:
                raise ValueError(f"Could not determine the FOV from relevant file {file_path}.")
            if filename_fov != directory_fov:
                raise ValueError(
                    f"FOV directory/file mismatch: {fov_dir.name} contains {file_path.name} for FOV {filename_fov}."
                )
            assign(directory_fov, product, file_path)

    analysis_results = root / "AnalysisResults"
    for file_path in sorted(analysis_results.rglob(f"*{_TRANSCRIPT_SUFFIX}")):
        if not file_path.is_file():
            continue
        fov = _fov_from_path(file_path)
        if fov is None:
            raise ValueError(f"Could not determine the FOV from transcript file {file_path}.")
        assign(fov, "transcripts", file_path)

    run_metadata, morphology_positions = _read_morphology_metadata(discovered)
    positions = _normalize_positions(
        morphology_positions,
        pixel_size_um=run_metadata.pixel_size_um,
    )

    declared_fovs = (
        set(range(1, run_metadata.declared_fov_count + 1)) if run_metadata.declared_fov_count is not None else set()
    )
    observed_fovs = set(discovered)
    unexpected_fovs = observed_fovs - declared_fovs if declared_fovs else set()
    if unexpected_fovs:
        raise ValueError(
            f"Files contain FOVs outside morphology NFov={run_metadata.declared_fov_count}: {sorted(unexpected_fovs)}"
        )
    all_fovs = sorted(declared_fovs | observed_fovs)

    instance_labels_dtype = _validate_label_family(
        discovered,
        product="instance_labels",
        expected_shape=run_metadata.tile_shape,
    )
    compartment_labels_dtype = _validate_label_family(
        discovered,
        product="compartment_labels",
        expected_shape=run_metadata.tile_shape,
    )
    run_metadata = replace(
        run_metadata,
        instance_labels_dtype=instance_labels_dtype,
        compartment_labels_dtype=compartment_labels_dtype,
    )

    fov_records = tuple(
        _CosmxFovFiles(
            fov=fov,
            **{product: discovered.get(fov, {}).get(product) for product in _PRODUCTS},
        )
        for fov in all_fovs
    )
    diagnostics = _availability_diagnostics(fov_records, positions)

    return _CosmxManifest(
        root=root,
        fovs=fov_records,
        positions=tuple(sorted(positions.values(), key=lambda item: item.fov)),
        run=run_metadata,
        diagnostics=tuple(diagnostics),
    )


def _label_product(name: str) -> str | None:
    lower = name.lower()
    if lower.startswith("celllabels_f") and lower.endswith((".tif", ".tiff")):
        return "instance_labels"
    if lower.startswith("compartmentlabels_f") and lower.endswith((".tif", ".tiff")):
        return "compartment_labels"
    return None


def _fov_from_name(name: str) -> int | None:
    match = _FOV_FILE_RE.search(name)
    return int(match.group(1)) if match is not None else None


def _fov_from_path(path: Path) -> int | None:
    for part in reversed(path.parts):
        if (match := _FOV_DIR_RE.match(part)) is not None:
            return int(match.group(1))
    return _fov_from_name(path.name)


def _read_morphology_metadata(
    discovered: dict[int, dict[str, Path]],
) -> tuple[_CosmxRunMetadata, dict[int, _MorphologyPosition]]:
    morphology_files = [(fov, files["morphology"]) for fov, files in discovered.items() if "morphology" in files]
    if not morphology_files:
        raise ValueError("Decoded CosMx layout has no morphology TIFFs; run metadata and positions are unavailable.")

    reference: dict[str, Any] | None = None
    reference_shape: tuple[int, ...] | None = None
    reference_dtype: str | None = None
    positions: dict[int, _MorphologyPosition] = {}

    for fov, file_path in sorted(morphology_files):
        with tifffile.TiffFile(file_path) as tif:
            description = tif.pages[0].description
            if not description:
                raise ValueError(f"Morphology TIFF has no JSON description: {file_path}")
            try:
                metadata = json.loads(description)
            except json.JSONDecodeError as error:
                raise ValueError(f"Morphology TIFF description is not valid JSON: {file_path}") from error
            if not isinstance(metadata, dict):
                raise ValueError(f"Morphology TIFF description is not a JSON object: {file_path}")
            shape = tuple(int(value) for value in tif.series[0].shape)
            dtype = np.dtype(tif.series[0].dtype).name

        metadata_fov = int(metadata.get("Fov", fov))
        if metadata_fov != fov:
            raise ValueError(
                f"Morphology filename FOV {fov} disagrees with TIFF metadata Fov={metadata_fov}: {file_path}"
            )
        if "X_mm" not in metadata or "Y_mm" not in metadata:
            raise ValueError(f"Morphology TIFF is missing X_mm/Y_mm stage coordinates: {file_path}")
        positions[fov] = _MorphologyPosition(
            fov=fov,
            x_mm=_finite_float(metadata["X_mm"], f"X_mm in {file_path}"),
            y_mm=_finite_float(metadata["Y_mm"], f"Y_mm in {file_path}"),
        )

        if reference is None:
            reference = metadata
            reference_shape = shape
            reference_dtype = dtype
        else:
            assert reference_shape is not None
            assert reference_dtype is not None
            _validate_morphology_metadata(
                reference,
                metadata,
                reference_shape=reference_shape,
                shape=shape,
                reference_dtype=reference_dtype,
                dtype=dtype,
                file_path=file_path,
            )

    assert reference is not None
    assert reference_shape is not None
    assert reference_dtype is not None
    if len(reference_shape) != 3:
        raise ValueError(f"Expected morphology TIFF axes compatible with (c, y, x), found {reference_shape}.")

    channel_order = _channel_order(reference.get("ChannelOrder"))
    if len(channel_order) != reference_shape[0]:
        raise ValueError(f"ChannelOrder {channel_order} does not match morphology plane count {reference_shape[0]}.")
    tile_shape = (reference_shape[-2], reference_shape[-1])
    declared_shape = (
        int(reference.get("ImRows", tile_shape[0])),
        int(reference.get("ImCols", tile_shape[1])),
    )
    if declared_shape != tile_shape:
        raise ValueError(f"Morphology ImRows/ImCols {declared_shape} disagree with TIFF shape {tile_shape}.")

    return (
        _CosmxRunMetadata(
            declared_fov_count=int(reference["NFov"]) if reference.get("NFov") is not None else None,
            channels=_channel_metadata(reference, channel_order),
            pixel_size_um=_pixel_size_um(reference),
            tile_shape=tile_shape,
            morphology_dtype=reference_dtype,
            instance_labels_dtype=None,
            compartment_labels_dtype=None,
        ),
        positions,
    )


def _validate_morphology_metadata(
    reference: dict[str, Any],
    metadata: dict[str, Any],
    *,
    reference_shape: tuple[int, ...],
    shape: tuple[int, ...],
    reference_dtype: str,
    dtype: str,
    file_path: Path,
) -> None:
    for key in ("NFov", "ChannelOrder", "ImPixelSize_nm", "ImRows", "ImCols"):
        if reference.get(key) != metadata.get(key):
            raise ValueError(
                f"Contradictory morphology metadata for {key}: {reference.get(key)!r} versus "
                f"{metadata.get(key)!r} in {file_path}."
            )
    if reference_shape != shape or reference_dtype != dtype:
        raise ValueError(
            f"Contradictory morphology raster metadata in {file_path}: {(shape, dtype)} versus "
            f"{(reference_shape, reference_dtype)}."
        )


def _channel_order(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        result = tuple(value)
    elif isinstance(value, list | tuple):
        result = tuple(str(item) for item in value)
    else:
        result = ()
    if not result:
        raise ValueError("Morphology metadata has no ChannelOrder.")
    return result


def _pixel_size_um(metadata: dict[str, Any]) -> float:
    value = (
        _finite_float(metadata["ImPixelSize_nm"], "ImPixelSize_nm") / 1000.0
        if metadata.get("ImPixelSize_nm") is not None
        else _DEFAULT_PIXEL_SIZE_UM
    )
    if value <= 0:
        raise ValueError(f"CosMx pixel size must be positive, found {value} micrometres.")
    return value


def _channel_metadata(metadata: dict[str, Any], channel_order: tuple[str, ...]) -> tuple[_CosmxChannel, ...]:
    reagents: dict[str, str] = {}
    morphology_kit = metadata.get("MorphologyKit") or {}
    for reagent in morphology_kit.get("MorphologyReagents") or []:
        fluorophore = reagent.get("Fluorophore") or {}
        channel_id = fluorophore.get("ChannelId")
        if channel_id is None:
            continue
        target = reagent.get("BiologicalTarget") or fluorophore.get("Name") or str(channel_id)
        reagents[str(channel_id)] = str(target)
    return tuple(
        _CosmxChannel(
            channel_id=channel_id,
            name=reagents.get(channel_id, channel_id),
        )
        for channel_id in channel_order
    )


def _validate_label_family(
    discovered: dict[int, dict[str, Path]],
    *,
    product: str,
    expected_shape: tuple[int, int],
) -> str | None:
    reference_dtype = None
    for fov, files in sorted(discovered.items()):
        if product not in files:
            continue
        with tifffile.TiffFile(files[product]) as tif:
            shape = tuple(int(value) for value in tif.series[0].shape)
            dtype = np.dtype(tif.series[0].dtype).name
        if product == "instance_labels" and np.dtype(dtype).kind != "u":
            raise ValueError(
                f"CosMx instance labels must use an unsigned integer dtype, found {dtype} for FOV {fov}."
            )
        if shape != expected_shape:
            raise ValueError(f"{product} for FOV {fov} has shape {shape}; expected {expected_shape}.")
        if reference_dtype is None:
            reference_dtype = dtype
        elif reference_dtype != dtype:
            raise ValueError(f"Contradictory {product} dtype for FOV {fov}: {dtype} versus {reference_dtype}.")
    return reference_dtype


def _normalize_positions(
    morphology: dict[int, _MorphologyPosition],
    *,
    pixel_size_um: float,
) -> dict[int, _CosmxFovPosition]:
    if not morphology:
        return {}
    origin_x_mm = min(item.x_mm for item in morphology.values())
    origin_y_mm = min(item.y_mm for item in morphology.values())
    return {
        fov: _CosmxFovPosition(
            fov=fov,
            x_px=round((position.x_mm - origin_x_mm) * 1000.0 / pixel_size_um),
            y_px=round((position.y_mm - origin_y_mm) * 1000.0 / pixel_size_um),
            x_mm=position.x_mm,
            y_mm=position.y_mm,
        )
        for fov, position in morphology.items()
    }


def _availability_diagnostics(
    fovs: tuple[_CosmxFovFiles, ...],
    positions: dict[int, _CosmxFovPosition],
) -> list[str]:
    diagnostics = []
    fov_ids = {item.fov for item in fovs}
    for product in _PRODUCTS:
        available = {item.fov for item in fovs if getattr(item, product) is not None}
        missing = sorted(fov_ids - available)
        if missing:
            diagnostics.append(f"{product}: available for {len(available)}/{len(fov_ids)} FOVs; missing {missing}.")
    unpositioned = sorted(fov_ids - set(positions))
    if unpositioned:
        diagnostics.append(f"No authoritative global position for FOVs {unpositioned}.")
    return diagnostics


def _finite_float(value: Any, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Expected a finite number for {context}, found {value!r}.")
    return result
