from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from loguru import logger as log

from harpy.io._cosmx._constants import (
    _DEFAULT_PIXEL_SIZE_UM,
    _FOV_DIR_RE,
    _FOV_FILE_RE,
    _MORPHOLOGY_FILE_RE,
    _MORPHOLOGY_GEOMETRY_INVARIANT_KEYS,
    _MORPHOLOGY_IMAGE_INVARIANT_KEYS,
    _PLEX_FILE_RE,
    _TIFF_SUFFIXES,
    _CosmxKeys,
)
from harpy.io._cosmx._models import (
    _COMPARTMENT_LABELS_PRODUCT,
    _INSTANCE_LABELS_PRODUCT,
    _MORPHOLOGY_PRODUCT,
    _PRODUCTS,
    _TRANSCRIPTS_PRODUCT,
    _CosmxChannel,
    _CosmxFeatureClass,
    _CosmxFeaturePanel,
    _CosmxFovFiles,
    _CosmxFovPosition,
    _CosmxManifest,
    _CosmxRunMetadata,
    _MorphologyPosition,
)


def _is_decoded_cosmx(path: str | Path) -> bool:
    path = Path(path)
    return (path / _CosmxKeys.CELL_STATS_DIR / _CosmxKeys.MORPHOLOGY_DIR).is_dir() and (
        path / _CosmxKeys.ANALYSIS_RESULTS_DIR
    ).is_dir()


def _resolve_decoded_cosmx_root(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"CosMx path does not exist: {path}")
    if _is_decoded_cosmx(path):
        return path

    candidates = sorted(
        candidate.parent.parent
        for candidate in path.rglob(_CosmxKeys.MORPHOLOGY_DIR)
        if candidate.is_dir()
        and candidate.parent.name == _CosmxKeys.CELL_STATS_DIR
        and (candidate.parent.parent / _CosmxKeys.ANALYSIS_RESULTS_DIR).is_dir()
    )
    if not candidates:
        raise ValueError(
            f"Could not find a decoded CosMx run below {path}. Expected "
            f"{_CosmxKeys.CELL_STATS_DIR}/{_CosmxKeys.MORPHOLOGY_DIR} and "
            f"{_CosmxKeys.ANALYSIS_RESULTS_DIR} directories."
        )
    if len(candidates) > 1:
        formatted = "\n".join(f"- {candidate}" for candidate in candidates)
        raise ValueError(f"Found multiple decoded CosMx runs below {path}; pass one run explicitly:\n{formatted}")
    return candidates[0]


def _discover_cosmx(path: str | Path, *, products: tuple[str, ...] = _PRODUCTS) -> _CosmxManifest:
    products = tuple(products)
    if not products or len(set(products)) != len(products):
        raise ValueError(f"CosMx discovery products must be non-empty and unique, found {products}.")
    unknown_products = set(products) - set(_PRODUCTS)
    if unknown_products:
        raise ValueError(
            f"Unknown CosMx discovery products {sorted(unknown_products)}; expected a subset of {_PRODUCTS}."
        )
    root = _resolve_decoded_cosmx_root(path)
    discovered: dict[int, dict[str, Path]] = {}

    def assign(fov: int, product: str, file_path: Path) -> None:
        file_path = file_path.resolve()
        existing = discovered.setdefault(fov, {}).get(product)
        if existing is not None and existing != file_path:
            raise ValueError(f"Duplicate {product} files for FOV {fov}: {existing} and {file_path}")
        discovered[fov][product] = file_path

    morphology_dir = root / _CosmxKeys.CELL_STATS_DIR / _CosmxKeys.MORPHOLOGY_DIR
    for file_path in sorted(morphology_dir.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in _TIFF_SUFFIXES:
            continue
        match = _MORPHOLOGY_FILE_RE.match(file_path.name)
        if match is not None:
            assign(int(match.group("fov")), _MORPHOLOGY_PRODUCT, file_path)

    fov_root = root / _CosmxKeys.CELL_STATS_DIR
    for fov_dir in sorted(fov_root.iterdir()):
        if not fov_dir.is_dir() or (match := _FOV_DIR_RE.match(fov_dir.name)) is None:
            continue
        directory_fov = int(match.group(1))
        for file_path in sorted(fov_dir.iterdir()):
            if not file_path.is_file():
                continue
            product = _label_product(file_path.name)
            if product is None or product not in products:
                continue
            filename_fov = _fov_from_name(file_path.name)
            if filename_fov is None:
                raise ValueError(f"Could not determine the FOV from relevant file {file_path}.")
            if filename_fov != directory_fov:
                raise ValueError(
                    f"FOV directory/file mismatch: {fov_dir.name} contains {file_path.name} for FOV {filename_fov}."
                )
            assign(directory_fov, product, file_path)

    if _TRANSCRIPTS_PRODUCT in products:
        analysis_results = root / _CosmxKeys.ANALYSIS_RESULTS_DIR
        for file_path in sorted(analysis_results.rglob(f"*{_CosmxKeys.TRANSCRIPT_SUFFIX}")):
            if not file_path.is_file():
                continue
            fov = _fov_from_path(file_path)
            if fov is None:
                raise ValueError(f"Could not determine the FOV from transcript file {file_path}.")
            assign(fov, _TRANSCRIPTS_PRODUCT, file_path)

    run_metadata, morphology_positions = _read_raster_metadata(discovered, products=products)
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

    fov_records = tuple(
        _CosmxFovFiles(
            fov=fov,
            **{product: discovered.get(fov, {}).get(product) for product in _PRODUCTS},
        )
        for fov in all_fovs
    )
    diagnostics = _availability_diagnostics(fov_records, positions, products=products)
    feature_panel = _discover_feature_panel(root) if _TRANSCRIPTS_PRODUCT in products else None

    return _CosmxManifest(
        root=root,
        fovs=fov_records,
        positions=tuple(sorted(positions.values(), key=lambda item: item.fov)),
        run=run_metadata,
        diagnostics=tuple(diagnostics),
        feature_panel=feature_panel,
    )


def _discover_feature_panel(root: Path) -> _CosmxFeaturePanel | None:
    """Discover and parse the optional run-level plex exactly once."""
    candidates = sorted(
        path for path in root.iterdir() if path.is_file() and _PLEX_FILE_RE.fullmatch(path.name) is not None
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(f"Found multiple CosMx plex files: {candidates}.")
    return _read_feature_panel(candidates[0])


def _read_feature_panel(path: Path) -> _CosmxFeaturePanel:
    """Read the authoritative feature-to-class relation from a CosMx plex."""
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        header = tuple(reader.fieldnames or ())
        if not header:
            raise ValueError(f"CosMx plex file is empty: {path}.")
        if any(not column for column in header) or len(set(header)) != len(header):
            raise ValueError(f"CosMx plex file has invalid column names {header}: {path}.")
        missing = sorted({_CosmxKeys.PLEX_FEATURE_COLUMN, _CosmxKeys.PLEX_CLASS_COLUMN} - set(header))
        if missing:
            raise ValueError(f"CosMx plex file {path} is missing required columns {missing}.")

        class_by_feature: dict[str, str] = {}
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"CosMx plex file {path} has too many fields on line {line_number}.")
            feature = row[_CosmxKeys.PLEX_FEATURE_COLUMN]
            feature_class = row[_CosmxKeys.PLEX_CLASS_COLUMN]
            if feature is None or not feature or feature != feature.strip():
                raise ValueError(
                    f"CosMx plex file {path} has an empty or untrimmed "
                    f"{_CosmxKeys.PLEX_FEATURE_COLUMN} on line {line_number}."
                )
            if feature_class is None or not feature_class or feature_class != feature_class.strip():
                raise ValueError(
                    f"CosMx plex file {path} has an empty or untrimmed "
                    f"{_CosmxKeys.PLEX_CLASS_COLUMN} on line {line_number}."
                )
            previous = class_by_feature.get(feature)
            if previous is not None:
                raise ValueError(
                    f"CosMx plex feature {feature!r} occurs more than once with classes "
                    f"{previous!r} and {feature_class!r}: {path}."
                )
            class_by_feature[feature] = feature_class

    if not class_by_feature:
        raise ValueError(f"CosMx plex file contains no features: {path}.")

    features_by_class: dict[str, list[str]] = {}
    for feature, feature_class in class_by_feature.items():
        features_by_class.setdefault(feature_class, []).append(feature)
    classes = tuple(
        _CosmxFeatureClass(name=feature_class, features=tuple(sorted(features)))
        for feature_class, features in sorted(features_by_class.items())
    )
    return _CosmxFeaturePanel(
        feature_key=_CosmxKeys.FEATURE_KEY,
        feature_class_key=_CosmxKeys.FEATURE_CLASS_KEY,
        classes=classes,
    )


def _label_product(name: str) -> str | None:
    lower = name.lower()
    if lower.startswith(_CosmxKeys.INSTANCE_LABEL_PREFIX) and lower.endswith(_TIFF_SUFFIXES):
        return _INSTANCE_LABELS_PRODUCT
    if lower.startswith(_CosmxKeys.COMPARTMENT_LABEL_PREFIX) and lower.endswith(_TIFF_SUFFIXES):
        return _COMPARTMENT_LABELS_PRODUCT
    return None


def _fov_from_name(name: str) -> int | None:
    match = _FOV_FILE_RE.search(name)
    return int(match.group(1)) if match is not None else None


def _fov_from_path(path: Path) -> int | None:
    for part in reversed(path.parts):
        if (match := _FOV_DIR_RE.match(part)) is not None:
            return int(match.group(1))
    return _fov_from_name(path.name)


def _read_raster_metadata(
    discovered: dict[int, dict[str, Path]],
    *,
    products: tuple[str, ...] = _PRODUCTS,
) -> tuple[_CosmxRunMetadata, dict[int, _MorphologyPosition]]:
    """Read the raster headers needed to describe and position a CosMx run.

    Morphology TIFFs provide both run-wide metadata and the stage position of
    each imaged FOV. The first morphology TIFF is used as the reference; all
    remaining morphology TIFFs must agree with it on the declared FOV count,
    pixel size, tile dimensions, and TIFF shape because every modality uses
    that geometry. Channel order and TIFF dtype are additionally validated only
    when morphology output is enabled. Instance and compartment label TIFF
    headers are checked against the morphology tile shape and within their
    respective label families only when those outputs are enabled. Pixel arrays
    are not loaded. The source ``OrigTimeStamp`` is retained verbatim only when
    every morphology TIFF provides the same non-empty value; otherwise it is
    omitted without making the run unreadable.

    Parameters
    ----------
    discovered
        Relevant product paths grouped by FOV.

    Returns
    -------
    _CosmxRunMetadata
        Run-wide channel, scale, raster-shape, and dtype metadata.
    dict[int, _MorphologyPosition]
        Morphology stage positions keyed by FOV number.

    Raises
    ------
    ValueError
        If no morphology TIFF is available, required morphology metadata is
        missing, or raster metadata is inconsistent within a product family.
    """
    morphology_files = [
        (fov, files[_MORPHOLOGY_PRODUCT]) for fov, files in discovered.items() if _MORPHOLOGY_PRODUCT in files
    ]
    if not morphology_files:
        raise ValueError("Decoded CosMx layout has no morphology TIFFs; run metadata and positions are unavailable.")

    reference: dict[str, Any] | None = None
    reference_shape: tuple[int, ...] | None = None
    reference_dtype: str | None = None
    positions: dict[int, _MorphologyPosition] = {}
    acquisition_timestamp_values: list[object] = []
    # Every modality needs geometry from the morphology TIFFs, but channel
    # order and dtype matter only when morphology images are being ingested.
    morphology_images_enabled = _MORPHOLOGY_PRODUCT in products

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

        acquisition_timestamp_values.append(metadata.get(_CosmxKeys.ACQUISITION_TIMESTAMP))

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
            _validate_morphology_image_metadata(
                reference,
                metadata,
                reference_shape=reference_shape,
                shape=shape,
                reference_dtype=reference_dtype,
                dtype=dtype,
                file_path=file_path,
                validate_channels_and_dtype=morphology_images_enabled,
            )

    assert reference is not None
    assert reference_shape is not None
    assert reference_dtype is not None
    if len(reference_shape) != 3:
        raise ValueError(f"Expected morphology TIFF axes compatible with (c, y, x), found {reference_shape}.")

    if morphology_images_enabled:
        channel_order = _channel_order(reference.get("ChannelOrder"))
        if len(channel_order) != reference_shape[0]:
            raise ValueError(
                f"ChannelOrder {channel_order} does not match morphology plane count {reference_shape[0]}."
            )
        channels = _channel_metadata(reference, channel_order)
        morphology_dtype = reference_dtype
    else:
        channels = ()
        morphology_dtype = None
    tile_shape = (reference_shape[-2], reference_shape[-1])
    declared_shape = (
        int(reference.get("ImRows", tile_shape[0])),
        int(reference.get("ImCols", tile_shape[1])),
    )
    if declared_shape != tile_shape:
        raise ValueError(f"Morphology ImRows/ImCols {declared_shape} disagree with TIFF shape {tile_shape}.")

    instance_labels_dtype = (
        _validate_label_metadata(
            discovered,
            product=_INSTANCE_LABELS_PRODUCT,
            expected_shape=tile_shape,
        )
        if _INSTANCE_LABELS_PRODUCT in products
        else None
    )
    compartment_labels_dtype = (
        _validate_label_metadata(
            discovered,
            product=_COMPARTMENT_LABELS_PRODUCT,
            expected_shape=tile_shape,
        )
        if _COMPARTMENT_LABELS_PRODUCT in products
        else None
    )

    return (
        _CosmxRunMetadata(
            declared_fov_count=int(reference["NFov"]) if reference.get("NFov") is not None else None,
            acquisition_timestamp=_consistent_acquisition_timestamp(acquisition_timestamp_values),
            channels=channels,
            pixel_size_um=_pixel_size_um(reference),
            tile_shape=tile_shape,
            morphology_dtype=morphology_dtype,
            instance_labels_dtype=instance_labels_dtype,
            compartment_labels_dtype=compartment_labels_dtype,
        ),
        positions,
    )


def _consistent_acquisition_timestamp(values: list[object]) -> str | None:
    """Return one verbatim source timestamp, or omit ambiguous optional metadata."""
    valid = [value for value in values if isinstance(value, str) and value and value == value.strip()]
    if len(valid) == len(values) and len(set(valid)) == 1:
        return valid[0]
    if any(value is not None for value in values):
        log.warning(
            "CosMx morphology TIFFs do not provide one consistent non-empty OrigTimeStamp; "
            "acquisition_timestamp metadata will be omitted."
        )
    return None


def _validate_morphology_image_metadata(
    reference: dict[str, Any],
    metadata: dict[str, Any],
    *,
    reference_shape: tuple[int, ...],
    shape: tuple[int, ...],
    reference_dtype: str,
    dtype: str,
    file_path: Path,
    validate_channels_and_dtype: bool,
) -> None:
    invariant_keys = _MORPHOLOGY_GEOMETRY_INVARIANT_KEYS
    if validate_channels_and_dtype:
        invariant_keys += _MORPHOLOGY_IMAGE_INVARIANT_KEYS
    for key in invariant_keys:
        if reference.get(key) != metadata.get(key):
            raise ValueError(
                f"Contradictory morphology metadata for {key}: {reference.get(key)!r} versus "
                f"{metadata.get(key)!r} in {file_path}."
            )
    if reference_shape != shape or (validate_channels_and_dtype and reference_dtype != dtype):
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


def _validate_label_metadata(
    discovered: dict[int, dict[str, Path]],
    *,
    product: str,
    expected_shape: tuple[int, int],
) -> str | None:
    """Validate label TIFF metadata across a product's FOVs.

    Parameters
    ----------
    discovered
        Relevant product paths grouped by FOV.
    product
        Label product identifier to validate.
    expected_shape
        Required ``(height, width)`` matching the morphology tiles.

    Returns
    -------
    str or None
        The common NumPy dtype name, or ``None`` when the product is absent.

    Raises
    ------
    ValueError
        If a label TIFF is not unsigned integer, has the wrong shape, or has a
        dtype that differs from another FOV in the family.
    """
    reference_dtype = None
    for fov, files in sorted(discovered.items()):
        if product not in files:
            continue
        with tifffile.TiffFile(files[product]) as tif:
            shape = tuple(int(value) for value in tif.series[0].shape)
            dtype = np.dtype(tif.series[0].dtype).name
        if np.dtype(dtype).kind != "u":
            raise ValueError(
                f"CosMx {product.replace('_', ' ')} must use an unsigned integer dtype, found {dtype} for FOV {fov}."
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
    *,
    products: tuple[str, ...] = _PRODUCTS,
) -> list[str]:
    diagnostics = []
    fov_ids = {item.fov for item in fovs}
    for product in products:
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
