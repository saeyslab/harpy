from __future__ import annotations

from harpy.io._cosmx._models import (
    _PRODUCTS,
    _CosmxFovPosition,
    _CosmxManifest,
    _CosmxMosaicGeometry,
    _CosmxPreview,
)

_DEFAULT_ADJACENCY_TOLERANCE_FRACTION = 0.02


def _preview_cosmx(
    manifest: _CosmxManifest,
    *,
    fovs: tuple[int, ...] | list[int] | set[int] | None = None,
) -> _CosmxPreview:
    """Select usable FOVs and organize them into spatial mosaic groups.

    A FOV is included only when it was requested, has a known morphology
    position, and has morphology, instance-label, compartment-label, and
    transcript files. Included FOVs are grouped by spatial adjacency into
    independent mosaic geometries; they are not placed in one shared global
    coordinate system. The function also records excluded and unpositioned
    FOVs and appends selection and grouping diagnostics. The returned preview
    calculates aggregate raster-size estimates from its mosaic geometries and
    validates that the planned instance-ID encoding fits in ``uint32``. No
    raster pixels or transcript contents are loaded.

    Parameters
    ----------
    manifest
        Discovered files, positions, and validated run metadata.
    fovs
        Optional FOV subset to consider. By default, all manifest FOVs are
        considered.

    Returns
    -------
    _CosmxPreview
        The validated FOV selection, mosaic geometries, aggregate size
        estimates, and diagnostics.

    Raises
    ------
    ValueError
        If a requested FOV is absent from the manifest or the planned raster
        outputs and instance-ID encoding cannot be represented consistently.
    """
    all_fovs = set(manifest.fov_ids)
    requested = all_fovs if fovs is None else {int(fov) for fov in fovs}
    unknown = requested - all_fovs
    if unknown:
        raise ValueError(f"Requested FOVs are not present in the manifest: {sorted(unknown)}")

    positions_by_fov = manifest.positions_by_fov
    positioned = set(positions_by_fov)
    common = set(positioned)
    for product in _PRODUCTS:
        common &= set(manifest.available_fovs(product))
    included = tuple(sorted(requested & common))
    excluded = tuple(sorted(all_fovs - set(included)))
    unpositioned = tuple(sorted(all_fovs - positioned))
    adjacency_tolerance_px = _default_adjacency_tolerance_px(manifest.run.tile_shape)
    mosaics = _mosaic_geometries(
        positions={fov: positions_by_fov[fov] for fov in included},
        tile_shape=manifest.run.tile_shape,
        adjacency_tolerance_px=adjacency_tolerance_px,
    )
    diagnostics = list(manifest.diagnostics)
    if mosaics:
        pixel_unit = "pixel" if adjacency_tolerance_px == 1 else "pixels"
        diagnostics.append(
            f"Grouped {len(included)} positioned FOV(s) into {len(mosaics)} spatial mosaic group(s) using an "
            f"adjacency tolerance of {adjacency_tolerance_px} {pixel_unit}."
        )
    if excluded:
        diagnostics.append(
            f"Selected {len(included)} common positioned FOV(s); excluded FOVs {list(excluded)} without reading "
            "transcript contents."
        )

    return _CosmxPreview(
        manifest=manifest,
        included_fovs=included,
        excluded_fovs=excluded,
        unpositioned_fovs=unpositioned,
        mosaics=mosaics,
        diagnostics=tuple(diagnostics),
    )


def _mosaic_geometries(
    *,
    positions: dict[int, _CosmxFovPosition],
    tile_shape: tuple[int, int],
    adjacency_tolerance_px: int = 0,
) -> tuple[_CosmxMosaicGeometry, ...]:
    """Group spatially adjacent FOV rectangles into mosaic geometries.

    Each FOV is represented by a rectangle with its top-left corner given by
    ``positions`` and its common height and width given by ``tile_shape``. FOVs
    that overlap, share an edge, or have a small axis-aligned gap are placed in
    the same mosaic group. Connectivity is transitive: if FOV 1 is adjacent to
    FOV 2 and FOV 2 is adjacent to FOV 3, all three belong to one mosaic.

    The grouping can be pictured as follows::

        FOV 1  FOV 2  FOV 3          FOV 20 FOV 21       FOV 50
        ┌────┐ ┌────┐ ┌────┐         ┌────┐ ┌────┐       ┌────┐
        │    │ │    │ │    │         │    │ │    │       │    │
        └────┘ └────┘ └────┘         └────┘ └────┘       └────┘
        └──── mosaic group 1 ────┘   └─ mosaic group 2 ─┘ mosaic group 3

    Small horizontal or vertical gaps up to ``adjacency_tolerance_px`` are
    bridged only when the FOV rectangles overlap along the other axis. FOVs
    that meet only diagonally at a corner are therefore kept separate. Each
    returned geometry is the bounding rectangle of one group and may contain
    uncovered background pixels between its FOVs.

    Parameters
    ----------
    positions
        Mapping from FOV number to its top-left position in pre-group source
        pixel coordinates. Only FOVs present in this mapping participate in
        grouping.
    tile_shape
        Common FOV shape as ``(height, width)`` in pixels.
    adjacency_tolerance_px
        Maximum nonnegative horizontal or vertical gap, in pixels, that is
        treated as adjacency. The default of zero requires overlap or edge
        contact; the CosMx preview supplies its documented small-gap tolerance.

    Returns
    -------
    tuple of _CosmxMosaicGeometry
        Derived mosaic groups ordered deterministically by decreasing FOV
        count, then global position and FOV number. These groups describe
        spatial proximity and are not authoritative vendor ROI assignments.
    """
    if not positions:
        return ()
    if adjacency_tolerance_px < 0:
        raise ValueError(f"Adjacency tolerance must be nonnegative, found {adjacency_tolerance_px}.")

    included_fovs = tuple(sorted(positions))
    height, width = tile_shape
    parent = {fov: fov for fov in included_fovs}

    def find(fov: int) -> int:
        while parent[fov] != fov:
            parent[fov] = parent[parent[fov]]
            fov = parent[fov]
        return fov

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, left in enumerate(included_fovs):
        left_position = positions[left]
        for right in included_fovs[index + 1 :]:
            right_position = positions[right]
            overlap_x = min(left_position.x_px + width, right_position.x_px + width) - max(
                left_position.x_px, right_position.x_px
            )
            overlap_y = min(left_position.y_px + height, right_position.y_px + height) - max(
                left_position.y_px, right_position.y_px
            )
            if (overlap_x >= -adjacency_tolerance_px and overlap_y > 0) or (
                overlap_y >= -adjacency_tolerance_px and overlap_x > 0
            ):
                union(left, right)

    groups: dict[int, list[int]] = {}
    for fov in included_fovs:
        groups.setdefault(find(fov), []).append(fov)
    ordered_groups = sorted(
        (tuple(sorted(group)) for group in groups.values()),
        key=lambda group: (
            -len(group),
            min(positions[fov].y_px for fov in group),
            min(positions[fov].x_px for fov in group),
            group,
        ),
    )

    mosaics = []
    for mosaic, group in enumerate(ordered_groups, start=1):
        origin_x = min(positions[fov].x_px for fov in group)
        origin_y = min(positions[fov].y_px for fov in group)
        max_x = max(positions[fov].x_px + width for fov in group)
        max_y = max(positions[fov].y_px + height for fov in group)
        mosaics.append(
            _CosmxMosaicGeometry(
                mosaic=mosaic,
                fovs=group,
                origin_x_px=origin_x,
                origin_y_px=origin_y,
                shape=(max_y - origin_y, max_x - origin_x),
            )
        )
    return tuple(mosaics)


def _default_adjacency_tolerance_px(tile_shape: tuple[int, int]) -> int:
    return max(1, round(min(tile_shape) * _DEFAULT_ADJACENCY_TOLERANCE_FRACTION))
