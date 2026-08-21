from __future__ import annotations

import numpy as np

from harpy.io._cosmx._models import (
    _PRODUCTS,
    _CosmxComponentGeometry,
    _CosmxComponentSizeEstimate,
    _CosmxFovPosition,
    _CosmxManifest,
    _CosmxPreview,
)


def _preview_cosmx(
    manifest: _CosmxManifest,
    *,
    fovs: tuple[int, ...] | list[int] | set[int] | None = None,
) -> _CosmxPreview:
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
    components = _component_geometry(
        positions={fov: positions_by_fov[fov] for fov in included},
        tile_shape=manifest.run.tile_shape,
    )
    estimates = _estimate_component_sizes(
        components,
        image_dtype=manifest.run.morphology_dtype,
        channel_count=len(manifest.run.channels),
        cell_labels_dtype=manifest.run.cell_labels_dtype,
        compartment_labels_dtype=manifest.run.compartment_labels_dtype,
    )

    diagnostics = list(manifest.diagnostics)
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
        components=components,
        estimates=estimates,
        diagnostics=tuple(diagnostics),
    )


def _component_geometry(
    *,
    positions: dict[int, _CosmxFovPosition],
    tile_shape: tuple[int, int],
) -> tuple[_CosmxComponentGeometry, ...]:
    if not positions:
        return ()

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
            if (overlap_x > 0 and overlap_y >= 0) or (overlap_y > 0 and overlap_x >= 0):
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

    components = []
    for component, group in enumerate(ordered_groups, start=1):
        origin_x = min(positions[fov].x_px for fov in group)
        origin_y = min(positions[fov].y_px for fov in group)
        max_x = max(positions[fov].x_px + width for fov in group)
        max_y = max(positions[fov].y_px + height for fov in group)
        components.append(
            _CosmxComponentGeometry(
                component=component,
                fovs=group,
                origin_x_px=origin_x,
                origin_y_px=origin_y,
                shape=(max_y - origin_y, max_x - origin_x),
            )
        )
    return tuple(components)


def _estimate_component_sizes(
    components: tuple[_CosmxComponentGeometry, ...],
    *,
    image_dtype: str,
    channel_count: int,
    cell_labels_dtype: str | None,
    compartment_labels_dtype: str | None,
) -> tuple[_CosmxComponentSizeEstimate, ...]:
    if not components:
        return ()
    if cell_labels_dtype is None or compartment_labels_dtype is None:
        raise ValueError("Cannot estimate label output sizes without relevant label dtypes.")

    image_itemsize = np.dtype(image_dtype).itemsize
    max_fov = max(fov for component in components for fov in component.fovs)
    cell_labels_itemsize = _remapped_cell_dtype(cell_labels_dtype, max_fov).itemsize
    compartment_itemsize = np.dtype(compartment_labels_dtype).itemsize
    return tuple(
        _CosmxComponentSizeEstimate(
            component=component.component,
            image_nbytes=component.shape[0] * component.shape[1] * channel_count * image_itemsize,
            cell_labels_nbytes=component.shape[0] * component.shape[1] * cell_labels_itemsize,
            compartment_labels_nbytes=component.shape[0] * component.shape[1] * compartment_itemsize,
        )
        for component in components
    )


def _remapped_cell_dtype(source_dtype: str, max_fov: int) -> np.dtype:
    source = np.dtype(source_dtype)
    base = 1 << (source.itemsize * 8)
    max_global_id = (max_fov - 1) * base + (base - 1)
    output = np.dtype(np.min_scalar_type(max_global_id))
    if output.kind != "u":
        raise ValueError(f"Could not select an unsigned global cell-ID dtype for maximum ID {max_global_id}.")
    return output
