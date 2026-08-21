from __future__ import annotations

import numpy as np

from harpy.io._cosmx._models import (
    _PRODUCTS,
    _CosmxComponentPreview,
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

    positioned = set(manifest.positions_by_fov)
    common = set(positioned)
    for product in _PRODUCTS:
        common &= set(manifest.available_fovs(product))
    included = tuple(sorted(requested & common))
    excluded = tuple(sorted(all_fovs - set(included)))
    unpositioned = tuple(sorted(all_fovs - positioned))
    components = _component_previews(manifest, included)

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
        diagnostics=tuple(diagnostics),
    )


def _component_previews(
    manifest: _CosmxManifest,
    included_fovs: tuple[int, ...],
) -> tuple[_CosmxComponentPreview, ...]:
    if not included_fovs:
        return ()
    if manifest.run.cell_labels_dtype is None or manifest.run.compartment_labels_dtype is None:
        raise ValueError("Cannot estimate label output sizes without relevant label dtypes.")

    positions = manifest.positions_by_fov
    height, width = manifest.run.tile_shape
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

    image_itemsize = np.dtype(manifest.run.morphology_dtype).itemsize
    cell_labels_itemsize = _remapped_cell_dtype(manifest.run.cell_labels_dtype, max(manifest.fov_ids)).itemsize
    compartment_itemsize = np.dtype(manifest.run.compartment_labels_dtype).itemsize
    previews = []
    for component, group in enumerate(ordered_groups, start=1):
        origin_x = min(positions[fov].x_px for fov in group)
        origin_y = min(positions[fov].y_px for fov in group)
        max_x = max(positions[fov].x_px + width for fov in group)
        max_y = max(positions[fov].y_px + height for fov in group)
        shape = (max_y - origin_y, max_x - origin_x)
        pixel_count = shape[0] * shape[1]
        previews.append(
            _CosmxComponentPreview(
                component=component,
                fovs=group,
                origin_x_px=origin_x,
                origin_y_px=origin_y,
                shape=shape,
                image_nbytes=pixel_count * len(manifest.run.channels) * image_itemsize,
                cell_labels_nbytes=pixel_count * cell_labels_itemsize,
                compartment_labels_nbytes=pixel_count * compartment_itemsize,
            )
        )
    return tuple(previews)


def _remapped_cell_dtype(source_dtype: str, max_fov: int) -> np.dtype:
    source = np.dtype(source_dtype)
    base = 1 << (source.itemsize * 8)
    max_global_id = (max_fov - 1) * base + (base - 1)
    output = np.dtype(np.min_scalar_type(max_global_id))
    if output.kind != "u":
        raise ValueError(f"Could not select an unsigned global cell-ID dtype for maximum ID {max_global_id}.")
    return output
