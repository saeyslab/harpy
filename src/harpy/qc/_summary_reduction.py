"""Reduction shared by the QC summaries of points and tables."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import dask


def _tree_reduce(tasks: Sequence, merge: Callable, *, fan_in: int = 8):
    """Build a lazy merge tree over delayed partial results that share one structure.

    The partial results come from independent parts of one computation: one
    per points partition in :func:`harpy.qc.summarize_points`, or the column
    sums of one row block of a table matrix in :func:`harpy.qc.summarize_table`.
    Each merge receives at most ``fan_in`` partial results. Repeat until one
    delayed result remains and return that task, without computing it here.
    This avoids collecting every part's partial result on the driver.

    For example, 16 parts with ``fan_in=8`` form this tree::

        Results of parts 0-7              Results of parts 8-15
                       |                                |
                       v                                v
                Merge -> result A                Merge -> result B
                       |                                |
                       +----------------+---------------+
                                        |
                                        v
                                Merge -> final result

    In that example, ``level`` is replaced after each round::

        # Initially: 16 tasks, one per part
        level = [part_0, ..., part_15]

        # After the first round: 2 merge tasks
        level = [merge_A, merge_B]

        # After the second round: 1 merge task
        level = [final_merge_task]

    The loop stops when only one task remains. ``return level[0]`` extracts
    that final task from the list; it does not select the first part's result.

    The arrows represent task dependencies, not computation performed by
    this function. Dask executes the graph when the returned task is computed.
    Every part's result and every merged result must have the same structure,
    so that the same ``merge`` works at each level. The final task therefore
    depends on all parts.
    """
    level = list(tasks)
    while len(level) > 1:
        next_level = []
        for start in range(0, len(level), fan_in):
            group = level[start : start + fan_in]
            merged_task = dask.delayed(merge)(*group)
            next_level.append(merged_task)
        level = next_level
    return level[0]
