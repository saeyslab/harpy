"""Private storage infrastructure shared by Harpy readers and writers.

``_publication`` owns filesystem publication, backups, rollback and cleanup;
it does not serialize or reopen payloads. ``_spatialdata`` writes and reopens
whole SpatialData elements through SpatialData's I/O APIs. ``_anndata``
encodes AnnData components in Zarr and reopens matrices with storage backing.

Format-specific callers use the shared publisher and remain responsible for
restoring their affected in-memory objects and metadata after a failure.
"""
