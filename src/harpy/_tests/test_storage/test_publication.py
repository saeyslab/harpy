"""The publisher moves paths; payload encoding is tested by each I/O adapter."""

from pathlib import Path

import pytest

from harpy._storage._publication import (
    _cleanup_owned_path,
    _publish_staged_paths,
    _remove_owned_path,
    _StagedPath,
    log,
)


def _staged_update(tmp_path, *, existing):
    root = tmp_path / "store"
    workspace = tmp_path / "workspace"
    root.mkdir()
    workspace.mkdir()
    (root / "unrelated").write_text("keep")
    paths = []
    for name in ("matrix", "metadata"):
        staged = workspace / name
        staged.mkdir()
        (staged / "payload").write_text(f"new {name}")
        destination = root / name
        if existing:
            destination.mkdir()
            (destination / "payload").write_text(f"old {name}")
        paths.append(_StagedPath(staged=staged, destination=destination))
    return root, workspace, tuple(paths)


@pytest.mark.parametrize("existing", [False, True])
def test_publish_staged_paths_publishes_one_consistency_unit(tmp_path, existing):
    root, workspace, paths = _staged_update(tmp_path, existing=existing)
    with _publish_staged_paths(root=root, workspace=workspace, paths=paths, operation="test"):
        for replacement in paths:
            assert (replacement.destination / "payload").read_text() == f"new {replacement.destination.name}"
        # Rollback copies live outside the store and survive throughout the body.
        assert list(tmp_path.glob(".store.harpy-test-backup-*"))
    assert not workspace.exists()
    assert not list(tmp_path.glob(".store.harpy-test-backup-*"))
    assert (root / "unrelated").read_text() == "keep"


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["publication", "body"])
def test_publish_staged_paths_restores_destinations_on_failure(tmp_path, monkeypatch, existing, failure):
    root, workspace, paths = _staged_update(tmp_path, existing=existing)
    rename = Path.rename

    def fail_second_publication(path, target):
        if path == paths[1].staged:
            raise OSError("injected publication failure")
        return rename(path, target)

    if failure == "publication":
        monkeypatch.setattr(Path, "rename", fail_second_publication)
    with pytest.raises((OSError, RuntimeError), match="injected"):
        with _publish_staged_paths(root=root, workspace=workspace, paths=paths, operation="test"):
            raise RuntimeError("injected body failure")
    for replacement in paths:
        if existing:
            assert (replacement.destination / "payload").read_text() == f"old {replacement.destination.name}"
        else:
            assert not replacement.destination.exists()
    assert (root / "unrelated").read_text() == "keep"
    assert not workspace.exists()
    assert not list(tmp_path.glob(".store.harpy-test-backup-*"))


def test_publication_rejects_workspace_containing_store_before_any_mutation(tmp_path):
    root, workspace, paths = _staged_update(tmp_path, existing=True)
    with pytest.raises(ValueError, match="cannot contain the publication root"):
        with _publish_staged_paths(root=root, workspace=tmp_path, paths=paths, operation="test"):
            pytest.fail("Unsafe workspace accepted")
    assert workspace.exists()
    assert (root / "matrix" / "payload").read_text() == "old matrix"


def test_failed_rollback_retains_backup_and_reports_location(tmp_path, monkeypatch):
    root, workspace, paths = _staged_update(tmp_path, existing=True)
    rename = Path.rename

    def fail_restore(path, target):
        if path.name == "path-0":
            raise OSError("injected restore failure")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_restore)
    with pytest.raises(RuntimeError, match="Remaining backup data are at"):
        with _publish_staged_paths(root=root, workspace=workspace, paths=paths, operation="test"):
            raise RuntimeError("injected body failure")
    backup = next(tmp_path.glob(".store.harpy-test-backup-*"))
    assert (backup / "path-0" / "payload").read_text() == "old matrix"


@pytest.mark.parametrize("location", ["workspace", "staged", "destination"])
@pytest.mark.parametrize("broken", [False, True])
def test_publication_rejects_symlink_paths_before_any_mutation(tmp_path, location, broken):
    root, workspace, paths = _staged_update(tmp_path, existing=True)
    if location == "workspace":
        link = tmp_path / "workspace-link"
        target = workspace
        publication_workspace = link
        replacements = paths
    elif location == "staged":
        link = workspace / "staged-link"
        target = paths[1].staged
        publication_workspace = workspace
        replacements = (paths[0], _StagedPath(staged=link, destination=paths[1].destination))
    else:
        link = root / "destination-link"
        target = paths[1].destination
        publication_workspace = workspace
        replacements = (paths[0], _StagedPath(staged=paths[1].staged, destination=link))
    if broken:
        target = tmp_path / "missing"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="Symbolic links are not supported"):
        with _publish_staged_paths(root=root, workspace=publication_workspace, paths=replacements, operation="test"):
            pytest.fail("Symlink paths must be rejected before publication")

    assert link.is_symlink()
    assert link.readlink() == target
    assert target.exists() == (not broken)
    for replacement in paths:
        assert (replacement.staged / "payload").read_text() == f"new {replacement.destination.name}"
        assert (replacement.destination / "payload").read_text() == f"old {replacement.destination.name}"
    assert not list(tmp_path.glob(".store.harpy-test-backup-*"))


def test_publication_allows_symlinked_ancestor_directory(tmp_path):
    parent = tmp_path / "real-parent"
    parent.mkdir()
    root, workspace, paths = _staged_update(parent, existing=True)
    alias = tmp_path / "alias"
    alias.symlink_to(parent, target_is_directory=True)
    replacements = tuple(
        _StagedPath(
            staged=alias / workspace.name / replacement.staged.name,
            destination=alias / root.name / replacement.destination.name,
        )
        for replacement in paths
    )

    with _publish_staged_paths(
        root=alias / root.name, workspace=alias / workspace.name, paths=replacements, operation="test"
    ):
        pass

    assert alias.is_symlink()
    for replacement in paths:
        assert (replacement.destination / "payload").read_text() == f"new {replacement.destination.name}"
    assert not workspace.exists()
    assert not list(parent.glob(".store.harpy-test-backup-*"))


@pytest.mark.parametrize("target_kind", ["file", "directory", "missing"])
def test_symlink_cleanup_refuses_removal_and_preserves_original_error(tmp_path, monkeypatch, target_kind):
    target = tmp_path / "target"
    if target_kind == "file":
        target.write_text("keep")
    elif target_kind == "directory":
        target.mkdir()
        (target / "payload").write_text("keep")
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=target_kind == "directory")

    with pytest.raises(ValueError, match="Refusing to remove symbolic link"):
        _remove_owned_path(link)

    messages = []
    monkeypatch.setattr(log, "warning", messages.append)
    original_error = RuntimeError("original operation failed")
    with pytest.raises(RuntimeError) as raised:
        try:
            raise original_error
        finally:
            _cleanup_owned_path(link)

    assert raised.value is original_error
    assert any(str(link) in message and "Refusing to remove symbolic link" in message for message in messages)
    assert link.is_symlink()
    assert link.readlink() == target
    if target_kind == "file":
        assert target.read_text() == "keep"
    elif target_kind == "directory":
        assert (target / "payload").read_text() == "keep"
    else:
        assert not target.exists()
