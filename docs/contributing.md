# Development

## Storage internals

See [Metadata ownership and layout](development/metadata.md) for a schematic
overview of shared panels, element records and table metadata.

See [Storage writes and overwrite guarantees](development/storage.md) for the
shared storage architecture, whole-element versus component updates, and the
responsibilities of writers and the publication layer during recovery.

```{toctree}
:hidden: true

development/storage
development/metadata
```

## Setting up a development environment

First clone the GitHub repo and set it as the current directory:

```bash
git clone https://github.com/saeyslab/harpy.git
cd harpy
```

Install Harpy:

```bash
uv venv --python=3.12 # set python version
source .venv/bin/activate # activate the virtual environment
uv pip install -e '.[dev]' # use uv to pip install dependencies
python -c 'import harpy; print(harpy.__version__)' # check if the package is installed
# make changes
python -m pytest # run the tests
```

If you update the [pyproject.toml](../pyproject.toml), please also update the the [uv lock file](../uv.lock):

```bash
# update the lock file
uv lock
# check if lock file was updated correctly.
uv lock --check
```

This development environment is supported for:

- CentOS
- Ubuntu
- MacOS with an M1/M2 Pro
- Windows 11

Continuous integration will automatically run the tests on all pull requests.

## Type testing

Do a type test:

```
mypy --ignore-missing-imports src/
```

## Automated commit checks

After setting up each clone, install the Git hooks using the project's environment:

```bash
.venv/bin/pre-commit install
```

This installs both the commit and push hooks configured in `.pre-commit-config.yaml`.
Installing the Python package alone does not activate Git hooks. The hooks use the
pinned tool versions in the configuration, matching pre-commit.ci.

Run checks on the files you changed before committing:

```bash
.venv/bin/pre-commit run --files path/to/changed_file.py
```

If a hook modifies files, review the changes, stage them again, and retry the
commit. A failed check after automatic fixes is expected: the fixes need to be
included in the commit. Resolve any remaining reported lint errors manually.

To reproduce pre-commit.ci's checks across all tracked files, run:

```bash
.venv/bin/pre-commit run --all-files --show-diff-on-failure
```

Run this broader check when changing hook versions or investigating a CI failure,
since CI can report problems in files outside your commit.
