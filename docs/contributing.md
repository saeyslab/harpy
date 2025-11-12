# Development

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

Install a pre-commit hook to run all configured checks in `.pre-commit-config.yaml`:

```
pre-commit install
pre-commit run -a
```
