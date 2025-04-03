# Installation

We recommend using Anaconda to install Harpy, and we provide an [`environment.yml`](../environment.yml).

## 1. Create the conda environment:

```bash
# Use standard Conda environment creation
conda env create -f environment.yml
# Or use Mamba as alternative
mamba env update -f environment.yml --prune

conda activate harpy
```

If you plan to use the `Harpy` function `harpy.im.tiling_correction`, please install `jax` and `basicpy`. On Mac and Linux, this can be done via `pip install ...`, on Windows you will have to run the following commands:

```bash
pip install "jax[cpu]" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver
pip install basicpy
```

On Mac, please comment out the line `mkl=2024.0.0` in `environment.yml`.

For a mimimal list of requirements for `Harpy`, we refer to the [pyproject.toml](../pyproject.toml).

## 2. Install `Harpy`:

```
pip install git+https://github.com/saeyslab/harpy.git
```

## Additional dependencies

To use the plugin, run

```bash
pip install "git+https://github.com/saeyslab/harpy.git#egg=harpy[plugin]"
```

To run `Harpy` from the `cli`:

```bash
pip install "git+https://github.com/saeyslab/harpy.git#egg=harpy[cli]"
```

To be able to run the unit tests:

```bash
pip install "git+https://github.com/saeyslab/harpy.git#egg=harpy[testing]"
```

## NVIDIA GPU support

We provide [environment_vib_compute.yml](../environment_vib_compute.yml) that will install `torch` with NVIDIA GPU support on Linux (tested on CentOS). After creation of the environment via `conda env create -f environment_vib_compute.yml`, activate the environment, and install `Harpy` via `pip install git+https://github.com/saeyslab/harpy.git`.

For VIB members we also refer to [this document](./tutorials/hpc/vib_compute.md), for an example on how to use the VIB compute cluster with GPU support.
