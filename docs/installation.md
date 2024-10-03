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

Depending on your hardware, you may need to adapt the [`environment.yml`](../environment.yml) file as follows:

- On Windows comment out the lines `basicpy==...`, `jax==...` and `jaxlib==...` in the `environment.yml`. We will install `basicpy` and `jax` manually as follows after the environment is build:

```bash
pip install "jax[cpu]===0.4.10" -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver
pip install basicpy==1.0.0
```

- On Mac comment out the line `mkl=2024.0.0`.

Note that `basicpy==...`, `jax==...` and `jaxlib==...` can be commented in the `environment.yml` if you do not plan to use the `Harpy` function `sp.im.tiling_correction`, you will still be able to use Harpy. For a mimimal list of requirements for `Harpy`, we refer to the [setup.cfg](../setup.cfg).

## 2. Install `Harpy`:

```
pip install git+https://github.com/saeyslab/harpy.git
```

## Additional dependencies

To use the plugin, run

```bash
pip install "git+https://github.com/saeyslab/harpy.git#egg=sparrow[plugin]"
```

To run `Harpy` from the `cli`:

```bash
pip install "git+https://github.com/saeyslab/harpy.git#egg=sparrow[cli]"
```

## NVIDIA GPU support

We provide [environment_vib_compute.yml](../environment_vib_compute.yml) that will install `torch` with NVIDIA GPU support on Linux (tested on CentOS). After creation of the environment via `conda env create -f environment_vib_compute.yml`, activate the environment, and install `Harpy` via `pip install git+https://github.com/saeyslab/harpy.git`.

For VIB members we also refer to [this document](./tutorials/hpc/vib_compute.md), for an example on how to use the VIB compute cluster with GPU support.
