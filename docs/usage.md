# Usage

## Input data

There are some built-in datasets. For example to get data from a [Resolve](https://resolvebiosciences.com/) experiment on mouse liver:

```
from sparrow.datasets.registry import registry

path_image = registry.fetch( "transcriptomics/resolve/mouse/20272_slide1_A1-1_DAPI.tiff" )
path_coordinates = registry.fetch("transcriptomics/resolve/mouse/20272_slide1_A1-1_results.txt")
```

And to download an example SpatialData object resuling from running the `Harpy` pipeline:

```
import sparrow as sp
sdata=sp.datasets.resolve_example()
```

### Jupyter notebooks

Check the [notebooks](notebooks.md#tutorials) in the tutorials section.

### Napari plugin

After installing [installing](installation.md) Harpy, you can run the plugin by first starting Napari, and starting the plugin from Napari's menu bar: `napari > Plugins > sparrow`.

Use the plugin to tune the parameters of Harpy for the different steps of the pipeline. Tuning can be done on small crops of the image. After every step, a corresponding configuration _.yaml_ file will be saved in the output directory chosen by the user. We refer to the [hpc](hpc.md) documentation for information on how to use these generated configuration files via the CLI.

### (Hydra) CLI

Run experiments from the CLI using [Hydra](https://hydra.cc). Experiments can be run locally, or on a SLURM cluster.

First copy the `configs` folder (in the root of this repository) locally, and set the paths to the input data and log directory via the _.yaml_ provided at `configs/default.example.yaml`. I.e., first rename `configs/default.example.yaml` to `configs/default.yaml` and update following fields:

```yaml
paths:
  log_dir: ${oc.env:HOME}/VIB/DATA/logs/
  data_dir: ${oc.env:HOME}/VIB/DATA/
```

When running locally, use the following setting for the hydra launcher:

```yaml
defaults:
  - override /hydra/launcher: submitit_local
```

If sparrow is run on a SLURM cluster, change this to:

```yaml
defaults:
  - override /hydra/launcher: submitit_slurm
```

Next, update `configs/dataset/resolve_liver.yaml`, with the correct path to the input data, relative to _$paths.data_dir_ set earlier, e.g. the fields:

```yaml
data_dir: ${paths.data_dir}/resolve/resolve_liver
image: ${dataset.data_dir}/20272_slide1_A1-1_DAPI.tiff
coords: ${dataset.data_dir}/20272_slide1_A1-1_results.txt
markers: ${dataset.data_dir}/markerGeneListMartinNoLow.csv
```

assuming the RESOLVE mouse liver data is used.

The RESOLVE mouse liver experiment is preconfigured in `configs/experiment/resolve_liver.yaml`, and can now be run from the CLI:

```bash
sparrow +experiment=resolve_liver hydra.searchpath="[/Path/to/local/configs]" task_name=results_sparrow
```

Please update the _hydra.searchpath_ with the path to the `configs` folder downloaded locally.

All parameters can also be overwritten from the CLI, e.g. for the size of the min max filter:

```bash
sparrow +experiment=resolve_liver hydra.searchpath="[/Path/to/local/configs]" task_name=results_sparrow clean.size_min_max_filter=35
```

The default values for all parameters for each step of the pipeline can be found at `src/sparrow/configs`.
