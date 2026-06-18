# Usage

## Input data

Harpy includes several built-in-datasets. For example to load data from a [Resolve Molecular Cartography](https://resolvebiosciences.com/) experiment on mouse liver:

```
from harpy.datasets.registry import get_registry

registry=get_registry()
path_image = registry.fetch( "transcriptomics/resolve/mouse/20272_slide1_A1-1_DAPI.tiff" )
path_coordinates = registry.fetch("transcriptomics/resolve/mouse/20272_slide1_A1-1_results.txt")
```

And to download an example `SpatialData` object resulting from running the `Harpy` pipeline for transcriptomics:

```
import harpy as hp
sdata=hp.datasets.resolve_example()
```

> 🔍 Explore more built-in datasets in the [**Harpy documentation**](https://harpy.readthedocs.io/en/latest/api.html#datasets).

### Jupyter notebooks

Check the notebooks in the [tutorials section](tutorials/index.md).
