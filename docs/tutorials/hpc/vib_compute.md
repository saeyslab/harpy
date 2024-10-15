# High Performance Computing on the VIB Data Core Compute.

## Set up environment

Log in to the compute cluster:

```bash
ssh -p 2022 user.name@compute.vib.be
```

Ask for an interactive session, e.g.:

```bash
salloc --partition=debug_28C_56T_750GB --ntasks=8 --mem=16G --time=02:00:00
```

In the default environment, install support for loading environment kernels (no modules should be loaded at this time, nor conda environments):

```bash
ml purge
ml Python
while [ ! -z $CONDA_PREFIX ]; do conda deactivate; done

pip install environment_kernels
```

Load the Mamba module to create a conda env.

```bash
ml load Mamba
```

Next use [environment_vib_compute.yml](../../../environment_vib_compute.yml) to build a conda environment:

```bash
mamba env create -f environment_vib_compute.yml
```

Activate the environment:

```bash
conda activate harpy
```

Install Harpy:

```bash
pip install git+ssh://git@github.com/saeyslab/harpy.git
```

## Run the Harpy notebook as an interactive session.

Make an ipython kernel to use in a JupyterLab notebook. The displayname is what you will select in JupyterLab.

```bash
ipython kernel install --user --name harpy --display-name "harpy"
```

Now on [https://compute.vib.be](https://compute.vib.be/pun/sys/dashboard/batch_connect/sys/jupyter-gpu/session_contexts/new), start a JupyterLab on GPU (select Python 3.10); check the conda environment box; select Mamba as the system wide Conda Module; fill in `harpy` as the name of the Custom Conda Environment.

You should now be able to run the notebook [Harpy_feature_calculation.ipynb](../general/Harpy_feature_calculation.ipynb) in an interactive session on the VIB compute cluster.
