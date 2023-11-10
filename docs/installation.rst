Installation
============

Install `mamba <https://mamba.readthedocs.io/en/latest/installation.html>` through `miniforge <https://github.com/conda-forge/miniforge>` and create environment using the predefined file. This will also install the required submodules.

::

    mamba env create -n DGS --file environment.yaml
    mamba activate DGS


Due to the possibilities of `pip install -e` this is all contained within `environment.yaml`.

To gain the cython speed bump of the torchreid package, we are cloning torchreid as submodule and build it from source while creating the environment.

`AlphaPose` needs `halpecocotools` where the installer is also broken for now. `halpecocotools` can be imported and build as git submodules too. After installing `halpecocotools`, the AlphaPose repository is installed.
