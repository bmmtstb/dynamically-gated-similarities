Installation
============

Packages and Environment
------------------------

Install `mamba <https://mamba.readthedocs.io/en/latest/installation.html>`
through `miniforge <https://github.com/conda-forge/miniforge>`
and create environment using the predefined file.
This will also install the required submodules. This will take a while!

.. note::
	This will install PyTorch with support for CUDA 11.8, you might want to change that for your environment.

.. note::
	This will install the `AlphaPose` and `torchreid` packages.
	For both the versions of the git submodules in the `./dependencies`-folder is used.

::

    mamba env create -n DGS --file environment.yaml
    mamba activate DGS


Due to the possibilities of conda being able to call `pip install`,
this is all contained within one single `environment.yaml` file.

To gain the cython speed bump of the torchreid package,
we are cloning torchreid as submodule and build it from source while creating the environment.
Torchreid can additionally be installed using pip directly, but it will warn constantly.

`AlphaPose` needs `halpecocotools` where the installer is also broken for now.
`halpecocotools` can be imported and build as git submodules too.
After installing `halpecocotools`, the AlphaPose repository is installed.
If this fails, check AlphaPose repo and _`install guide <ap install>` for tips, especially on windows machines.

Models
------

If you are planning on using the default AlphaPose Backend,
follow the _`models installation guide <ap install models>` over at AlphaPose.
And yes, these files have to be inserted into the submodule structure in
`./dependencies/AlphaPose_Fork/`.
You don't have to set up the AP tracker, if you plan on only using the DGS tracker.


Additionally check the information in _`weights` to download more example model weights.


If you want to train your own models, make sure to check out the _`dataset installation guide <dataset>`.


:: _ap install: `https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md`
:: _ap install models: `https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md#models`
