Installation
============

Use condas ``environment.yaml`` to create environment with specific python version and some packages

::

    conda env create -n DGS --file environment.yaml
    conda activate DGS

Because the torchreid pip-package seems broken right-now, we are cloning torchreid as submodule and build it from source while creating the environment.

AlphaPose needs halpecocotools where the installer is also broken for now. halpecocotools can be imported and build as submodules too. After installing halpecocotools, the AlphaPose repository is installed similarly.
