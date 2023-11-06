Installation
============

Use condas ``environment.yaml`` to create environment with specific python version and some packages

::

    conda env create -n DGS --file environment.yaml
    conda activate DGS

Because the torchreid pip-package seems broken right-now, clone torchreid as submodule and build it from source.

::

    cd dependencies/torchreid/
    pip install -e .
    cd ../..

AlphaPose needs halpecocotools where the installer is also broken.

::

    cd dependencies/halpecocotools/PythonAPI/
    pip install -e .
    cd ../../..


After installing halpecocotools, the AlphaPose repository is installed similarly.

::

    cd dependencies/AlphaPose_Fork/
    pip install -e .
    cd ../..
