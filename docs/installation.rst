Installation
============

Packages and Environment
------------------------

Create a new ``python 3.10`` environment and install the requirements.
This will also install the required submodules. This will take a while!

.. note::
	This will install PyTorch with support for CUDA 12.1, you might want to change that for your environment.

.. note::
	Due to pip not being able to install packages in a specific order,
	make sure to install the ``requirements.txt`` file line by line (excluding comments and empty lines).
	Therefore, you can not use inline comments behind a package.

::
	python3.10 -m venv venv
	grep -E -v '^[\s\t]*#.*$|^[\s\t]*$' requirements.txt | xargs -L 1 pip install

This will install the `torchreid` and `posetrack21` (pt21 evaluation toolkit) packages.
For all of those, the git submodules in the `./dependencies/`-folder are used.
This has multiple reasons:

- We only need the eval part of `posetrack21`. (As long as the dataset is downloaded separately).

Models - WIP
------------

There are a few examples in the `./scripts/` directory, with their explanations in _`scripts_page`.

Additionally check the information in _`weights` to download more example model weights.

If you want to train your own models, make sure to check out the _`dataset installation guide <dataset>`.


AlphaPose Backbone
~~~~~~~~~~~~~~~~~~

Currently it is only possible to use AP as predictor using a separate AlphaPose installation,
and then import the json prediction-files.
There is a WIP-version of a AlphaPose backbone on the ``alpha_pose`` branch.
