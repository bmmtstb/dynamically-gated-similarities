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

This will install the `AlphaPose`, `torchreid`, `halpecocotools`, and `posetrack21` (pt21 evaluation toolkit) packages.
For all of those, the git submodules in the `./dependencies/`-folder are used.
This has multiple reasons:

- To gain the cython speed bump of the torchreid package,
  we are cloning torchreid as submodule and build it from source while creating the environment.
  Torchreid can additionally be installed using pip directly, but it will warn about cython constantly.
- The installer for `halpecocotools` seems broken right now.
- `AlphaPose` needs `halpecocotools` and both the installers are broken for now.
  If installing `AlphaPose` fails,
  check the original repo and _`install guide <ap install>` for tips, especially on windows machines.
- We only need the eval part of `posetrack21`. (As long as the dataset is downloaded separately).

Models - WIP
------------

There are a few examples in the `./scripts/` directory, with their explanations in _`scripts_page`.

Additionally check the information in _`weights` to download more example model weights.

If you want to train your own models, make sure to check out the _`dataset installation guide <dataset>`.


AlphaPose Backbone
~~~~~~~~~~~~~~~~~~

If you are planning on using the default AlphaPose Backend,
follow the _`models installation guide <ap install models>` over at AlphaPose.
And yes, these files have to be inserted into the submodule structure in
`./dependencies/AlphaPose_Fork/`.
You don't have to set up the AP tracker, if you plan on only using the DGS tracker.



:: _ap install: `https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md`
:: _ap install models: `https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md#models`
