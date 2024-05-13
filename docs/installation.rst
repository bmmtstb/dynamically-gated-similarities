Installation
============

Packages and Environment
------------------------

Create a new ``python 3.10`` environment and install the requirements.
This will also install the required submodules. This will take a while!

.. note::
	This will install PyTorch with support for CUDA 12.1, you might want to change that for your environment.

::
	python3.10 -m venv venv
	pip install -r requirements.txt


To properly install the |torchreid|_-package, we need to run the setup script in the submodules / dependencies.
The respective requirements of |torchreid| has been installed by this packages requirements file already.

::
	cd ./dependencies/torchreid/
	python setup.py develop

Now there are two evaluation tools installed:

- The |PT21|_ evaluation toolkit, named ``posetrack21``.
  Note, that you need to download the |PT21|_ dataset first.
  Have a look at the respective repository for how to do so.
- The ``poseval`` -package has been installed by the requirements and can be used for evaluation too.

Next Steps
----------

There are a few examples in the `./scripts/` directory, with their explanations _`here <scripts_page>`.

Check the information in _`weights <weights>` to download (more) example model weights or use your own.

There are multiple example configurations in the `./configs/` directory,
with some additional explanation _`here <configs>`.

If you want to train your own models, use _`custom datasets<dataset>`, or validate results,
make sure to check out the rest of the docs.
There are methods to register new modules for every type of existing module,
which simplifies the usage of custom modules in the configuration files.

Backbone Models
---------------

TODO

AlphaPose Backbone
~~~~~~~~~~~~~~~~~~

Currently it is only possible to use AP as predictor using a separate AlphaPose installation,
and then import the json prediction-files.
There is a WIP-version of a AlphaPose backbone on the ``alpha_pose`` branch.
