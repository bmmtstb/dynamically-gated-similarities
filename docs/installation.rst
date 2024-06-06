Installation
============

Clone the repository, but make sure to recursively clone the submodules too.

::
	git clone --recursive git@github.com:bmmtstb/dynamically-gated-similarities

If you cloned the repository already, use ``git submodule update --init --recursive``.

Packages and Environment
------------------------

Create a new ``python 3.10`` environment and install the requirements.
This will also install the required submodules. This will take a while!

.. note::
	This will install PyTorch with support for CUDA 12.1, you might want to change that for your environment.

::
	python3.10 -m venv venv

Activate the environment according to you OS, then install the base requirements.

::
	pip install -r requirements.txt


To properly install the ``lapsolver`` and |torchreid|_-package, we need to run the setup scripts in the submodules / dependencies.
Both are submodules, make sure that they have been cloned properly.
There seemed to be multiple issues with the sub-sub-module ``lapsopver/pybind11``.
If there is no ``CMakeLists.txt``-file in there, make sure to run ```git submodule init && git submodule update``` in that folder.

Some of the respective requirements of |torchreid| have been installed by this packages requirements file already, because the setup script seems broken right now.

And finally, make sure to install the dgs module itself.

::
	cd ./dependencies/py-lapsolver/
	python setup.py develop
	cd ../torchreid
	python setup.py develop
	cd ../..
	pip install -e .

Now the dgs module is installed, including two evaluation tools:

- The |PT21|_ evaluation toolkit, named ``posetrack21``.
  Note, that you need to download the |PT21|_ dataset first.
  (It might be possible to use it for PT17 or PT18 data, but no guarantees.)
  Have a look at the respective repository for how to do so.
- The ``poseval`` -package has been installed by the requirements and can be used for evaluation too.
  For more information, visit the respective _`GitHub <https://github.com/leonid-pishchulin/poseval>` page.

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

Pytorch Keypoint-RCNN
~~~~~~~~~~~~~~~~~~~~~

See :class:`KeypointRCNNImageBackbone`.

AlphaPose Backbone
~~~~~~~~~~~~~~~~~~

Currently it is only possible to use AP as predictor using a separate AlphaPose installation,
and then import the json prediction-files.
There is a WIP-version of a AlphaPose backbone on the ``alpha_pose`` branch.
