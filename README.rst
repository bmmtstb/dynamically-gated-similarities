.. image:: https://github.com/bmmtstb/dynamically-gated-similarities/actions/workflows/wiki.yaml/badge.svg
    :target: https://github.com/bmmtstb/dynamically-gated-similarities/actions/workflows/wiki.yaml
    :alt: Documentation Status

.. image:: https://github.com/bmmtstb/dynamically-gated-similarities/actions/workflows/ci.yaml/badge.svg
    :target: https://github.com/bmmtstb/dynamically-gated-similarities/actions/workflows/ci.yaml
    :alt: Linting and Testing


.. image:: https://zenodo.org/badge/713506951.svg
  :target: https://doi.org/10.5281/zenodo.14910546



Dynamically Gated Similarities
==============================

This is the code for the Thesis *"Multi-Person Pose Tracking using Dynamically Gated Similarities"*, available ` ``./thesis.pdf`` <https://github.com/bmmtstb/dynamically-gated-similarities/tree/master/thesis.pdf>`_ .


You can found the extended Documentation on `bmmtstb.github.io <https://bmmtstb.github.io/dynamically-gated-similarities/>`_.

Notes
~~~~~

You can find a visual Pipeline on
`LucidChart <https://lucid.app/lucidchart/848ef9df-ac3d-464d-912f-f5760b6cfbe9/edit?viewport_loc=-201%2C-52%2C2143%2C1007%2CuW69bC8kN~kl&invitationId=inv_e5a52469-f95f-414f-a78b-3416435fcb2d>`_ or downloadable as
`PDF (main) <https://github.com/bmmtstb/dynamically-gated-similarities/tree/master/docs/figures/Pipeline-DGS-Overview.pdf>`_ (or see: ``./docs/figures/Pipeline-DGS-Overview.pdf``).
The visual pipeline of the training module is also available as `PDF (training) <https://github.com/bmmtstb/dynamically-gated-similarities/tree/master/docs/figures/Pipeline-DGS-Training.pdf>`_ (or see: ``./docs/figures/Pipeline-DGS-Training.pdf``).


Folder Structure
~~~~~~~~~~~~~~~~

::

    dynamically_gated_similarities
    │
    └─── configs
    │        Multiple configuration.yaml files for running DGS or different submodules.
    │
    └─── docs
    │    │    Source files for the documentation via sphinx and autodoc.
    │    │
    │    └─── figures
    │             Images for the documentation and general explanation.
    │
    └─── data
    │        folder containing the datasets, for structure see './data/dataset.rst' for more info.
    │
    └─── dependencies
    │        References to git submodules e.g. to torchreid and my custom AlphaPose Fork.
    │
    └─── dgs
    │    │    The source code of the algorithm.
    │    │
    │    └    dgs_config.py
    │    │        Some default configuration if not overridden by config.yaml
    │    │        This file will soon be replaced by 'dgs_values.yaml' .
    │    └    dgs_values.yaml
    │    │        Some default values if not overridden by config.yaml
    │    │
    │    └─── models
    │    │        The building blocks for the DGS algorithm. Most models should be extendable fairly
    │    │        straight-forward to implement custom sub-modules.
    │    │
    │    └─── utils
    │             File-handling, IO, classes for State and Track handling, constants,
    │             functions for torch module handling  visualization, and overall image handling
    └─── pre_trained_models
    │        storage for downloaded or custom pre-trained models
    │
    └─── tests
    │        tests for dgs module
    │
    │
    └─── .gitmodules      - The project uses git submodules to include different libraries.
    └─── .pylintrc        - Settings for the pylint linter.
    └─── LICENSE          - MIT License
    └─── pyproject.toml   - Information about this project and additional build parameters.
    └─── requirements.txt - Use pip to install the requirements,
    │                       see './docs/installation.rst' for more information.


Abbreviations and Definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is expected that all joints have 2D coordinates,
but extending the code to 3D should be possible with minor adjustments.
If joints have three-dimensions in the given code, it is expected, that the third dimension is the joint visibility.

Images in PyTorch and torchvision expect the dimensions as: ``[B x C x H x W]``.
Matplotlib and PIL use another structure: ``[B x H x W x C]``.
In which format the image tensor is, depends on the location in the code.
Most general functions in torchvision expect uint8 (byte) tensors,
while the torch Modules expect a float (float32) image, to be able to compute gradients over images.
Some single images might not have the first dimension ``[C x H x W]``,
even though most parts of the code expect a given Batch size.

With the :class:`~.State` object, a general class for passing data between modules is created.
Therefore, modules, where child-modules might have different outputs,
generally use this State object instead of returning possibly non descriptive tensors.
This can be seen in the :class:`~.SimilarityModule` class and its children.
SimilarityModules can be quite different,
the pose similarity (e.g. :class:`~.ObjectKeypointSimilarity` ) does need the key-point coordinates to compute the OKS,
while the visual similarity (e.g. :class:`~.TorchreidVisualSimilarity` ) needs the image crops to compute embeddings.

+----------------------------+-------------------------------------------------------------------------+
|  Name                      | Description                                                             |
+============================+=========================================================================+
| J                          | Number of joint-key-points in the given model (e.g. ``coco=17``)        |
+----------------------------+-------------------------------------------------------------------------+
| C                          | Number of channels of the current image (e.g. ``RGB=3``)                |
+----------------------------+-------------------------------------------------------------------------+
| B                          | Current batch-size, can be 0 in some cases                              |
+----------------------------+-------------------------------------------------------------------------+
| N                          | Number of detections in the current frame                               |
+----------------------------+-------------------------------------------------------------------------+
| T                          | Number of tracks at the current time                                    |
+----------------------------+-------------------------------------------------------------------------+
| L                          | Number of "historical" frames in a dataset.                             |
|                            | The dataset has length :math:`L+1`                                      |
+----------------------------+-------------------------------------------------------------------------+
| H,W                        | Height and Width of the current image, as image shape: :math:`(H, W)`   |
+----------------------------+-------------------------------------------------------------------------+
| h,w                        | Specific given height or width, as image shape: :math:`(h, w)`          |
+----------------------------+-------------------------------------------------------------------------+
| HM\ :sub:`H`, HM\ :sub:`W` | Size of the heatmap, equals size of the cropped resized image           |
+----------------------------+-------------------------------------------------------------------------+
| E\ :sub:`V`, E\ :sub:`P`   | Embedding size, denoted for visual or pose based shape                  |
+----------------------------+-------------------------------------------------------------------------+


Citing
~~~~~~

To cite this thesis, you can use the following BibTeX entry:

::

    @mastersthesis{steinborn2025dgs,
        author		= {Martin Steinborn},
        language	= {en},
        year		= {2025},
        month		= {Februar},
        address		= {Darmstadt},
        school		= {Technische Universit{\"a}t Darmstadt},
        title		= {Multi-Person Pose Tracking Using Dynamically Gated Similarities},
        keywords	= {tracking,pose-tracking,mppt},
        url			= {http://tuprints.ulb.tu-darmstadt.de/29468/},
    }

To cite the code, you can use the following BibTeX entry:

::

    @software{brizar_2025_14910547,
      author       = {Brizar},
      title        = {bmmtstb/dynamically-gated-similarities},
      month        = feb,
      year         = 2025,
      publisher    = {Zenodo},
      version      = {v0.3.0},
      doi          = {10.5281/zenodo.14910547},
      url          = {https://doi.org/10.5281/zenodo.14910547},
    }

