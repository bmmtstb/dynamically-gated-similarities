.. image:: https://readthedocs.org/projects/tracking-via-dynamically-gated-similarities/badge/?version=latest
    :target: https://tracking-via-dynamically-gated-similarities.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://github.com/bmmtstb/dynamically-gated-similarities/actions/workflows/wiki.yaml/badge.svg
    :target: https://github.com/bmmtstb/dynamically-gated-similarities/actions/workflows/wiki.yaml
    :alt: Docs

.. image:: https://github.com/bmmtstb/dynamically-gated-similarities/actions/workflows/ci.yaml/badge.svg
    :target: https://github.com/bmmtstb/dynamically-gated-similarities/actions/workflows/ci.yaml
    :alt: Linting and Testing


Dynamically Gated Similarities
==============================


Notes
-----

You can find a visual Pipeline on `LucidChart <https://lucid.app/documents/view/848ef9df-ac3d-464d-912f-f5760b6cfbe9>`_ or downloadable as `PDF <https://lucid.app/publicSegments/view/ddbebe1b-4bd3-46b8-9dfd-709b281c4b01>`_.


Folder Structure
~~~~~~~~~~~~~~~~


.. rst-class:: monospace-block

    dynamically_gated_similarities
    │
    └───configs
    │   │   configuration.yaml files for running different versions of DGS
    └───docs
    │   │   documentation and code for readthedocs
    └───data
    │   │   folder containing the datasets, for structure see :ref:`Dataset <dataset_page>`
    └───dependencies
    │   │   references to git submodules e.g. to my AlphaPose_Fork
    └───dgs
    │   │   source code of algorithm
    │   └───dgs_api.py      - structure of tracker, calls sub methods and classes defined in utils
    │   └───dgs_config.py   - default configuration if not overridden by config.yaml
    │   │
    │   └───utils
    │           block-models, helpers, and other util
    └───pre_trained_models
    │       storage for downloaded or custom pre-trained models
    └───tests
    │       tests for dgs module
    │
    │
    │   .gitmodules     - use git submodules to include different backends and libraries
    │   .pylintrc       - linting with pylint
    │   .readthedocs.yaml - near automatic documentation through readthedocs
    │   LICENSE         - MIT License


Abbreviations and Definitions
-----------------------------

It is expected that all joints have 2D coordinates, but 3D should be possible with minor adjustments.

Images in PyTorch have shape: [B x C x H x W] and for plotting in matplotlib [B x H x W x C]. Single images don't have the first dimension [C x H x W].

+--------------------------+--------------------------------------------------------+
|  Name                    | Description                                            |
+==========================+========================================================+
| J                        | Number of key-points in the given model (e.g. coco=17) |
+--------------------------+--------------------------------------------------------+
| C                        | Number of channels of the current image (e.g. RGB=3)   |
+--------------------------+--------------------------------------------------------+
| B                        | Current batch-size                                     |
+--------------------------+--------------------------------------------------------+
| H,W                      | Height and Width of the current image                  |
+--------------------------+--------------------------------------------------------+
| h,w                      | Specific given height or width                         |
+--------------------------+--------------------------------------------------------+
| E :sub:`V`, E :sub:`P`   | Embedding size, denoted for visual or pose based shape |
+--------------------------+--------------------------------------------------------+
