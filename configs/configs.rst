#############
Configuration
#############

.. toctree::
 :maxdepth: 2

Most to all parameters of the different modules can be set in respective configuration files.
Let's have look at how they are structured and how to use them.

Using the Configuration Files
=============================

The goal of the dynamically=gated=similarities [DGS] package was to create a modular pipeline,
that could run many different models using nearly identical python scripts.
Have a look at the file: ``./scripts/demo_predict.py`` .
This file can be used to predict the tracking results of the DGS module using a single config file.
The demo file is similar to ``./scripts/own/predict.py`` with a few of the boilerplate functions removed.
It is easy to see, that there is nearly no configuration in the python files itself.
The user can simply change the value of the ``CONFIG_FILE`` parameter,
and the rest of the script will adapt automatically.

Contents of the Configuration Files
-----------------------------------

The size and structure of the configuration file will depend on the use case of the overall module.
If the model, the user wants to use, defines parameters,
the configuration file needs to contain values for those parameters.
If the model contains additional or optional parameters, the user can change those using the configuration too.

Using the :class:`.DGSModule` as example, there are currently two required parameters: ``names`` and ``combine``.
Both are references to other keys in the same configuration file, used to load the different required submodules.
Further, there are two optional boolean parameters.
Both are used to define whether the module computes a probability distribution using a softmax.

To know to which module the respective configuration links, the configuration file uses a dict like structure.
The key is either used to reference from a module to another one (like for the DGS module),
or to tell the model loader which parameters to use.


.. code-block:: yaml

    dgs:
        names: [mod1, mod2]
        combine: static_alpha

    mod1:
        module_name: iou
        softmax: true

    mod2:
        ...

    static_alpha:
        module_name: "static_alpha"
        alpha: [0.6, 0.4]
        softmax: true


The :func:`module_loader` function accepts the key as an (keyword) argument,
together with the path to the configuration file, and the base name of the module.
For a list of modules, see :ref:`modules`.
The user can choose arbitrary names of the keys, but I recommend using descriptive ones.

```
module_loader(config="path/to/cfg.yaml", module_class="dgs", key="key_of_sub_config", **additional_kwargs)
```

Example Configurations
======================

There are multiple example configurations provided in the ``./configs/``-folder.

+---------------------+---------------------------+-------------+
| Configuration File  | Folder                    | Description |
+=====================+===========================+=============+
| eval_dgs.yaml       | DGS                       | ...         |
+---------------------+---------------------------+-------------+
| predict_images.yaml | DGS                       | ...         |
+---------------------+---------------------------+-------------+
| predict_video.yaml  | DGS                       | ...         |
+---------------------+---------------------------+-------------+
| eval_visual.yaml    | ReID Embedding Generation | ...         |
+---------------------+---------------------------+-------------+
| train_pose.yaml     | ReID Embedding Generation | ...         |
+---------------------+---------------------------+-------------+
| train_visual.yaml   | ReID Embedding Generation | ...         |
+---------------------+---------------------------+-------------+