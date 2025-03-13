.. _scripts_page:

Explanation of Different Scripts
================================

To compute the results and to help with the overwhelming amount of data, multiple scripts were created to run DGS.

Most scripts have some sort of error handling using the decorator :py:meth:`~dgs.utils.notify_on_completion_or_error`.
This decorator expects an environment variable called ``DISCORD_WEBHOOK_URL`` to be set to a discord webhook,
which then receives all success and error messages.

demo_predict.py
~~~~~~~~~~~~~~~

In this file the basic procedure of tracking a single example video file is shown.

AlphaPose
~~~~~~~~~

The script ``ap_results_to_pt21.py`` can be used to convert the results by `AlphaPose <https://github.com/MVIG-SJTU/AlphaPose/>`_ into the PoseTrack21 result file format.
There was the idea to use the AlphaPose results as a backbone for DGS, but it was decided against, because too much preprocessing is done already.

eval
~~~~

The ``eval``-folder contains the main run-files used to evaluate (and test) the datasets with all the different parameters.
``run_eval_dance.sh`` is used to run, evaluate, and test the |DT|_ dataset.
The same does ``run_eval_pt21.sh`` do for the |PT21|_ dataset.
To run all the preprocessing, all scripts, and all the experiments, you can use ``run_everything.sh``.
A single computer takes more than two weeks to run everything!
All the scripts should be able to restart where they left off after being shutdown or if they failed.

Additionally, the file ``results_to_csv.py`` is used to combine all the results of the different datasets into combined csv files.
To save on storage space in the |DT| files, the precision was lowered and only the combined results are saved.


helpers
~~~~~~~

Within the ``helpers``-folder the scripts for updating the environments and the local CI are stored.

own
~~~

Within the ``own``-folder all python scripts that run and create the DGS engines are stored.
There are scripts for predicting, evaluating, testing, and training various datasets and different configurations.


preprocessing
~~~~~~~~~~~~~

This folder contains the two preprocessing scripts for the |PT21| and |DT| datasets respectively.

torchreid
~~~~~~~~~

Custom scripts for training a pose model and a visual-embedding generator using the torchreid pipeline.
Not fully functional!

visualization
~~~~~~~~~~~~~

``plot_triple_similarities.py`` can be used to create a different kind of plot for the triple similarities.
The code for most other visualizations, including the ones for the thesis, were done using ``seaborn`` using a local console, which means the scripts have not been stored unfortunately.