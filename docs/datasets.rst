.. _dataset_page:

Datasets
========

The datasets all lie in the ``./data/`` folder, each dataset in its respective subfolder.
To speed up inference times, a preprocessing step will be done for most datasets.

DanceTrack
----------

As the name suggests, ``DanceTrack`` consists of 100 videos of people dancing in groups of two to about ten.
Depending on the number of people in the video, there are many occlusions.
Additionally, the highly irregular motion of the dance moves makes it harder for the tracker to keep up with all the people.

The dataset, the evaluation server, and more information can be found on https://dancetrack.github.io/ .

Preprocessing
~~~~~~~~~~~~~

The script ``./scripts/helpers/extract_bboxes_MOT.py`` can be used to extract the bounding boxes and the image crops of the whole dataset.
The script extracts the ground-truth bounding boxes and uses the RCNN backbone to pre-compute the keypoints, bounding-boxes, and image crops of the test and validation datasets.
Depending on the number combinations between the IoU and score thresholds set, this script can run multiple hours.
The results are saved directly in the respective dataset folder, and need roughly 30GB of additional space.

Citing DanceTrack
~~~~~~~~~~~~~~~~~

.. code-block:: bibtex

	@inproceedings{sun2022dance,
		title={DanceTrack: Multi-Object Tracking in Uniform Appearance and Diverse Motion},
		author={Sun, Peize and Cao, Jinkun and Jiang, Yi and Yuan, Zehuan and Bai, Song and Kitani, Kris and Luo, Ping},
		booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
		year={2022}
	}

PoseTrack21
-----------

Similar to ``PoseTrack18`` does ``PoseTrack21`` contain hundreds of short videos of day to day activities, plus some high adrenaline sporting videos.
The dataset also includes solo and team-sport videos from the olympic games.
Many people in the background are not properly annotated, and sometimes the masks to hide those people are set incorrectly.

The original evaluation server from ``PoseTrack18`` has been taken offline.

The dataset and the evaluation tools can be found at https://github.com/andoer/PoseTrack21 .

Key-Point Locations
~~~~~~~~~~~~~~~~~~~

The backbone model used on default (``KeypointRCNN``) returns the ``left_eye`` and ``right_eye`` instead of the ``head_bottom`` (or sometimes called ``neck``) and ``head_top``
This means that comparing the raw results of this model with other models on ``PoseTrack21`` might yield different evaluation scores.

Preprocessing
~~~~~~~~~~~~~

The script ``./scripts/helpers/extract_bboxes_pt21.py`` can be used to extract the bounding boxes and the image crops of the whole dataset.
The script extracts the ground-truth bounding boxes and uses the RCNN backbone to pre-compute the keypoints, bounding-boxes, and image crops of the test and validation datasets.
Depending on the number combinations between the IoU and score thresholds set, this script can run multiple hours.
The results are saved directly in the respective dataset folder, and need roughly 100GB of additional space.

Citing PoseTrack21
~~~~~~~~~~~~~~~~~~

.. code-block:: bibtex

	@inproceedings{doering22,
		title={Pose{T}rack21: {A} Dataset for Person Search, Multi-Object Tracking and Multi-Person Pose Tracking},
		author={Andreas Doering and Di Chen and Shanshan Zhang and Bernt Schiele and Juergen Gall},
		booktitle={CVPR},
		year={2022}
	}


PoseTrack21 is based on PoseTrack18, therefore please cite the original paper as well:

.. code-block:: bibtex

	@inproceedings{andriluka18,
		Title = {Pose{T}rack: {A} Benchmark for Human Pose Estimation and Tracking},
		booktitle = {CVPR},
		Author = {Andriluka, M. and Iqbal, U. and Ensafutdinov, E. and Pishchulin, L. and Milan, A. and Gall, J. and Schiele B.},
		Year = {2018}
	}