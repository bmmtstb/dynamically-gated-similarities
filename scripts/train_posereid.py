r"""
Train a pose-based ReID-model using the torchreid workflow
==========================================================

Workflow
--------

#. use AlphaPose to create pose estimates for Market1501 and underground_reid as long as the data does not exist
#. create / load pose-based ReID generation model
#. create DataManager for pose-based data

"""
