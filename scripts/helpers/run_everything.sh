#!/bin/bash

{
  # Run all the image generation
#  echo "Run image generation"
#  python ./scripts/helpers/extract_bboxes_pt21.py
#  python ./scripts/helpers/extract_bboxes_MOT.py

  # Run all the evaluation functions
  echo "Run single"
  python ./scripts/own/eval_single_similarity_const.py
  echo "Run evaluation of initial track weight"
  python ./scripts/own/eval_initial_weight.py

  echo "Run pairwise"
  python ./scripts/own/eval_pair_similarities_const.py
  echo "Run triplet"
  python ./scripts/own/eval_triplet_similarities_const.py

  # run the training
  echo "Run Training"
  python ./scripts/own/train_dynamic_weights.py

  # Run evaluation
  echo "Run eval"
  ./scripts/helpers/run_eval_pt21.sh
  ./scripts/helpers/run_eval_dance.sh

  # Results to csv
  echo "Run results to csv"
  python ./scripts/helpers/results_to_csv.py
} 2>&1 | tee -a output.txt  # Redirect and append all output to both output.txt and the console
