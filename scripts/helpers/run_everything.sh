#!/bin/bash

{
  # Run all the evaluation functions
  echo "Run single"
  python ./scripts/own/eval_single_similarity_const.py
  echo "Run pairwise"
  python ./scripts/own/eval_pair_similarities_const.py
  echo "Run triplet"
  python ./scripts/own/eval_triplet_similarities_const.py

  # Run evaluation
  echo "Run eval"
  ./scripts/helpers/run_eval_pt21.sh

  # Results to csv
  echo "Run results to csv"
  python ./scripts/helpers/results_to_csv.py
} 2>&1 | tee -a output.txt  # Redirect and append all output to both output.txt and the console
