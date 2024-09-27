#!/bin/bash

# Array of base directories to search
base_dirs=( "./results/own/eval/" "./results/own/train_single/" )

# Iterate over every base directory
for base_dir in "${base_dirs[@]}"; do
  # Iterate over every dataset
  for dataset_dir in "$base_dir"/*; do

    # Skip if not a directory
    if [ ! -d "$dataset_dir" ]; then
        continue
    fi
    dataset=$(basename "$dataset_dir")

    if [[ $dataset =~ ^OLD_ || $dataset =~ ^train_ || $dataset =~ _Dance_ ]]; then
      # echo "Skipping dataset $dataset"
      continue
    fi

    # Iterate over every key / name within the current dataset
    for name_dir in "$dataset_dir"/*; do
      if [ ! -d "$name_dir" ]; then
          continue
      fi

      name=$(basename "$name_dir")

      # skip old directories
      if [[ $name =~ ^OLD_ ]]; then
          continue
      fi

      # Check if results_json directory exists
      results_json_dir="$name_dir/results_json"

      if [ ! -d "$results_json_dir" ]; then
          echo "Skipping folder $name in $dataset because there is no results_json directory."
          continue
      fi

      mot_metrics_file="$name_dir/eval_data/total_MOT_metrics.json"
      pose_hota_results_file="$results_json_dir/pose_hota_results.txt"

      if [ -f "$mot_metrics_file" ] && [ -f "$pose_hota_results_file" ]; then
          # echo "Skipping $name for dataset $dataset because results already exist."
          continue
      fi
      echo "Running evaluation $name for dataset $dataset :"

      if [ ! -f "$mot_metrics_file" ]; then
        # Run pose evaluation from poseval (originally used by AP)
        python -m poseval.evaluate \
            --groundTruth ./data/PoseTrack21/posetrack_data/val/ \
            --predictions ./"$base_dir"/"$dataset"/"$name"/results_json/ \
            --outputDir ./"$base_dir"/"$dataset"/"$name"/eval_data/ \
            --evalPoseTracking --saveEvalPerSequence
            # --evalPoseEstimation  # skipped to speed-up evaluation
      fi

      if [ ! -f "$pose_hota_results_file" ]; then
        # Run PoseTrack21 challenge evaluation
        python ./dependencies/PoseTrack21/eval/posetrack21/scripts/run_posetrack_challenge.py \
            --GT_FOLDER ./data/PoseTrack21/posetrack_data/val/ \
            --TRACKERS_FOLDER ./"$base_dir"/"$dataset"/"$name"/results_json/ \
            --USE_PARALLEL True --NUM_PARALLEL_CORES 4 \
            --PRINT_RESULTS False --OUTPUT_DETAILED True

      fi
    done
  done
done