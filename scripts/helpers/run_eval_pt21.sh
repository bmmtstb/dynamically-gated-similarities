#!/bin/bash

# Base directory to search
base_dir="./results/own/eval/"

# Iterate over every dataset
for dataset_dir in "$base_dir"/*; do

    # Skip if not a directory
    if [ ! -d "$dataset_dir" ]; then
        continue
    fi
    dataset=$(basename "$dataset_dir")

    # Iterate over every key / name within the current dataset
    for name_dir in "$dataset_dir"/*; do
        if [ ! -d "$name_dir" ]; then
            continue
        fi

        name=$(basename "$name_dir")

        # Check if results_json directory exists
        results_json_dir="$name_dir/results_json"
        if [ ! -d "$results_json_dir" ]; then
            continue
        fi

        echo "Running evaluation $name for dataset $dataset :"

        if [ ! -f "$name_dir/eval_data/total_AP_metrics.json" ]; then
          # Run pose evaluation from poseval (originally used by AP)
          python -m poseval.evaluate \
              --groundTruth ./data/PoseTrack21/posetrack_data/val/ \
              --predictions ./"$base_dir"/"$dataset"/"$name"/results_json/ \
              --outputDir ./"$base_dir"/"$dataset"/"$name"/eval_data/ \
              --evalPoseTracking --evalPoseEstimation --saveEvalPerSequence
        fi

        if [ ! -f "$results_json_dir/pose_hota_results.txt" ]; then
          # Run PoseTrack21 challenge evaluation
          python ./dependencies/PoseTrack21/eval/posetrack21/scripts/run_posetrack_challenge.py \
              --GT_FOLDER ./data/PoseTrack21/posetrack_data/val/ \
              --TRACKERS_FOLDER ./"$base_dir"/"$dataset"/"$name"/results_json/ \
              --USE_PARALLEL True --NUM_PARALLEL_CORES 4 \
              --PRINT_RESULTS True --OUTPUT_DETAILED True

        fi
    done
done
