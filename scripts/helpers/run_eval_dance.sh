#!/bin/bash

# With lots of love from Chat-GPT

# Check the operating system and activate the virtual environment accordingly
if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "darwin"* ]]; then
    # Linux or macOS
    source ./venv/bin/activate
elif [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    ./venv/Scripts/activate
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

# Define the base directory
base_dir="./data/DanceTrack/"

# Get all subdirectories in the $base_dir/val/ directory that start with "results_"
dirs=$(find "$base_dir/val/" -maxdepth 1 -type d -name "results_*")

# Loop over each directory in $dirs
for dir in $dirs; do
    eval_data_folder="$dir/eval_data"

    # Check if the "eval_data" folder exists and contains the two required files
    if [[ ! -d "$eval_data_folder" || ! -f "$eval_data_folder/pedestrian_detailed.csv" || ! -f "$eval_data_folder/pedestrian_summary.txt" ]]; then

        # Print the name of the directory
        echo "Processing directory: $(basename "$dir")"

        # Construct the Python command
        python_command="python ./dependencies/DanceTrack/TrackEval/scripts/run_mot_challenge.py \
                        --SKIP_SPLIT_FOL True \
                        --SPLIT_TO_EVAL val \
                        --METRICS HOTA CLEAR Identity \
                        --GT_FOLDER ./data/DanceTrack/val/ \
                        --SEQMAP_FILE $base_dir/val_seqmap.txt \
                        --TRACKERS_FOLDER ./data/DanceTrack/val/ \
                        --TRACKERS_TO_EVAL $(basename "$dir") \
                        --TRACKER_SUB_FOLDER . \
                        --OUTPUT_SUB_FOLDER ./eval_data/ \
                        --USE_PARALLEL True \
                        --NUM_PARALLEL_CORES 8 \
                        --OUTPUT_DETAILED True \
                        --OUTPUT_SUMMARY True \
                        --PRINT_CONFIG False \
                        --PRINT_RESULTS False"

        # Execute the Python command
        eval $python_command
    fi
done