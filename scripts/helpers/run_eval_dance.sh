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


# Function to recursively search directories and add matching ones to the dirs array
search_dirs() {
    local current_dir="$1"
    local dirs=("${!2}")  # Retrieve the current state of dirs array

    # Check if the current directory contains at least one file matching "dancetrack*.txt"
    if ls "$current_dir"/dancetrack*.txt 1> /dev/null 2>&1; then
        # Add directory to the dirs array
        dirs+=("$current_dir")
    fi

    # Iterate over subdirectories in the current directory
    for subdir in "$current_dir"/*/; do
        if [ -d "$subdir" ] && [[ $(basename "$subdir") != "eval_data" ]]; then
            # Recursively search inside this subdirectory and update dirs array
            dirs=($(search_dirs "$subdir" dirs[@]))
        fi
    done

    # Return the updated dirs array
    echo "${dirs[@]}"
}

# Define the base directory
base_dir="./data/DanceTrack/val/"

# Initialize an empty array to store matching directories
dirs=()

# Find all directories in the base directory that start with "results_*"
for results_dir in "$base_dir"/results_*/; do
  if [ -d "$results_dir" ]; then
    # Call the search_dirs function on each "results_*" directory and update dirs
    dirs=($(search_dirs "$results_dir" dirs[@]))
  fi
done


# Loop over each directory in $dirs
for dir in "${dirs[@]}"; do

  eval_data_folder="$dir./eval_data"

  # Check if the "eval_data" folder exists and contains the two required files
  if [[ ! -d "$eval_data_folder" || ! -f "$eval_data_folder/pedestrian_detailed.csv" || ! -f "$eval_data_folder/pedestrian_summary.txt" ]]; then
    # Print the name of the directory
    echo "Processing directory: $dir"

    # Construct the Python command
    python_command="python ./dependencies/DanceTrack/TrackEval/scripts/run_mot_challenge.py \
                    --SKIP_SPLIT_FOL True \
                    --SPLIT_TO_EVAL val \
                    --METRICS HOTA CLEAR Identity \
                    --GT_FOLDER $base_dir \
                    --SEQMAP_FILE $base_dir/../val_seqmap.txt \
                    --TRACKERS_FOLDER $(dirname "$dir") \
                    --TRACKERS_TO_EVAL $(basename "$dir") \
                    --TRACKER_SUB_FOLDER . \
                    --OUTPUT_SUB_FOLDER ./eval_data/ \
                    --USE_PARALLEL True \
                    --NUM_PARALLEL_CORES 8 \
                    --OUTPUT_DETAILED True \
                    --OUTPUT_SUMMARY True \
                    --PRINT_CONFIG False \
                    --PRINT_RESULTS False"

    # Execute the Python command and ignore errors
    eval "$python_command" || true
  fi
done
