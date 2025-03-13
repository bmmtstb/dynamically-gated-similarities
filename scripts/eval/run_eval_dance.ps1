# With lots of love from Chat-GPT

# activate the virtual environment
.\venv\Scripts\activate

# Define the base directory
$base_dir = ".\data\DanceTrack\"

# Get all subdirectories in the $base_dir\val\ directory that start with "results_"
$dirs = Get-ChildItem "$base_dir\val\" -Directory | Where-Object { $_.Name -like "results_*" }

# Loop over each directory in $dirs
foreach ($dir in $dirs) {
    $eval_data_folder = "$($dir.FullName)\eval_data"

    # Check if the "eval_data" folder exists and contains the two required files
    if (-Not (Test-Path $eval_data_folder) -or -Not (Test-Path "$eval_data_folder\pedestrian_detailed.csv") -or -Not (Test-Path "$eval_data_folder\pedestrian_summary.txt")) {

        # Print the name of the directory
        Write-Output "Processing directory: $($dir.Name)"

        # Construct the Python command
        $python_command = "python .\dependencies\DanceTrack\TrackEval\scripts\run_mot_challenge.py" +
                          " --SKIP_SPLIT_FOL True" +
                          " --SPLIT_TO_EVAL val" +
                          " --METRICS HOTA CLEAR Identity" +
                          " --GT_FOLDER .\data\DanceTrack\val\" +
                          " --SEQMAP_FILE $base_dir\val_seqmap.txt" +
                          " --TRACKERS_FOLDER .\data\DanceTrack\val\" +
                          " --TRACKERS_TO_EVAL $($dir.Name)" +
                          " --TRACKER_SUB_FOLDER ." +
                          " --OUTPUT_SUB_FOLDER .\eval_data\" +
                          " --USE_PARALLEL True" +
                          " --NUM_PARALLEL_CORES 8" +
                          " --OUTPUT_DETAILED True" +
                          " --OUTPUT_SUMMARY True" +
                          " --PRINT_CONFIG False" +
                          " --PRINT_RESULTS False"

        # Execute the Python command
        Invoke-Expression $python_command
    }
}
