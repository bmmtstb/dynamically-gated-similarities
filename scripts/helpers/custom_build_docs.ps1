# Check if mamba exists before attempting to deactivate
if (Get-Command mamba -ErrorAction SilentlyContinue) {
    mamba deactivate *>$null
}

# Source the user's profile to load environment variables
. $PROFILE

# Activate the virtual environment
. venv\Scripts\Activate

# Clean previous build artifacts
make clean *> $null
Remove-Item -Recurse -Force dgs_autosummary, _build -ErrorAction SilentlyContinue

# Log start time
$startTime = Get-Date
Write-Host "Build process starting..."

# First linkcheck without cache, then build using that cache
$env:PYTHONWARNINGS=""
sphinx-build . _build/ -T -q -E -a -j auto -b linkcheck
sphinx-build . _build/ -T -q -j auto

# Log end time and total duration
$endTime = Get-Date
$duration = $endTime - $startTime
$formattedDuration = "{0:D2}:{1:D2}" -f $duration.Minutes, $duration.Seconds
Write-Host "Build process finished. Total time: $formattedDuration"
