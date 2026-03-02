#!/bin/bash

# Create a logs directory if it doesn't exist
mkdir -p logs

# Loop through every txt file in your sweeps folder
for file in sweeps/*.txt; do
    # Read and trim whitespace from the sweep ID
    SWEEP_ID=$(cat "$file" | tr -d '[:space:]')
    
    # Skip empty files
    if [ -z "$SWEEP_ID" ]; then
        echo "Skipping empty file: $file"
        continue
    fi
    
    echo "Submitting job for Sweep: $SWEEP_ID"
    
    # Submit the slurm script and pass the Sweep ID as an argument
    sbatch ost-train-multirun.slurm "$SWEEP_ID"
done