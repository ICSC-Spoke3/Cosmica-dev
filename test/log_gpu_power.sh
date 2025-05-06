#!/bin/bash

# File to save the data
OUTFILE="gpu_power_log.csv"

# Write header
echo "timestamp,power.draw" > "$OUTFILE"

# Loop for 3600 seconds (adjust as needed)
for i in {1..3600}; do
    nvidia-smi --query-gpu=timestamp,power.draw --format=csv,noheader,nounits >> "$OUTFILE"
    sleep 1
done
