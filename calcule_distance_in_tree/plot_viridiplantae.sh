#!/bin/bash

# Script for plotting only
BASE_OUTPUT_DIR="results/MPD/viridiplantae_results"

echo "Start batch plotting..."

for SERIES in "$BASE_OUTPUT_DIR"/*; do
    if [ -d "$SERIES/results" ]; then
        echo "Plotting $SERIES ..."
        python scripts/MPD/process_alignments.py plot_results --input "$SERIES/results" --output "$SERIES/plots"
    fi
done

echo "All plots are completed!"