#!/bin/bash

# Set base output directory
BASE_OUTPUT_DIR="results/MPD/mammals_results"

# Create base output directory
mkdir -p $BASE_OUTPUT_DIR

# Process WAG_sampling_seq_root series
echo "Processing WAG_sampling_seq_root series..."
SERIES_DIR="$BASE_OUTPUT_DIR/WAG_sampling_seq_root"
mkdir -p $SERIES_DIR
for rate in 0 0.05 0.1 0.2 0.5; do
    echo "Processing rate=$rate..."
    python scripts/MPD/process_alignments.py --empirical results/preprocess_mammals_new/gap_and_ambigless --simulation results/mammals_simulations/BPP/WAG_sampling_seq_root_$rate --plot --threads 5 --output $SERIES_DIR
done
python scripts/MPD/process_alignments.py plot_results --input $SERIES_DIR/results --output $SERIES_DIR/plots

# Process WAG_sampling_seq_data2_root series
echo "Processing WAG_sampling_seq_data2_root series..."
SERIES_DIR="$BASE_OUTPUT_DIR/WAG_sampling_seq_data2_root"
mkdir -p $SERIES_DIR
for rate in 0 0.05 0.1 0.2 0.5; do
    echo "Processing rate=$rate..."
    python scripts/MPD/process_alignments.py --empirical results/preprocess_mammals_new/gap_and_ambigless --simulation results/mammals_simulations/BPP/WAG_sampling_seq_data2_root_$rate --plot --threads 5 --output $SERIES_DIR
done
python scripts/MPD/process_alignments.py plot_results --input $SERIES_DIR/results --output $SERIES_DIR/plots

# Process WAG_frequencies_posterior_extra_length_ext series
echo "Processing WAG_frequencies_posterior_extra_length_ext series..."
SERIES_DIR="$BASE_OUTPUT_DIR/WAG_frequencies_posterior_extra_length_ext"
mkdir -p $SERIES_DIR
for rate in 0 0.05 0.1 0.2 0.5; do
    echo "Processing rate=$rate..."
    python scripts/MPD/process_alignments.py --empirical results/preprocess_mammals_new/gap_and_ambigless --simulation results/mammals_simulations/BPP/WAG_frequencies_posterior_extra_length_ext_$rate --plot --threads 5 --output $SERIES_DIR
done
python scripts/MPD/process_alignments.py plot_results --input $SERIES_DIR/results --output $SERIES_DIR/plots

# Process WAG_frequencies_posterior_extra_length_data2_ext series
echo "Processing WAG_frequencies_posterior_extra_length_data2_ext series..."
SERIES_DIR="$BASE_OUTPUT_DIR/WAG_frequencies_posterior_extra_length_data2_ext"
mkdir -p $SERIES_DIR
for rate in 0 0.05 0.1 0.2 0.5; do
    echo "Processing rate=$rate..."
    python scripts/MPD/process_alignments.py --empirical results/preprocess_mammals_new/gap_and_ambigless --simulation results/mammals_simulations/BPP/WAG_frequencies_posterior_extra_length_data2_ext_$rate --plot --threads 5 --output $SERIES_DIR
done
python scripts/MPD/process_alignments.py plot_results --input $SERIES_DIR/results --output $SERIES_DIR/plots

# Process WAG_frequencies_sampling_seq series
echo "Processing WAG_frequencies_sampling_seq series..."
SERIES_DIR="$BASE_OUTPUT_DIR/WAG_frequencies_sampling_seq"
mkdir -p $SERIES_DIR
python scripts/MPD/process_alignments.py --empirical results/preprocess_mammals_new/gap_and_ambigless --simulation results/mammals_simulations/BPP/WAG_frequencies_sampling_seq --plot --threads 5 --output $SERIES_DIR
python scripts/MPD/process_alignments.py plot_results --input $SERIES_DIR/results --output $SERIES_DIR/plots

# Process WAG_frequencies_sampling_seq_data2 series
echo "Processing WAG_frequencies_sampling_seq_data2 series..."
SERIES_DIR="$BASE_OUTPUT_DIR/WAG_frequencies_sampling_seq_data2"
mkdir -p $SERIES_DIR
python scripts/MPD/process_alignments.py --empirical results/preprocess_mammals_new/gap_and_ambigless --simulation results/mammals_simulations/BPP/WAG_frequencies_sampling_seq_data2 --plot --threads 5 --output $SERIES_DIR
python scripts/MPD/process_alignments.py plot_results --input $SERIES_DIR/results --output $SERIES_DIR/plots

# Process WAG_frequencies series
echo "Processing WAG_frequencies series..."
SERIES_DIR="$BASE_OUTPUT_DIR/WAG_frequencies"
mkdir -p $SERIES_DIR
python scripts/MPD/process_alignments.py --empirical results/preprocess_mammals_new/gap_and_ambigless --simulation results/mammals_simulations/BPP/WAG_frequencies --plot --threads 5 --output $SERIES_DIR
python scripts/MPD/process_alignments.py plot_results --input $SERIES_DIR/results --output $SERIES_DIR/plots

echo "All processing completed! Results are saved in the $BASE_OUTPUT_DIR directory"


