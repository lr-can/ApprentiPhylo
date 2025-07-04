#!/bin/bash

# Enter the target directory
cd runs_viridiplantae

# The basic model folders are already correctly named, no need to modify

# WAG Posterior Models
# With DATA2
for dir in WAG_frequencies_posterior_extra_length_data2_ext_*; do
    if [ -d "$dir" ]; then
        # Extract the numeric part
        num=$(echo $dir | grep -o '[0-9.]*$')
        # Create new name
        new_name="WAG_F_P_DATA2_E_$num"
        mv "$dir" "$new_name"
        echo "Renamed $dir to $new_name"
    fi
done

# Without DATA2
for dir in WAG_frequencies_posterior_extra_length_ext_*; do
    if [ -d "$dir" ]; then
        # Extract the numeric part
        num=$(echo $dir | grep -o '[0-9.]*$')
        # Create new name
        new_name="WAG_F_P_E_$num"
        mv "$dir" "$new_name"
        echo "Renamed $dir to $new_name"
    fi
done

# WAG Sampling Sequence Models
# With DATA2
for dir in WAG_sampling_seq_data2_root_*; do
    if [ -d "$dir" ]; then
        # Extract the numeric part
        num=$(echo $dir | grep -o '[0-9.]*$')
        # Create new name
        new_name="WAG_EP_DATA2_R_$num"
        mv "$dir" "$new_name"
        echo "Renamed $dir to $new_name"
    fi
done

# Without DATA2
for dir in WAG_sampling_seq_root_*; do
    if [ -d "$dir" ]; then
        # Extract the numeric part
        num=$(echo $dir | grep -o '[0-9.]*$')
        # Create new name
        new_name="WAG_EP_R_$num"
        mv "$dir" "$new_name"
        echo "Renamed $dir to $new_name"
    fi
done

# WAG Basic Comparison
# With DATA2
if [ -d "WAG_frequencies_sampling_seq_data2" ]; then
    mv "WAG_frequencies_sampling_seq_data2" "WAG_F_EP_DATA2"
    echo "Renamed WAG_frequencies_sampling_seq_data2 to WAG_F_EP_DATA2"
fi

# Without DATA2
if [ -d "WAG_frequencies_sampling_seq" ]; then
    mv "WAG_frequencies_sampling_seq" "WAG_F_EP"
    echo "Renamed WAG_frequencies_sampling_seq to WAG_F_EP"
fi

echo "Folder renaming completed!" 