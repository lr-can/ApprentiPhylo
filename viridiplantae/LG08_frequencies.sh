#!/bin/bash
python3 ../scripts/simulation.py \
--tools ../tools \
-s 'BPP' \
-o ../results/viridiplantae_simulations \
-t ../results/trees_viridiplantae_new \
-a ../results/preprocess_viridiplantae_new/gap_and_ambigless \
-c ../config/bpp/aa/classic/LG08_frequencies.bpp \

