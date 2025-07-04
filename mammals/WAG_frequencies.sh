#!/bin/bash
python3 ../scripts/simulation.py \
--tools ../tools \
-s 'BPP' \
-o ../results/mammals_simulations \
-t ../results/trees_mammals_new \
-a ../results/preprocess_mammals_new/gap_and_ambigless \
-c ../config/bpp/aa/classic/WAG_frequencies.bpp \