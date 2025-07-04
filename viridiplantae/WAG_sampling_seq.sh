#!/bin/bash
for root in 0 0.05 0.1 0.2 0.5; do
  python3 ../scripts/simulation.py \
    --tools ../tools \
    -s 'BPP' \
    -o ../results/viridiplantae_simulations \
    -t ../results/trees_viridiplantae_new \
    -a ../results/preprocess_viridiplantae_new/gap_and_ambigless \
    -c ../config/bpp/aa/classic/WAG_sampling_seq.bpp \
    -r $root &
  sleep 10
done
wait