#!/bin/bash
for ext in 0 0.05 0.1 0.2 0.5; do
  python3 ../scripts/simulation.py \
    --tools ../tools \
    -s 'BPP' \
    -o ../results/mammals_simulations \
    -t ../results/trees_mammals_new \
    -a ../results/preprocess_mammals_new/gap_and_ambigless \
    -c ../config/bpp/aa/classic/WAG_frequencies_posterior_extra_length_data2.bpp \
    -e $ext &
  sleep 10
done
wait

