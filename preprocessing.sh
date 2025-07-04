python3 scripts/filter_mono.py \
--source-dir Original_Data/mammals/mammals_prot_results \
--fam-file Original_Data/mammals/fam2nbseqnbspec.mono \
--target-dir data/mammals_new

python3 scripts/filter_mono.py \
--source-dir Original_Data/viridiplantae/viridiplantae_prot_results \
--fam-file Original_Data/viridiplantae/fam2nbseqnbspec.mono \
--target-dir data/viridiplantae_new

python3 scripts/preprocessing_dataset.py \
--input data/mammals_new \
--output results/preprocess_mammals_new \
--minseq 5 \
--maxsites 10000 \
--minsites 10 \
--type aa

python3 scripts/preprocessing_dataset.py \
--input data/viridiplantae_new \
--output results/preprocess_viridiplantae_new \
--minseq 5 \
--maxsites 10000 \
--minsites 10 \
--type aa