#Simulation with ESM
python3 ../scripts/simulation.py -s 'ESM' \
-o ../results/simulations \
-t ../data/trees \
-a ../data/preprocess/gap_and_ambigless \
--tools ~/tools

#Simulation with BPP and classic models
###If mapping is needed for GTR or HKY for example 
python3 ../scripts/mapping.py \
--align_path ../data/preprocess/clean_data \
--tree_path ../data/trees \
--output ../results/mapping \
--config ../config/mapping/mapnh.bpp

python3 ../scripts/simulation.py \
--tools ~/tools \
-s 'BPP' \
-o ../results/simulations \
-t ../data/trees \
-a ../data/preprocess/gap_and_ambigless \
-c ../config/bpp/dna/classic/GTR.bpp \
-m ../results/mapping/mapping_GTR

###If mapping is not needed
python3 ../scripts/simulation.py \
--tools ~/tools \
-s 'BPP' \
-o ../results/simulations \
-t ../data/trees \
-a ../data/preprocess/gap_and_ambigless \
-c ../config/bpp/dna/classic/JC69.bpp

###For simulations with the new bpp model (it doesn't work for now)
python3 ../scripts/simulation.py \
--tools ~/tools \
-s 'BPP' \
-o ../results/simulations \
-t ../data/trees \
-a ../data/preprocess/gap_and_ambigless \
-c ../config/bpp/dna/bpp/bpp.bpp \
-e 0.1
#-e for external branch length
