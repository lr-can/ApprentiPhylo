from deelogeny_m2.mapping2model import GTR, HKY, estim_seq
import argparse
from pathlib import Path
import os 
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--align_path', help = 'path to directory containing alignments (FASTA or other format).')
    parser.add_argument('--tree_path', help = 'path to the directory containing the corresponding phylogenetic trees (newick format).')
    parser.add_argument('--config', help = 'configuration file used by mapnh (from Bio++) to execute mapping.')
    parser.add_argument('--output', help = 'Path to the directory where intermediate data ans estimated models will be stored.')

    args = parser.parse_args()

    align_path = Path(args.align_path)
    tree_path = Path(args.tree_path)
    config = Path(args.config)
    output = Path(args.output)

    output_map = output / "mapping_data"
    output_model_GTR = output / "mapping_GTR"
    output_model_HKY = output / "mapping_HKY"

    output_model_GTR.mkdir(parents=True)
    output_model_HKY.mkdir(parents=True)
    output_map.mkdir(parents=True)

    for alignment in tqdm(os.listdir(align_path), desc="Mapping alignment", unit="alignment") :
        famname = alignment.split('.')[0]
        treename = famname + '_tree.nwk'
        align = align_path / alignment
        tree = tree_path / treename

        [dcounts, dnorm]=estim_seq(str(align), str(tree), config, str(output_map))

        with open(output_model_GTR / f"{famname}_model.txt", "w") as f :
            f.write(GTR(dcounts,dnorm))
        
        with open(output_model_HKY / f"{famname}_model.txt", "w") as f :
            f.write(HKY(dcounts,dnorm))