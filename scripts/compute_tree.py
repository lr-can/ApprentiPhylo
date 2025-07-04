import argparse
from deelogeny_m2.computing_trees import Computing_trees

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help = "path to directory containing input alignment files.")
    parser.add_argument('--output', '-o', help = "path to directory where generated phylogenetic trees will be saved.")
    parser.add_argument('--alphabet', '-a', help='aa or nt')
    parser.add_argument('--only', default = None, help = 'path to a file listing the names of specific alignment files to be processed (one per line). If the option is supplied, only these files wille be taken into account.')
    args = parser.parse_args()

    tree_cp = Computing_trees(args.input, args.output, args.alphabet, args.only)
    tree_cp.compute_all_trees()