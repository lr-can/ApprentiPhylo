import argparse
from deelogeny_m2.preprocess import Preprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help = 'path to output directory where results will be saved.')
    parser.add_argument('--output', '-o', help = 'path to output direcory where results will be saved')
    parser.add_argument('--minseq', '-s', type=int, help = "Minimum number of sequences required to keep the alignment.")
    parser.add_argument('--maxsites',type=int, help = "Maximum number of sites required to keep the alignment.")
    parser.add_argument('--minsites',type=int, help = "Minimum number of sites required to keep the alignment.")
    parser.add_argument('--type', help='aa or dna')
    args = parser.parse_args()

    pr = Preprocess(input = args.input,
                    output = args.output,
                    minseq = args.minseq,
                    maxsites = args.maxsites,
                    minsites = args.minsites,
                    type = args.type)

    pr.preprocessing()
    pr.remove_gaps()
    pr.remove_ambig_sites('gapless')
    pr.remove_ambig_sites('clean')
