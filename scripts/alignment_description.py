from deelogeny_m2.alignment_descriptor import Descriptor
import argparse

if __name__ == "__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='specifies the path of the directory containing the alignment files')
    parser.add_argument('--output', '-o', help='specifies the path of the directory where the results will be stored')
    args = parser.parse_args()

    desc = Descriptor(args.input, args.output)
    desc.calculate()

