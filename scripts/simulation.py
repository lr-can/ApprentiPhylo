import argparse
from pathlib import Path
import os 
from Bio import SeqIO
from deelogeny_m2.simulators import ESMsimulator, Bppsimulator
import shutil

#arguments
parser = argparse.ArgumentParser()
parser.add_argument('--simulator', '-s', nargs= '+', help = "List of simulators to be used ('ESM' or 'BPP')")
parser.add_argument('--output', '-o', help = 'Output directory for simulated alignments.')
parser.add_argument('--tree', '-t', help = "path to directory containing phylogenetic trees.")
parser.add_argument('--external_branch_length', '-e', help = "(optional for BPP) length of external branches to be applied")
parser.add_argument('--root_length', '-r', help = "(optional for BPP) length of root branch to be applied")
parser.add_argument('--align', '-a', help = "directory containing alignments to be used as references.")
parser.add_argument('--tools', help = "path to necessary tools, such as ESM scripts of Apptainer files (.cif).")
parser.add_argument('--config', '-c', nargs='+', help = "(optional for BPP) list of configuration files for BPP simulations.")
parser.add_argument('--modelmapping', '-m', help = "(optional for BPP): path to the directory containing the evolution models to be applied")
parser.add_argument('--gap', default=False, type=bool, help = "Option to add gaps to simulated alignments (False or True).")
args = parser.parse_args()

if 'ESM' in args.simulator :
    ESMsimul = ESMsimulator(align = args.align, 
                            tree = args.tree, 
                            output = args.output, 
                            tools = args.tools)
    ESMsimul.simulate()

if 'BPP' in args.simulator : 
    for config in args.config :
        BPPsimul = Bppsimulator(align = args.align, 
                                tree = args.tree,
                                config = config, 
                                output = args.output, 
                                tools = args.tools,
                                ext_length = args.external_branch_length,
                                root_length = args.root_length,
                                gaps = args.gap,
                                mapping = args.modelmapping)
        BPPsimul.simulate()
