from Bio import Phylo
from Bio import SeqIO
from datetime import datetime
from pathlib import Path
import argparse
import os , sys
import random
import re
import shutil
import subprocess

class AddGap() :
    def __init__(self, empirical, simulate, output) :
        self.empirical = Path(empirical)
        self.simulate = Path(simulate)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.add_gap()
    
    def add_gap(self) :
        for file in os.listdir(self.simulate):
            empirical_file_name = file.split('.')[0] + '.fasta'

            simul_sequences = SeqIO.parse(self.simulate / file,format='fasta')
            empirical_sequences = SeqIO.parse(self.empirical / empirical_file_name, format='fasta')
            
            for nb_seq, sequence in enumerate(simul_sequences) : 
                index_gap = [i for i,char in enumerate(sequence.seq) if char == '-' or char == '_']
                for idx in index_gap :
                    empirical_sequences[nb_seq].seq[idx] = '-'

            SeqIO.write(empirical_sequences, self.output / file, format='fasta')

class ESMsimulator() :
    def __init__(self, align, tree, output, tools) :
        self.align = Path(align)
        self.tree = Path(tree)
        self.output = Path(output)
        self.tools = Path(tools)

        #create directories
        self.outputsim = self.output / 'ESM'
        self.outputsim.mkdir(parents=True, exist_ok=False)

    def simulate(self, gap = False) :
        align_names = os.listdir(self.align)
        for align_name in align_names :
            sequences = list(SeqIO.parse(self.align / align_name, "fasta"))
            n_seq = random.randint(0, len(sequences) - 1)
            seq = sequences[n_seq].seq

            famname = align_name.split('.')[0]
            tree_path = self.tree / f"{famname}.nwk"
            output_file = self.outputsim / f"{famname}.fasta"

            command = f"python {self.tools}/simulatewithesm/src/simulateGillespie.py --tree={tree_path} --rescale=1.0 --output={output_file} --useesm --inputseq={seq} --model-location=esm2_t6_8M_UR50D"
            os.system(command)
            
        print("Simulations Done...")

        if gap == True : 
            print("Adding Gap...")
            AddGap(self.align, self.outputsim, self.output / 'ESM_gap')
            print("Done")
       
class Bppsimulator():
    def __init__(self, args, config):#, ext_length=None, root_length=None, gaps=False, mapping=None):
        self.align_dir = Path(args.align) 
        self.tree_dir = Path(args.tree)
        self.config = Path(config)
        self.output_dir = Path(args.output)

        self.args = args # for other parameters
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.config, 'r') as f:
            rconfig = f.read()

        pat=re.compile("\$\(([^,^\n^\)^ ]+)\)")
        self.lmacros = pat.findall(rconfig)  ## all macros compulsory in command line
        for kw in ["IN_SEQ", "TREE", "OUT_SEQ"]:
          if not kw in self.lmacros:
            print("Error: Macro " + kw + " not in config file " + str(self.config))
            sys.exit(0)
            
    def compute_root_trees(self):
        # For each tree file, add a root branch of specified length and write to output
        for file in os.listdir(self.tree):
            if file.endswith('.nwk'):
                input_path = self.tree / file
                output_path = self.outputtree / file
                
                # Read the original Newick tree string
                with open(input_path, 'r') as f:
                    nwk_str = f.read().strip()
                
                # Remove the semicolon at the end if present
                if nwk_str.endswith(';'):
                    nwk_str = nwk_str[:-1]
                
                # Add the root length as a new branch
                rooted_nwk = f"({nwk_str}:{self.root_length});"
                
                # Write the new rooted tree to file
                with open(output_path, 'w') as f:
                    f.write(rooted_nwk)

    def change_external_length(self, filename):
        # Change the branch length of all terminal nodes (leaves) in the tree
        arbre = Phylo.read(self.tree / filename, 'newick')

        for terminal in arbre.get_terminals():
            terminal.branch_length = self.ext_length
    
        Phylo.write(arbre, self.outputtree / filename, 'newick')

    def compute_new_trees(self):
        # For each tree file, update the external branch lengths
        for file in os.listdir(self.tree):
            self.change_external_length(file)

    def pick_longer_alignment(self, align, rep):
        """ Pick a random an alignment longer than align file in directory rep.
        """
        data1_len = len(next(SeqIO.parse(align, "fasta")).seq)
        candidates = []
        for fname in os.listdir(rep):
          fpath=os.path.join(rep,fname)
          if os.path.isfile(fpath) and fname.endswith('.fasta') and len(next(SeqIO.parse(fpath, "fasta")).seq) >= data1_len:
            candidates.append(fpath)
        return random.choice(candidates) if candidates else align

    def simulate(self):
        align_names = os.listdir(self.align_dir)
        for align_name in align_names:
          align_path = os.path.join(self.align_dir,align_name)
          if not os.path.isfile(align_path) or not align_name.endswith('.fasta'):
            continue

          ## general macros
          dargs={}
          dargs["IN_SEQ"]=align_path
          
          # Choose the matching tree path 
          famname = align_name.split('.')[0]
          dargs["TREE"]= os.path.join(self.tree_dir,f"{famname}.nwk")

          dargs["OUT_SEQ"]= os.path.join(self.output_dir, f"{famname}.fasta")

          ## config specific macros

          if "NSEQ" in self.lmacros:
            parser=SeqIO.parse(align_path, format='fasta')
            lseq=[seq for seq in parser]
            if len(lseq)==0:
              continue
            first_seq = lseq[0]
            nseq = len(first_seq)
            dargs["NSEQ"]=nseq
            
          if "IN_SEQ2" in self.lmacros:
            dargs["IN_SEQ2"]=self.pick_longer_alignment(align_path,self.align_dir)

          if "EXT_RATE" in self.lmacros:
            dargs["EXT_RATE"]=self.args.ext_rate
            
          command = ["bppseqgen", f"param={self.config}"] + [k+"="+str(v) for k,v in dargs.items()]
          try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE)
            print(f'Simulation completed for {famname}')
          except subprocess.CalledProcessError as e:
            print(f"Command failed with error: {e}")
          except FileNotFoundError:
            print("Error: bppseqgen command not found. Check your PATH settings.")
            
        print("Simulations Done...")

        # if self.gaps:
        #     print("Adding Gaps...")
        #     AddGap(self.align, self.outputsim, self.outputsim_gap)
        #     print("Done")

        
if __name__ == "__main__":
   #arguments
   parser = argparse.ArgumentParser()
   parser.add_argument('--align', '-a', help = "directory containing alignments to be used as references.")
   parser.add_argument('--tree', '-t', help = "path to directory containing phylogenetic trees.")
   parser.add_argument('--config', '-c', nargs='+', help = "list of configuration files for BPP simulation to be applied")
   parser.add_argument('--output', '-o', help = 'Output directory for simulated alignments.')
#   parser.add_argument('--simulator', '-s', nargs= '+', help = "List of simulators to be used ('ESM' or 'BPP')")
   parser.add_argument('--ext_rate', '-e', help = "(option for BPP) rate of external branches to be applied")
   # parser.add_argument('--root_length', '-r', help = "(optional for BPP) length of root branch to be applied")
   # parser.add_argument('--tools', help = "path to necessary tools, such as ESM scripts of Apptainer files (.cif).")
   # parser.add_argument('--modelmapping', '-m', help = "(optional for BPP): path to the directory containing the evolution model")
   # parser.add_argument('--gap', default=False, type=bool, help = "Option to add gaps to simulated alignments (False or True).")
   args = parser.parse_args()
   
   for config in args.config :
     BPPsimul = Bppsimulator(args, 
                             config = config                      
                             )
     BPPsimul.simulate()
