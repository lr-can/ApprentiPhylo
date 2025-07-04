from pathlib import Path
import os
from Bio import SeqIO
from Bio import Phylo
import re
import subprocess
from datetime import datetime
import random
import shutil

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
            tree_path = self.tree / f"{famname}_tree.nwk"
            output_file = self.outputsim / f"{famname}.fasta"

            command = f"python {self.tools}/simulatewithesm/src/simulateGillespie.py --tree={tree_path} --rescale=1.0 --output={output_file} --useesm --inputseq={seq} --model-location=esm2_t6_8M_UR50D"
            os.system(command)
            
        print("Simulations Done...")

        if gap == True : 
            print("Adding Gap...")
            AddGap(self.align, self.outputsim, self.output / 'ESM_gap')
            print("Done")
       
class Bppsimulator():
    def __init__(self, align, tree, config, output, tools, ext_length=None, root_length=None, gaps=False, mapping=None):
        self.align = Path(align)
        self.tree = Path(tree)
        self.config = Path(config)
        self.output = Path(output)
        self.tools = Path(tools)
        self.gaps = gaps
        if mapping: 
            self.mapping = Path(mapping)
        else:
            self.mapping = None

        if ext_length: 
            self.ext_length = float(ext_length)
        else: 
            self.ext_length = None

        if root_length:
            self.root_length = float(root_length)
        else:
            self.root_length = None

        with open(self.config, 'r') as f:
            self.config_template = f.read()
        self.has_data2 = 'input.data2' in self.config_template

        # Create directories
        name = config.split('/')[-1].split('.')[0]
        self.outputsim = self.output / 'BPP' / name

        if self.gaps: 
            self.outputsim_gap = self.output / 'BPP_gap' / name
            
        tree_prefix = 'with_data2_' if self.has_data2 else ''

        if ext_length:
            self.outputsim = self.output / 'BPP' / f"{name}_ext_{ext_length}"
            if self.gaps: 
                self.outputsim_gap = self.output / 'BPP_gap' / f"{name}_ext_{ext_length}"
            self.outputtree = self.output / f'{tree_prefix}extra_trees' / f"tree_{str(ext_length)}"
            self.outputtree.mkdir(parents=True, exist_ok=False)
            print("Computing new trees...")
            self.compute_new_trees()
            print("Done.")

        if root_length:
            self.outputsim = self.output / 'BPP' / f"{name}_root_{root_length}"
            if self.gaps: 
                self.outputsim_gap = self.output / 'BPP_gap' / f"{name}_root_{root_length}"
            self.outputtree = self.output / f'{tree_prefix}root_trees' / f"root_tree{root_length}"
            self.outputtree.mkdir(parents=True, exist_ok=False)
            print("Computing root trees...")
            self.compute_root_trees()
            print("Done.")

        if ext_length and root_length:
            self.outputsim = self.output / 'BPP' / f"{name}_ext_{ext_length}_root_{root_length}"
            if self.gaps: 
                self.outputsim_gap = self.output / 'BPP_gap' / f"{name}_ext_{ext_length}_root_{root_length}"

        self.outputsim.mkdir(parents=True, exist_ok=False)

        if self.gaps:
            self.outputsim_gap.mkdir(parents=True, exist_ok=False)

        # Set up environment variables for BPP
        os.environ["PATH"] = "/home/lpengjun/devel/bpp/bppsuite/bppSuite:" + os.environ.get("PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join([
            "/home/lpengjun/devel/bpp/bpp-phyl/build/src",
            "/home/lpengjun/devel/bpp/bpp-seq/build/src",
            "/home/lpengjun/devel/bpp/bpp-core/build/src",
            os.environ.get("LD_LIBRARY_PATH", "")
        ])

    def compute_root_trees(self):
        for file in os.listdir(self.tree):
            if file.endswith('.nwk'):
                input_path = self.tree / file
                output_path = self.outputtree / file
                
                # Read the original file
                with open(input_path, 'r') as f:
                    nwk_str = f.read().strip()
                
                # Remove the semicolon if present
                if nwk_str.endswith(';'):
                    nwk_str = nwk_str[:-1]
                
                # Add the root length
                rooted_nwk = f"({nwk_str}:{self.root_length});"
                
                # Write to new file
                with open(output_path, 'w') as f:
                    f.write(rooted_nwk)

    def change_external_length(self, filename):
        arbre = Phylo.read(self.tree / filename, 'newick')

        for terminal in arbre.get_terminals():
            terminal.branch_length = self.ext_length
    
        Phylo.write(arbre, self.outputtree / filename, 'newick')

    def compute_new_trees(self):
        for file in os.listdir(self.tree):
            self.change_external_length(file)

    def pick_data2(self, data1_path):
        data1_dir = data1_path.parent
        data1_len = len(next(SeqIO.parse(data1_path, "fasta")).seq)
        candidates = []
        for fname in os.listdir(data1_dir):
            if fname.endswith('.fasta') and fname != data1_path.name:
                fpath = data1_dir / fname
                if len(next(SeqIO.parse(fpath, "fasta")).seq) > data1_len:
                    candidates.append(fpath)
        return random.choice(candidates) if candidates else data1_path

    def simulate(self):
        align_names = os.listdir(self.align)

        date = datetime.now()
        # Get result type (e.g., BPP) and main name (e.g., WAG_frequencies_posterior_extra_length_data2)
        result_type = self.outputsim.parent.name  # BPP
        main_name = self.outputsim.name  # WAG_frequencies_posterior_extra_length_data2
        # Simplify main name: WAG_frequencies_posterior_extra_length_data2 -> WAG_FPEL_data2
        parts = main_name.split('_')
        if len(parts) >= 4:
            short_main = f"{parts[0]}_{''.join([p[0].upper() for p in parts[1:4]])}"  # WAG_FPEL
            if len(parts) > 4:
                short_main += '_' + '_'.join(parts[4:])
        else:
            short_main = main_name.replace('frequencies', 'F').replace('posterior', 'P').replace('extra', 'E').replace('length', 'L')
        time_str = date.strftime("%m%d_%H%M%S")
        rep_name = f"config_{result_type}_{short_main}_{time_str}"
        temp_config_rep = Path(self.output / rep_name)
        temp_config_rep.mkdir(parents=True, exist_ok=False)
        
        for align_name in align_names:
            famname = align_name.split('.')[0]
            if self.ext_length or self.root_length:
                tree_path = self.outputtree / f"{famname}_tree.nwk"
            else:
                tree_path = self.tree / f"{famname}_tree.nwk"
            output_file = self.outputsim / f"{famname}.fasta"

            align_path = self.align / align_name
            first_seq = next(SeqIO.parse(align_path, format='fasta'))
            nseq = len(first_seq)

            config_content = self.config_template
            config_content = re.sub('tree_path', str(tree_path), config_content)
            config_content = re.sub('output_path', str(output_file), config_content)
            config_content = re.sub('align_path', str(align_path), config_content)
            config_content = re.sub('nseq', str(nseq), config_content)

            if self.has_data2:
                data2_path = self.pick_data2(align_path)
                config_content = re.sub('align2_path', str(data2_path), config_content)

            config_file = temp_config_rep / f"{famname}_config.bpp"
            with open(config_file, 'w') as f:
                f.write(config_content)

            command = ["bppseqgen", f"param={config_file}"]
            if self.mapping:
                with open(self.mapping / f"{famname}_model.txt", 'r') as f:
                    model_mapping = f.read().strip()
                    command.append(f"MODEL={model_mapping}")

            try:
                subprocess.run(command, check=True)
                print(f'Simulation completed for {famname}\n')
            except subprocess.CalledProcessError as e:
                print(f"Command failed with error: {e}")
            except FileNotFoundError:
                print("Error: bppseqgen command not found. Check your PATH settings.")
        
        print("Simulations Done...")

        if self.gaps:
            print("Adding Gaps...")
            AddGap(self.align, self.outputsim, self.outputsim_gap)
            print("Done")

        # shutil.rmtree(temp_config_rep)

        