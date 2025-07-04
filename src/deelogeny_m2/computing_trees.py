from pathlib import Path
import subprocess
import os
from tqdm import tqdm

class Computing_trees :
    def __init__(self, inputdir, outputdir, alphabet, only = None) :
        self.input = Path(inputdir)
        self.output = Path(outputdir)
        self.alphabet = alphabet
        if only :
            self.only = Path(only)
        else :
            self.only = None

        self.output.mkdir(parents=True)

    
    def compute_tree(self, inputname, outputname) :
        """
        Compute a tree with FastTree
        """
        fastaname = self.input / inputname

        if self.alphabet == 'nt' :
            cmd = ['fasttree', '-gtr', '-gamma','-nt', fastaname]
        elif self.alphabet == 'aa' :
            cmd = ['fasttree', '-lg', '-gamma', fastaname]

        results = subprocess.run(cmd, capture_output=True, text = True)

        if results.returncode == 0 :
            with open(self.output / outputname, 'w') as f :
                f.write(results.stdout)
        else :
            print(f"Error : {self.input / inputname}")


    def compute_all_trees(self) :
        """
        Compute all trees
        """
        files = os.listdir(self.input)

        if self.only : 
            with open(self.only) as f :
                lines = f.readlines()
                only_files = [line.strip() for line in lines]
                files = [file for file in files if file in only_files]
        
        for file in tqdm(files, desc="trees computing", unit='file') :
            famname = file.split('.')[0]
            outputname = f"{famname}_tree.nwk"
            self.compute_tree(file, outputname)        
