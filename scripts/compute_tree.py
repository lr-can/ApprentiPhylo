"""
compute_tree.py
================
Construit des arbres phylogénétiques à partir d’alignements simulés ou réels.  
Utilise des outils externes (ex. FastTree) et sauvegarde les arbres au format Newick (.nw).  
Peut traiter un répertoire complet ou un fichier unique.
"""

from pathlib import Path
import subprocess
from tqdm import tqdm
from Bio import Phylo


class ComputingTrees:
    def __init__(self, inputdir, outputdir, alphabet, only=None):
        self.input = Path(inputdir)
        self.output = Path(outputdir)
        self.alphabet = alphabet
        self.only = Path(only) if only else None
        self.output.mkdir(parents=True, exist_ok=True)

    def compute_tree(self, inputname, outputname):
        """
        Calcule un arbre phylogénétique avec FastTree.
        """
        fastaname = self.input / inputname

        if self.alphabet == 'nt':
            cmd = ['fasttree', '-gtr', '-gamma', '-nt', str(fastaname)]
        elif self.alphabet == 'aa':
            cmd = ['fasttree', '-lg', '-gamma', str(fastaname)]
        else:
            raise ValueError("Alphabet must be 'nt' or 'aa'")

        results = subprocess.run(cmd, capture_output=True, text=True)

        if results.returncode == 0:
            arbre_fichier = self.output / outputname
            arbre_fichier.write_text(results.stdout)
        else:
            print(f"Error computing tree for: {fastaname}")

    def compute_all_trees(self):
        """
        Calcule les arbres pour tous les fichiers de l’entrée.
        """
        files = [f.name for f in self.input.iterdir() if f.is_file()]

        if self.only:
            only_files = [line.strip() for line in self.only.read_text().splitlines()]
            files = [f for f in files if f in only_files]

        for file in tqdm(files, desc="Computing trees", unit='file'):
            famname = file.split('.')[0]
            outputname = f"{famname}.nw"
            self.compute_tree(file, outputname)
