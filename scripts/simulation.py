"""
simulation.py
==============
Module principal pour la simulation d’alignements.  
Permet la simulation phylogénétique via Bio++ (BPP).  
Les résultats sont enregistrés dans un dossier dédié et peuvent être accompagnés d’un rapport PDF.
"""

from pathlib import Path
from Bio import Phylo, SeqIO
import subprocess
import random
import re
import sys
import time
from tqdm import tqdm


class BppSimulator:
    """
    Simulation basée sur bppseqgen.
    Attrs:
        align (str): Dossier des alignements d’entrée.
        tree (str): Dossier des arbres phylogénétiques.
        config (str): Fichier de configuration pour bppseqgen.
        output (str): Dossier de sortie pour les alignements simulés.
        ext_rate (float, optional): Taux d’extension des gaps.
    """

    def __init__(self, align, tree, config, output, ext_rate=None):
        self.align_dir = Path(align)
        self.tree_dir = Path(tree)
        self.config = Path(config)
        self.output_dir = Path(output)
        self.ext_rate = ext_rate
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lecture du fichier de configuration
        with open(self.config, 'r') as f:
            rconfig = f.read()

        pat = re.compile(r"\$\(([^,\n\)\s]+)\)")
        self.macros = pat.findall(rconfig)
        for kw in ["IN_SEQ", "TREE", "OUT_SEQ"]:
            if kw not in self.macros:
                raise ValueError(f"Macro {kw} manquante dans {self.config}")

    def pick_longer_alignment(self, align):
        """
        Choisit un alignement plus long que celui fourni, sinon renvoie l'original.
        Args:
            align (Path): Chemin vers l’alignement de référence.
        Returns:
            choice (Path): Chemin vers l’alignement choisi.
        """
        data1_len = len(next(SeqIO.parse(align, "fasta")).seq)
        candidates = [
            f for f in self.align_dir.glob("*.fasta")
            if len(next(SeqIO.parse(f, "fasta")).seq) >= data1_len
        ]
        return random.choice(candidates) if candidates else align

    def simulate(self):
        """
        Lance les simulations d’alignements avec bppseqgen à partir des alignements propres.
        Args:
            None
        Returns:
            None
        """
        start = time.time()
        alignments = sorted(Path(self.align_dir).glob("*.fasta"))
        n = len(alignments)
        print(f"  > Found {n} alignments to simulate.\n")

        if n == 0:
            print("No alignments found for simulation. Exiting.")
            return

        for align_path in tqdm(alignments, desc="Alignments' simulation", unit="fichier"):
            famname = align_path.stem
            try:
                tree_path = Path(self.tree_dir) / f"{famname}.nwk"
                if not tree_path.exists():
                    tqdm.write(f"Tree not found for {famname}, skipping.")
                    continue

                sequences = list(SeqIO.parse(align_path, "fasta"))
                nseq = len(sequences)
                nsite = len(sequences[0].seq) if nseq > 0 else 100

                dargs = {
                    "IN_SEQ": str(align_path),
                    "TREE": str(tree_path),
                    "OUT_SEQ": str(self.output_dir / f"{famname}.fasta"),
                    "NSEQ": nsite
                }

                command = [
                    "bppseqgen",
                    f"param={self.config.resolve()}"
                ] + [f"{key}={value}" for key, value in dargs.items()]

                result = subprocess.run(command, capture_output=True, text=True)

                if result.returncode != 0:
                    tqdm.write(f"Error for {famname}: {result.stderr.strip()}")

            except Exception as e:
                tqdm.write(f"Unexpected error for {famname}: {e}")

        print(f"\nSimulation phase completed in {time.time() - start:.1f}s.")
