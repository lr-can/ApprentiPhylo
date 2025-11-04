"""
simulation.py
==============
Module principal pour la simulation d’alignements.  
Permet deux modes :
 - 'esm' : simulation avec un modèle de langage de protéines (ESM).
 - 'bpp' : simulation phylogénétique via Bio++ (BPP).  
Les résultats sont enregistrés dans un dossier dédié et peuvent être accompagnés d’un rapport PDF.
"""

from pathlib import Path
from Bio import Phylo, SeqIO
import subprocess
import random
import re
import sys
import time


class AddGap:
    """Ajoute les motifs de gaps des alignements empiriques aux simulations."""

    def __init__(self, empirical, simulate, output):
        self.empirical = Path(empirical)
        self.simulate = Path(simulate)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.add_gap()

    def add_gap(self):
        for file in self.simulate.iterdir():
            if file.suffix != ".fasta":
                continue

            empirical_file = self.empirical / f"{file.stem}.fasta"
            if not empirical_file.exists():
                print(f"Empirical file missing for {file.name}")
                continue

            simul_sequences = list(SeqIO.parse(file, "fasta"))
            empirical_sequences = list(SeqIO.parse(empirical_file, "fasta"))

            for nb_seq, sequence in enumerate(simul_sequences):
                index_gap = [i for i, char in enumerate(sequence.seq) if char in {'-', '_'}]
                if nb_seq < len(empirical_sequences):
                    seq_list = list(empirical_sequences[nb_seq].seq)
                    for idx in index_gap:
                        if idx < len(seq_list):
                            seq_list[idx] = '-'
                    empirical_sequences[nb_seq].seq = ''.join(seq_list)

            SeqIO.write(empirical_sequences, self.output / file.name, "fasta")


class ESMsimulator:
    """Simulation basée sur ESM (Evolutionary Scale Modeling)."""

    def __init__(self, align, tree, output, tools):
        self.align = Path(align)
        self.tree = Path(tree)
        self.output = Path(output)
        self.tools = Path(tools)

        self.outputsim = self.output / 'ESM'
        self.outputsim.mkdir(parents=True, exist_ok=True)

    def simulate(self, gap=False):
        for align_name in self.align.iterdir():
            if align_name.suffix != ".fasta":
                continue

            sequences = list(SeqIO.parse(align_name, "fasta"))
            if not sequences:
                continue

            n_seq = random.randint(0, len(sequences) - 1)
            seq = sequences[n_seq].seq

            famname = align_name.stem
            tree_path = self.tree / f"{famname}.nwk"
            output_file = self.outputsim / f"{famname}.fasta"

            command = [
                "python",
                str(self.tools / "simulatewithesm" / "src" / "simulateGillespie.py"),
                f"--tree={tree_path}",
                "--rescale=1.0",
                f"--output={output_file}",
                "--useesm",
                f"--inputseq={seq}",
                "--model-location=esm2_t6_8M_UR50D"
            ]
            subprocess.run(command, check=True)

        print("Simulations Done...")

        if gap:
            print("Adding gaps...")
            AddGap(self.align, self.outputsim, self.output / 'ESM_gap')
            print("Done.")


class BppSimulator:
    """Simulation basée sur bppseqgen."""

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
        """Choisit un alignement plus long que celui fourni, sinon renvoie l'original."""
        data1_len = len(next(SeqIO.parse(align, "fasta")).seq)
        candidates = [
            f for f in self.align_dir.glob("*.fasta")
            if len(next(SeqIO.parse(f, "fasta")).seq) >= data1_len
        ]
        return random.choice(candidates) if candidates else align

    def simulate(self):
        """
        Lance les simulations d’alignements avec bppseqgen à partir des alignements propres.
        """
        start = time.time()
        alignments = sorted(Path(self.align_dir).glob("*.fasta"))
        n = len(alignments)
        print(f"  > Found {n} alignments to simulate.")

        if n == 0:
            print("⚠️ Aucun alignement trouvé, arrêt de la simulation.")
            return

        for i, align_path in enumerate(alignments, start=1):
            famname = align_path.stem
            try:
                tree_path = Path(self.tree_dir) / f"{famname}.nwk"
                if not tree_path.exists():
                    print(f"  ⚠️ Tree not found for {famname}, skipping.")
                    continue

                # Lire le nombre de sites dans l’alignement
                sequences = list(SeqIO.parse(align_path, "fasta"))
                nseq = len(sequences)
                nsite = len(sequences[0].seq) if nseq > 0 else 100  # fallback

                # Préparer les macros pour bppseqgen
                dargs = {
                    "IN_SEQ": str(align_path),
                    "TREE": str(tree_path),
                    "OUT_SEQ": str(self.output_dir / f"{famname}.fasta"),
                    "NSEQ": nsite
                }

                # Construire la commande
                command = [
                    "bppseqgen",
                    f"param={self.config.resolve()}"
                ] + [f"{key}={value}" for key, value in dargs.items()]


                print(f"[{i}/{n}] Simulating {famname}...")
                print("  Running:", " ".join(command))  # affichage de debug

                result = subprocess.run(command, capture_output=True, text=True)

                if result.returncode != 0:
                    print(f"❌ Error for {famname}: {result.stderr.strip()}")
                else:
                    print(f"✅ Simulation done for {famname}")

            except Exception as e:
                print(f"❌ Unexpected error for {famname}: {e}")

        print(f"\nSimulation phase completed in {time.time() - start:.1f}s.")