import os
from Bio import SeqIO
import argparse
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help = "path to directory containing alignment files.")
parser.add_argument('--output', '-o', help = "path to directory where classified results will be saved.")
parser.add_argument('--threshold', '-t', type=float, help = "threshold for average GC rate. alignment with a rate greater than or equal to this threshold will be classified in the high GC rate group.")

args = parser.parse_args()

input_dir = Path(args.input)
output_dir = Path(args.output)
gc_threshold = args.threshold

# Répertoires de sortie pour les alignements à haut et bas taux de GC
high_gc_dir = output_dir / "high_gc_alignments"
low_gc_dir = output_dir / "low_gc_alignments"
high_gc_dir.mkdir(parents=True, exist_ok=True)
low_gc_dir.mkdir(parents=True, exist_ok=True)

# Fonction pour calculer le taux de GC
def calculate_gc_content(sequence):
    gc_count = 0 
    for base in sequence:
        if base in "GC":
            gc_count += 1
    return gc_count / len(sequence)

# Parcours des fichiers dans le répertoire
for filename in tqdm(os.listdir(input_dir), desc='file', unit='file'):
    if filename.endswith(".dna.aln"):
        filepath = os.path.join(input_dir, filename)

        # Lecture des alignements dans le fichier
        with open(filepath) as file:
            records = list(SeqIO.parse(file, "fasta"))
            if len(records) != 0 :
                # Calcul du taux de GC moyen pour l'alignement
                gc_contents = [calculate_gc_content(record.seq) for record in records]
                if len(gc_contents) != 0 : 
                    avg_gc_content = sum(gc_contents) / len(gc_contents)
                else : 
                    avg_gc_content = 0

                # Séparer l'alignement en fonction du taux de GC moyen
                if avg_gc_content >= gc_threshold:
                    # Écrire dans le dossier high_gc
                    with open(high_gc_dir / filename, "w") as high_gc_file:
                        SeqIO.write(records, high_gc_file, "fasta")
                else:
                    # Écrire dans le dossier low_gc
                    with open(low_gc_dir / filename, "w") as low_gc_file:
                        SeqIO.write(records, low_gc_file, "fasta")

print("Séparation des alignements par taux de GC terminée.")
