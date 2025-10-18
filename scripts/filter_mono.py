"""
filter_mono.py
===============
Filtre et copie les familles dites « mono » (ou spécifiques) depuis un ensemble de données.  
Permet de ne conserver que certaines familles selon un fichier d’identifiants fourni (.txt ou .csv).
"""

from pathlib import Path
import shutil


def copy_mono_files(target_dir, source_dir, fam_file):
    """
    Copie les fichiers .aln correspondant aux FAM IDs spécifiés.
    """
    target_dir = Path(target_dir)
    source_dir = Path(source_dir)
    fam_file = Path(fam_file)

    target_dir.mkdir(parents=True, exist_ok=True)

    fam_ids = {line.strip().split()[0] for line in fam_file.read_text().splitlines() if line.strip()}

    copied_count = 0
    for aln_file in source_dir.glob("*.aln"):
        fam_id = aln_file.stem
        if fam_id in fam_ids:
            shutil.copy2(aln_file, target_dir / aln_file.name)
            copied_count += 1

    print(f"Copied {copied_count} files to {target_dir}")
