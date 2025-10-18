"""
preprocess.py
==============
Module de prétraitement des alignements multiples.  
Filtre les familles selon leur taille, retire les sites ambigus et les gaps si demandé.  
Produit des alignements nettoyés prêts pour la simulation ou la reconstruction d’arbres.
"""

from pathlib import Path
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
import os


class Preprocess:
    def __init__(self, input_dir, output_dir, minseq, maxsites, minsites, alphabet):
        self.input = Path(input_dir)
        self.output = Path(output_dir)
        self.minseq = minseq
        self.maxsites = maxsites
        self.minsites = minsites
        self.output.mkdir(parents=True, exist_ok=True)
        self.logfile = self.output / 'preprocess.log'

        if alphabet == 'aa':
            self.ambiguous = "BZJUOX"
        elif alphabet == 'dna':
            self.ambiguous = "NDHVBRYKMSWX*"
        else:
            self.ambiguous = ""

        self.clean_output = None

    def remove_outliers(self):
        files = [f for f in self.input.iterdir() if f.suffix in (".aln", ".fasta")]

        correct_files = []
        empty_files = too_long = too_short = 0

        for file in tqdm(files, unit='files', desc='Remove outliers'):
            if file.stat().st_size == 0:
                empty_files += 1
                continue

            first_seq = next(SeqIO.parse(file, "fasta"))
            if len(first_seq.seq) > self.maxsites:
                too_long += 1
                continue
            if len(first_seq.seq) < self.minsites:
                too_short += 1
                continue

            nb_seq = sum(1 for _ in SeqIO.parse(file, "fasta"))
            if nb_seq >= self.minseq:
                correct_files.append(file)

        not_enough_seq = len(files) - too_long - too_short - empty_files - len(correct_files)

        with open(self.logfile, 'w') as log:
            log.write("Removing outlier step : \n")
            log.write(f"{len(correct_files)}/{len(files)} files conserved ({(len(correct_files)/len(files))*100:.2f}%)\n")
            log.write(f"\t- Empty files : {empty_files}\n")
            log.write(f"\t- Too long sequences : {too_long}\n")
            log.write(f"\t- Too short sequences : {too_short}\n")
            log.write(f"\t- Not enough sequences : {not_enough_seq}\n")

        return correct_files

    def write_clean_file(self, filename):
        fileout = filename.stem + '.fasta'
        with open(self.clean_output / fileout, 'w') as f:
            for record in SeqIO.parse(filename, "fasta"):
                new_record = SeqIO.SeqRecord(record.seq, id=record.id, description="")
                SeqIO.write(new_record, f, "fasta")

    def preprocessing(self):
        clean_files = self.remove_outliers()
        self.clean_output = self.output / 'clean_data'
        self.clean_output.mkdir(exist_ok=True)
        for file in clean_files:
            self.write_clean_file(file)

    def gaps_seq(self, filename):
        file_path = Path(filename)
        if not file_path.is_absolute():
            file_path = self.clean_output / file_path
        for i, record in enumerate(SeqIO.parse(file_path, "fasta")):
            if i == 0:
                gap = [0] * len(record.seq)
        for j, char in enumerate(record.seq):
                if char == '-':
                    gap[j] = 1
        return gap

    def remove_gaps(self):
        if not self.clean_output:
            raise Exception("You must run preprocessing before running remove_gaps")
        
        self.gapless_output = self.output / 'gap_less'
        self.gapless_output.mkdir(exist_ok=True)
        
        total_site = 0
        conserved_site = 0

        for file in tqdm(os.listdir(self.clean_output), unit='file', desc='Removing sites with gap'):
            file_name = Path(file).name
            file_path = self.clean_output / file_name

            gaps = self.gaps_seq(file_name)
            total_site += len(gaps)
            count_conserved = sum(1 for x in gaps if x == 0)
            conserved_site += count_conserved
            if count_conserved == 0:
                continue

            with open(self.gapless_output / file_name, 'w') as fout:
                for record in SeqIO.parse(file_path, "fasta"):
                    new_seq = [char for i, char in enumerate(record.seq) if gaps[i] == 0]
                    new_seq = Seq(''.join(new_seq))
                    new_record = SeqIO.SeqRecord(new_seq, id=record.id, description="")
                    SeqIO.write(new_record, fout, "fasta")

        with open(self.logfile, 'a') as log:
            log.write(f"Removing gap step : {conserved_site}/{total_site} sites conserved ({(conserved_site/total_site)*100:.2f} %) \n")


    def ambig_in_align(self, filepath):
        for i, record in enumerate(SeqIO.parse(filepath, 'fasta')):
            if i == 0:
                ambig = [0] * len(record.seq)
            for j, char in enumerate(record.seq):
                if char in self.ambiguous:
                    ambig[j] = 1
        return ambig

    def remove_ambig_sites(self, where):
        if where == 'gapless':
            without_ambig_output = self.output / 'gap_and_ambigless'
            path = self.output / 'gap_less'
        elif where == 'clean':
            without_ambig_output = self.output / 'ambigless'
            path = self.clean_output
        else:
            raise ValueError("where must be 'gapless' or 'clean'")

        without_ambig_output.mkdir(exist_ok=True)
        conserved_site = total_site = 0

        for file in tqdm(path.iterdir(), unit='file', desc='Removing ambiguous sites'):
            ambig = self.ambig_in_align(file)
            total_site += len(ambig)
            count_conserved = sum(1 for x in ambig if x == 0)
            conserved_site += count_conserved

            if count_conserved == 0:
                continue

            with open(without_ambig_output / file.name, 'w') as fout:
                for record in SeqIO.parse(file, 'fasta'):
                    new_seq = Seq(''.join([char for i, char in enumerate(record.seq) if ambig[i] == 0]))
                    new_record = SeqIO.SeqRecord(new_seq, id=record.id, description="")
                    SeqIO.write(new_record, fout, "fasta")

        with open(self.logfile, 'a') as log:
            log.write(f"Removing ambiguous site step on {where} data: {conserved_site}/{total_site} sites conserved ({(conserved_site/total_site)*100:.2f}%)\n")
