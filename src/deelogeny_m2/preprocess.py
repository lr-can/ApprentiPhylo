from pathlib import Path
import os
from tqdm import tqdm
from Bio import SeqIO
from Bio.Seq import Seq

class Preprocess :
    def __init__(self, input, output, minseq, maxsites, minsites, type) : 
        self.input = Path(input)
        self.output = Path(output)
        self.minseq = minseq
        self.maxsites = maxsites
        self.minsites = minsites
        self.output.mkdir(parents=True)
        self.logfile = self.output / 'preprocess.log'
        self.clean_output = None
        if type == 'aa' : 
            self.ambiguous = "BZJUOX"
        elif type == 'dna' :
            self.ambiguous = "NDHVBRYKMSWX*"

    def remove_outliers(self) :
        """
        remove files with less than self.minseq sequences, less than self.minsites sites or more than self.maxsites sites.
        """
        files = files = [f for f in os.listdir(self.input) if f.endswith(".aln") or f.endswith(".fasta")] 
        correct_files = []
        empty_files = 0
        too_long = 0
        too_short = 0
        for file in tqdm(files, unit = 'files', desc='Remove outliers') :
            if os.path.getsize(self.input / file) == 0.0 : 
                empty_files += 1
                continue
    
            first_seq = next(SeqIO.parse(self.input / file, "fasta"))
            if len(first_seq.seq) > self.maxsites :
                too_long += 1
                continue

            if len(first_seq.seq) < self.minsites :
                too_short += 1
                continue

            nb_seq = 0
            for _ in SeqIO.parse(self.input / file, "fasta") : #boucle pour éviter de charger toutes les séquences en mémoire
                nb_seq += 1
                if nb_seq >= self.minseq :
                    correct_files.append(file)
                    break 
            
        not_enough_seq = len(files) - too_long - too_short - empty_files - len(correct_files)

        with open(self.logfile,'w') as log :
            log.write("Removing outlier step : \n")
            log.write(f"{len(correct_files)}/{len(files)} files conserved ({(len(correct_files)/len(files))*100:.2f} %) \n")
            log.write(f"\t- Empty files : {empty_files} ({(empty_files/len(files))*100:.2f}%) \n")
            log.write(f"\t- Too long sequences (>{self.maxsites} sites) : {too_long} ({(too_long/len(files))*100:.2f}%)\n")
            log.write(f"\t- Too short sequences (<{self.minsites} sites) : {too_short} ({(too_short/len(files))*100:.2f}%)\n")
            log.write(f"\t- Not enough sequences (<{self.minseq}) : {not_enough_seq} ({(not_enough_seq/len(files))*100:.2f}%)\n")

        return correct_files 
    
    def write_clean_file(self, filename) : 
        fileout = filename.split('.')[0] + '.fasta'
        with open(self.clean_output / fileout, 'w') as f : 
            for record in SeqIO.parse(self.input / filename, format = 'fasta') :
                new_record = SeqIO.SeqRecord(record.seq, id=record.id, description="")
                SeqIO.write(new_record, f, "fasta")

    def preprocessing(self) :
        clean_files = self.remove_outliers()
        self.clean_output = self.output / 'clean_data'
        self.clean_output.mkdir()

        for file in clean_files : 
            self.write_clean_file(file)

    def gaps_seq(self, filename) :
        for i, record in enumerate(SeqIO.parse(self.clean_output / filename, format = 'fasta')) : 
            if i == 0 :
                gap = len(record.seq)*[0]

            for j, char in enumerate(record.seq) :
                if char == '-' :
                    gap[j] = 1
        return gap
        
    def remove_gaps(self) :
        if not self.clean_output : 
            raise Exception("You must run preprocessing before run remove_gap")
        
        self.gapless_output = self.output / 'gap_less'
        self.gapless_output.mkdir()
        
        conserved_site = 0 
        total_site = 0
        for file in tqdm(os.listdir(self.clean_output), unit = 'file', desc = 'Removing sites with gap') :
            gaps = self.gaps_seq(file)
            total_site += len(gaps)
            count_conserved = sum(1 for x in gaps if x == 0)
            conserved_site += count_conserved
            if count_conserved == 0 : 
                continue
            with open(self.gapless_output / file, 'w') as fout : 
                for record in SeqIO.parse(self.clean_output / file, "fasta") :
                    new_seq = [char for i, char in enumerate(record.seq) if gaps[i] == 0]
                    new_seq = Seq(''.join(new_seq))
                    new_record = SeqIO.SeqRecord(new_seq, id=record.id, description="")
                    SeqIO.write(new_record, fout, "fasta")
        with open(self.logfile,'a') as log :
            log.write("Removing gap step : ")
            log.write(f"{conserved_site}/{total_site} sites conserved ({(conserved_site/total_site)*100:.2f} %) \n")

    def ambig_in_align(self, filepath) :
        for i, record in enumerate(SeqIO.parse(filepath, 'fasta')): 
            if i == 0 : 
                ambig = [0]*len(record.seq)

            for j, char in enumerate(record.seq) : 
                if char in self.ambiguous : 
                    ambig[j] = 1
        return ambig


    def remove_ambig_sites(self, where) :
        """where = gapless or clean"""
        if where == 'gapless' :
            self.without_ambig_output = self.output / 'gap_and_ambigless'
            path = self.gapless_output
        elif where == 'clean' :
            self.without_ambig_output = self.output / 'ambigless'
            path = self.clean_output
        self.without_ambig_output.mkdir()

        conserved_site = 0 
        total_site = 0
        for file in tqdm(os.listdir(path), unit='file', desc='Removing sites with ambiguous characters') :
            ambig = self.ambig_in_align(path / file)
            total_site += len(ambig)
            count_conserved = sum(1 for x in ambig if x == 0)
            conserved_site += count_conserved
            if count_conserved == 0 : 
                continue
            with open(self.without_ambig_output / file, 'w') as fout : 
                for record in SeqIO.parse(path / file, 'fasta') :
                    new_seq = [char for i, char in enumerate(record.seq) if ambig[i] == 0]
                    new_seq = Seq(''.join(new_seq))
                    new_record = SeqIO.SeqRecord(new_seq, id=record.id, description="")
                    SeqIO.write(new_record, fout, "fasta")
        with open(self.logfile,'a') as log :
            log.write(f"Removing ambiguous site step on {where} data : ")
            log.write(f"{conserved_site}/{total_site} sites conserved ({(conserved_site/total_site)*100:.2f} %) \n")
