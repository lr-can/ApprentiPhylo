import os
from pathlib import Path
from Bio import AlignIO
import numpy as np
import plotly.express as px
import plotly.io as pio
from tqdm import tqdm

class Descriptor :
    def __init__(self, input, output) :
        self.input = Path(input)
        self.output = Path(output)
        self.resumefile = self.output / 'data.tsv'
        self.create_directories()

    def create_directories(self) :
        self.output.mkdir(parents=True)

    def calculate_identity_percentage(self, alignment) :
        """
        Calculate the percentage of columns with only one character (including gaps)
        """
        ncol = alignment.get_alignment_length()
        idcol = 0

        for i in range(ncol):
            col = alignment[:,i]
            diff_char = set(col.replace("-",""))
            if len(diff_char) == 1 : 
                idcol += 1
        
        return (idcol / ncol) * 100
    
    def calculate_gap_percentage(self, alignment) :
        """
        Calculate the percentage of gaps in the alignment
        """
        ncol = alignment.get_alignment_length()
        ntot = ncol * len(alignment)
        gap_count = sum(str(record.seq).count("-") for record in alignment)
        return (gap_count / ntot) * 100
    
    def identity_percentages_file(self, identity_percentages) :
        """
        Plot the distribution of identity percentages
        """
        fig1 = px.histogram(
            x=identity_percentages,
            nbins=30,
            title="Distribution des pourcentages d'identité",
            labels={'x': "Pourcentage d'identité", 'y': "Fréquence"},
            template="plotly_white"
        )
        fig1.update_layout(xaxis_title="Pourcentage d'identité", yaxis_title="Fréquence")
        pio.write_image(fig1, file=self.output / "distribution_identite.png")

    def gap_distribution_file(self, gap_percentages) :
        """
        Plot the distribution of gap percentages
        """
        fig2 = px.histogram(
            x=gap_percentages,
            nbins=30,
            title="Distribution des pourcentages de gaps",
            labels={'x': "Pourcentage de gaps", 'y': "Fréquence"},
            template="plotly_white"
        )
        fig2.update_layout(xaxis_title="Pourcentage de gaps", yaxis_title="Fréquence")
        pio.write_image(fig2, file=self.output / "distribution_gaps.png")

    def distribution_length_file(self, align_lengths) :
        """
        Plot the distribution of alignment lengths
        """
        fig3 = px.histogram(
            x=align_lengths,
            nbins=50,
            title="Distribution des longueurs d'alignement",
            labels={'x': "Longueur de l'alignement", 'y': "Fréquence"},
            template="plotly_white"
        )
        fig3.update_layout(xaxis_title="Longueur de l'alignement", yaxis_title="Fréquence")
        pio.write_image(fig3, file=self.output / "distribution_longueurs.png")

    def num_sequences_file(self, num_sequences): 
        """
        Plot the distribution of number of sequences
        """
        fig4 = px.histogram(
            x=num_sequences,
            nbins=20,
            title="Nombre de séquences par alignement",
            labels={'x': "Nombre de séquences", 'y': "Fréquence"},
            template="plotly_white"
        )
        fig4.update_layout(xaxis_title="Nombre de séquences", yaxis_title="Fréquence")
        pio.write_image(fig4, file=self.output / "distribution_sequences.png")
    
    def calculate(self) :
        """
        Calculate the alignment descriptors
        """
        print("computing...")
        files = [f for f in os.listdir(self.input) if f.endswith(".aln") or f.endswith(".fasta")] 

        identity_percentages = []
        gap_percentages = []
        align_lengths = []
        num_sequences = []

        with open(self.resumefile,'w') as f :
            f.write(f"name\tidentity\tgap\tlength\tnseq\n")

        for file in tqdm(files, unit='file'):
            if os.path.getsize(self.input / file) > 0 : 
                alignment = AlignIO.read(self.input / file, "fasta")
                identity_percentages.append(self.calculate_identity_percentage(alignment))
                gap_percentages.append(self.calculate_gap_percentage(alignment))
                align_lengths.append(alignment.get_alignment_length())
                num_sequences.append(len(alignment))

            with open(self.resumefile,'a') as f :
                f.write(f"{file}\t{identity_percentages[-1]:.2f}\t{gap_percentages[-1]:.2f}\t{align_lengths[-1]}\t{num_sequences[-1]}\n")
        
        self.identity_percentages_file(identity_percentages)
        self.gap_distribution_file(gap_percentages)
        self.distribution_length_file(align_lengths)
        self.num_sequences_file(num_sequences)
        print("done")




