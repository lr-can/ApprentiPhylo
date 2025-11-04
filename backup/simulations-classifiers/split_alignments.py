import os
import sys
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def split_alignment(input_file, output_dir):
    """
    Split a FASTA alignment file into individual sequence files
    
    Args:
        input_file (str): Path to the input FASTA file
        output_dir (str): Path to the output directory
    """
    # Get the base name of the input file (without extension)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Read the FASTA file
    for record in SeqIO.parse(input_file, "fasta"):
        # Create a new sequence record
        new_record = SeqRecord(
            seq=record.seq,
            id=record.id,
            description=record.description
        )
        
        # Create the output file name
        output_file = os.path.join(output_dir, f"{base_name}_{record.id}.fasta")
        
        # Write the single sequence file
        SeqIO.write(new_record, output_file, "fasta")
        print(f"Created file: {output_file}")

def process_fasta_dir(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith('.fasta'):
            input_file = os.path.join(input_dir, filename)
            print(f"\nProcessing file: {input_file}")
            split_alignment(input_file, output_dir)
    print(f"\nAll sequence files have been saved to: {output_dir}")

def process_main_folder(main_folder):
    # Process real/gap_and_ambigless
    real_gap = os.path.join(main_folder, 'real', 'gap_and_ambigless')
    if os.path.exists(real_gap):
        outdir = os.path.join(main_folder, 'real', 'gap_and_ambigless_split_seq')
        print(f"\nProcessing: {real_gap} -> {outdir}")
        process_fasta_dir(real_gap, outdir)
    # Process all subfolders under simulation
    sim_dir = os.path.join(main_folder, 'simulation')
    if os.path.exists(sim_dir):
        sim_outdir = os.path.join(main_folder, 'simulation_split_seq')
        for sub in os.listdir(sim_dir):
            sub_path = os.path.join(sim_dir, sub)
            if os.path.isdir(sub_path):
                outdir = os.path.join(sim_outdir, sub)
                print(f"\nProcessing: {sub_path} -> {outdir}")
                process_fasta_dir(sub_path, outdir)

def main():
    if len(sys.argv) < 2:
        print("Usage: python split_alignments.py data_mammals [data_viridiplantae ...]")
        sys.exit(1)
    for folder in sys.argv[1:]:
        if not os.path.exists(folder):
            print(f"Error: Directory {folder} does not exist")
            continue
        print(f"\n==== Processing main directory: {folder} ====")
        process_main_folder(folder)
    print("\nAll processing completed!")

if __name__ == "__main__":
    main() 