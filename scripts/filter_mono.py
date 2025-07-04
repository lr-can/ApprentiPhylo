import os
import shutil
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Copy .aln files whose FAM IDs are listed in a given file.')
parser.add_argument('--target-dir', required=True, help='Target directory to copy files to')
parser.add_argument('--source-dir', required=True, help='Source directory containing .aln files')
parser.add_argument('--fam-file', required=True, help='Path to fam2nbseqnbspec.mono file')
args = parser.parse_args()

target_dir = args.target_dir
source_dir = args.source_dir
fam_file = args.fam_file

# Create target directory if it does not exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Read the first column of the fam2nbseqnbspec.mono file
fam_ids = set()
with open(fam_file, 'r') as f:
    for line in f:
        fam_id = line.strip().split()[0]
        fam_ids.add(fam_id)

# Traverse files in the source directory
copied_count = 0
for filename in os.listdir(source_dir):
    if filename.endswith('.aln'):
        fam_id = filename.split('.')[0]  # Get FAM ID from filename
        if fam_id in fam_ids:
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)
            shutil.copy2(source_path, target_path)
            copied_count += 1

print(f"Copied {copied_count} files to {target_dir}") 