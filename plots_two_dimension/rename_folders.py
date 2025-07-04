import os
import re

# Root directories to process (relative to this script)
ROOTS = [
    '../results/MPD/viridiplantae_group_results',
]

def rename_rule(name):
    # Special handling
    if name == 'with_data2':
        return 'WITH_DATA2'
    if name == 'without_data2':
        return 'WITHOUT_DATA2'
    # Extract main model name (e.g., WAG, LG08, JTT92, DSO78, etc.)
    m = re.match(r'([A-Za-z0-9]+)', name)
    model = m.group(1).upper() if m else ''
    # F
    f = '_F' if re.search(r'_frequencies|_F', name, re.IGNORECASE) else ''
    # P
    p = '_P' if re.search(r'_posterior|_P', name, re.IGNORECASE) else ''
    # EP
    ep = '_EP' if re.search(r'sampling_seq|_EP', name, re.IGNORECASE) else ''
    # DATA2
    data2 = '_DATA2' if re.search(r'data2', name, re.IGNORECASE) else ''
    # E_{value}
    e = ''
    m = re.search(r'(ext|E)_?([0-9.]+)', name, re.IGNORECASE)
    if m:
        e = f'_E_{m.group(2)}'
    # R_{value}
    r = ''
    m = re.search(r'(root|R)_?([0-9.]+)', name, re.IGNORECASE)
    if m:
        r = f'_R_{m.group(2)}'
    # Concatenate
    new_name = model + f + p + ep + data2 + e + r
    # Remove redundant underscores
    new_name = re.sub(r'__+', '_', new_name)
    new_name = re.sub(r'_+$', '', new_name)
    new_name = re.sub(r'^_', '', new_name)
    return new_name

def batch_rename(root):
    # Recursively traverse all subfolders, depth-first, rename subfolders first
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        for dirname in dirnames:
            old_path = os.path.join(dirpath, dirname)
            new_dirname = rename_rule(dirname)
            new_path = os.path.join(dirpath, new_dirname)
            if new_dirname != dirname:
                print(f'Renaming: {old_path} -> {new_path}')
                try:
                    os.rename(old_path, new_path)
                except Exception as e:
                    print(f'Failed to rename {old_path} to {new_path}: {e}')

if __name__ == '__main__':
    for root in ROOTS:
        if os.path.exists(root):
            batch_rename(root)
        else:
            print(f'Path not found: {root}') 