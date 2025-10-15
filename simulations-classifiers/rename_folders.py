import os
import re

# Root directories to process
ROOTS = [
    'runs_viridiplantae',
    'runs_mammals',
    'runs_comparsion_mammals_viridiplantae/comparsion_mammals_viridiplantae',
    'viridiplantae_group_results',
]

def rename_rule(name):
    # Special handling for with_data2/without_data2
    if name == 'with_data2':
        return 'WITH_DATA2'
    if name == 'without_data2':
        return 'WITHOUT_DATA2'
    # Convert to uppercase
    name = name.upper()
    # Extract main model name (e.g., WAG, LG08, DSO78, JTT92, etc.)
    m = re.match(r'([A-Z0-9]+)', name)
    prefix = m.group(1) if m else ''
    # F/P/EP
    F = '_F' if '_F' in name else ''
    P = '_P' if '_P' in name else ''
    EP = '_EP' if '_EP' in name else ''
    # DATA2
    DATA2 = '_DATA2' if 'DATA2' in name else ''
    # E_{value}
    E = ''
    mE = re.search(r'_E_([0-9.]+)', name)
    if mE:
        E = f'_E_{mE.group(1)}'
    # R_{value}
    R = ''
    mR = re.search(r'_R_([0-9.]+)', name)
    if mR:
        R = f'_R_{mR.group(1)}'
    # Combine
    # 1. WAG_F_P_DATA2_E_{value}
    if F and P and DATA2 and E:
        return f'{prefix}{F}{P}{DATA2}{E}'
    # 2. WAG_F_P_E_{value}
    if F and P and not DATA2 and E:
        return f'{prefix}{F}{P}{E}'
    # 3. WAG_F
    if F and not P and not EP and not DATA2 and not E and not R:
        return f'{prefix}{F}'
    # 4. WAG_F_EP
    if F and EP and not DATA2 and not E and not R:
        return f'{prefix}{F}{EP}'
    # 5. WAG_F_EP_DATA2
    if F and EP and DATA2 and not E and not R:
        return f'{prefix}{F}{EP}{DATA2}'
    # 6. WAG_EP_DATA2_R_{value}
    if EP and DATA2 and R and not F and not P and not E:
        return f'{prefix}{EP}{DATA2}{R}'
    # 7. WAG_EP_R_{value}
    if EP and R and not DATA2 and not F and not P and not E:
        return f'{prefix}{EP}{R}'
    # 8. Other cases, keep original name and print warning
    print(f'[WARN] Folder name "{name}" does not match any template, keep original.')
    return name

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