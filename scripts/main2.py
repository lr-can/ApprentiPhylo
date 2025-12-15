"""
main2.py
========
Unified pipeline :
- Simulation (preprocess → simulate → trees → metrics)
- Classification (run1 or run1+run2 depending on user option)

Logging goes into logs/pipeline_log.csv.
"""

import argparse
import time
import csv
import os
from pathlib import Path
import logging
import yaml
import sys
import json


from preprocess import Preprocess
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from simulation import BppSimulator
from compute_tree import ComputingTrees
from classification import run_classification
from analyse_classif import process_classification_results
from fix_logreg_history import generate_logreg_train_history
from dashboard2 import run_dashboard
from analyse_predictions_plotly import main as generate_plotly_plots
from phylo_metrics import compute_metrics_for_pair


logging.basicConfig(level=logging.INFO)


# === YAML CONFIG LOADING ===
def load_yaml_config(yaml_path):
    """Load configuration from YAML file."""
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading YAML config: {e}")
        sys.exit(1)


def check_yaml_conflict(args, parser):
    """Check if --yaml is used with other flags (which is not allowed)."""
    if hasattr(args, 'yaml') and args.yaml:
        # Get parser defaults to compare
        defaults = {}
        for action in parser._actions:
            if action.dest not in ['help', 'yaml', 'command', 'func']:
                defaults[action.dest] = action.default
        
        # Check which arguments were explicitly set by user (differ from defaults)
        explicit_args = []
        for arg, value in vars(args).items():
            if arg in ['yaml', 'command', 'func']:
                continue
            
            # Compare with default value
            default_val = defaults.get(arg)
            if value != default_val:
                explicit_args.append(arg)
        
        if explicit_args:
            print("ERROR: You cannot use --yaml with other flags.")
            print("Choose one of:")
            print("  1. Use --yaml config.yaml (loads all settings from YAML)")
            print("  2. Use individual flags (--pre-input, --pre-output, etc.)")
            print(f"\nConflicting flags detected: {', '.join(explicit_args)}")
            sys.exit(1)


def apply_yaml_config(args, config, command):
    """Apply YAML configuration to args namespace."""
    if command not in config:
        print(f"ERROR: Command '{command}' not found in YAML config.")
        sys.exit(1)
    
    cmd_config = config[command]
    for key, value in cmd_config.items():
        setattr(args, key.replace('-', '_'), value)


# === LOGGING ===
def log_step(step_name, args_dict, status, start_time):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "pipeline_log.csv"
    duration = round(time.time() - start_time, 2)

    write_header = not log_file.exists()
    with open(log_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "status", "duration", "args"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "step": step_name,
            "status": status,
            "duration": duration,
            "args": str(args_dict)
        })


# === HELPER FUNCTIONS ===
def find_default_bpp_config(alphabet="aa"):
    """
    Find default BPP configuration file in backup directory or project root.
    Prioritizes WAG_frequencies.bpp and files with WAG or frequencies in name.
    Returns the first matching .bpp file found, or None if not found.
    """
    project_root = Path(__file__).parent.parent
    backup_dir = project_root / "backup"
    
    # Try common paths in backup first
    possible_paths = [
        backup_dir / "config" / "bpp" / alphabet / "WAG_frequencies.bpp",
        backup_dir / "config" / "bpp" / f"{alphabet}_config.bpp",
        backup_dir / "config" / f"{alphabet}.bpp",
        backup_dir / f"config_{alphabet}.bpp",
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    # Search for WAG_frequencies.bpp specifically (highest priority)
    for wag_file in project_root.rglob("WAG_frequencies.bpp"):
        return str(wag_file)
    
    # Search for files with "WAG" or "frequencies" in name (high priority)
    exclude_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache"}
    wag_candidates = []
    for bpp_file in project_root.rglob("*.bpp"):
        if any(excluded in bpp_file.parts for excluded in exclude_dirs):
            continue
        file_name_lower = bpp_file.name.lower()
        if "wag" in file_name_lower or "frequencies" in file_name_lower:
            wag_candidates.append(bpp_file)
    
    if wag_candidates:
        # Prefer files in config directories
        for candidate in wag_candidates:
            parts_lower = [str(p).lower() for p in candidate.parts]
            if any("config" in part for part in parts_lower):
                return str(candidate)
        # Return first WAG/frequencies file found
        return str(wag_candidates[0])
    
    # Search recursively in backup (medium priority)
    if backup_dir.exists():
        for bpp_file in backup_dir.rglob("*.bpp"):
            return str(bpp_file)
    
    # If not found in backup, search in project root (excluding some directories)
    # Prefer files in config directories
    config_candidates = []
    for bpp_file in project_root.rglob("*.bpp"):
        if any(excluded in bpp_file.parts for excluded in exclude_dirs):
            continue
        parts_lower = [str(p).lower() for p in bpp_file.parts]
        if any("config" in part for part in parts_lower) or any("bpp" in part for part in parts_lower):
            config_candidates.append(bpp_file)
    
    if config_candidates:
        return str(config_candidates[0])
    
    # Last resort: return any .bpp file found (but skip Examples directories)
    for bpp_file in project_root.rglob("*.bpp"):
        if any(excluded in bpp_file.parts for excluded in exclude_dirs):
            continue
        # Skip Examples directories as they are usually not suitable for simulation
        if "Examples" in bpp_file.parts:
            continue
        return str(bpp_file)
    
    return None


# === SIMULATION ===
def simulate_cmd(args):
    global_start = time.time()

    try:
        print("\n[1/3] Preprocessing input alignments...")
        pr = Preprocess(
            input_dir=args.pre_input,
            output_dir=args.pre_output,
            minseq=args.minseq,
            maxsites=args.maxsites,
            minsites=args.minsites,
            alphabet=args.alphabet,
        )
        pr.preprocessing()
        pr.remove_gaps()
        pr.remove_ambig_sites("gapless")
        pr.remove_ambig_sites("clean")

        clean_align_dir = Path(args.align)
        print(f"Clean alignments ready in: {clean_align_dir}")

        print("\n[2/3] Running simulations...")
        config_file = args.config[0] if isinstance(args.config, list) else args.config
        
        if not config_file:
            raise ValueError("--config is required. Please provide a BPP configuration file path.")
        
        config_path = Path(config_file)
        # Try to resolve relative paths
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"BPP config file not found: {config_file}\n"
                f"Resolved path: {config_path}\n"
                "Please provide a valid --config path."
            )
        
        config_file = str(config_path.resolve())
        print(f"Using config file: {config_file}")
        bpp_sim = BppSimulator(
            align=str(clean_align_dir),
            tree=str(args.tree),
            config=str(config_file),
            output=str(args.sim_output),
            ext_rate=args.ext_rate
        )
        bpp_sim.simulate()

        print("\n[3/3] Computing trees...")
        tree_cp = ComputingTrees(args.sim_output, args.tree_output, args.alphabet)
        tree_cp.compute_all_trees()

        log_step("simulate_pipeline", vars(args), "success", global_start)
        print("\nSimulation pipeline completed successfully!")

    except Exception as e:
        log_step("simulate_pipeline", vars(args), f"error: {e}", global_start)
        print(f"\nSimulation pipeline failed: {e}")
        raise


# === METRICS ===
def run_metrics(args):
    emp_dir = os.path.abspath(args.empirical)
    sim_dir = os.path.abspath(args.simulation)
    outdir = os.path.abspath(args.output)
    os.makedirs(outdir, exist_ok=True)

    emp_files = sorted([f for f in os.listdir(emp_dir) if f.endswith('.fasta')])
    sim_files = sorted([f for f in os.listdir(sim_dir) if f.endswith('.fasta')])

    # match by prefix instead of exact name
    matching = []
    for e in emp_files:
        prefix = os.path.splitext(e)[0]
        for s in sim_files:
            if os.path.splitext(s)[0] == prefix:
                matching.append((os.path.join(emp_dir, e), os.path.join(sim_dir, s)))
                break

    pbar = tqdm(total=len(matching), desc="Metrics", ncols=80)
    results = []

    # calcul des métriques en parallèle
    with ProcessPoolExecutor(max_workers=args.threads) as ex:
        futs = [ex.submit(compute_metrics_for_pair, e, s, outdir) for e, s in matching]
        for f in as_completed(futs):
            basename, mpd, n_leaves = f.result()
            results.append({
                "File": basename,
                "MPD": mpd if mpd is not None else "NA",
                "n_leaves": n_leaves
            })
            pbar.update(1)

    pbar.close()

    res_dir = os.path.join(outdir, "metrics_results")
    os.makedirs(res_dir, exist_ok=True)
    outfile = os.path.join(res_dir, "mpd_results.csv")

    # écrire toutes les lignes dans le CSV
    with open(outfile, "w", newline="") as csvfile:
        fieldnames = ["File", "MPD", "n_leaves"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    logging.info(f"MPD results written to: {outfile}")


# === CLASSIFICATION ===
def classify_cmd(args):
    start = time.time()

    try:
        print("\n[1/3] Running classification pipeline...")
        
        # Prepare simulation config for run2 if provided
        sim_config_2 = None
        if args.two_iterations:
            # First, check if sim_config_2 is already a dict (from YAML)
            if hasattr(args, 'sim_config_2') and args.sim_config_2 is not None:
                if isinstance(args.sim_config_2, dict):
                    sim_config_2 = args.sim_config_2
                    print("Using sim_config_2 from YAML config (dict)")
                elif isinstance(args.sim_config_2, str):
                    # It's a path to a JSON file
                    if Path(args.sim_config_2).exists():
                        with open(args.sim_config_2, "r") as f:
                            sim_config_2 = json.load(f)
                        print(f"Loaded simulation config for run2 from {args.sim_config_2}")
                    else:
                        print(f"Warning: sim_config_2 path not found: {args.sim_config_2}")
            # If not set yet, try individual arguments
            if sim_config_2 is None and args.sim_tree_2 and args.sim_alphabet_2:
                # Build config from individual arguments
                if not args.config:
                    raise ValueError("--config is required when building sim_config from individual arguments")
                sim_config_2 = {
                    "config": args.config,
                    "tree": args.sim_tree_2,
                    "alphabet": args.sim_alphabet_2,
                }
                if args.sim_ext_rate_2 is not None:
                    sim_config_2["ext_rate"] = args.sim_ext_rate_2
                print(f"Built simulation config for run2 from arguments")
            else:
                # Check if sim_config_2 was provided via YAML (as a dict or path)
                if hasattr(args, 'sim_config_2') and args.sim_config_2:
                    # Could be a dict (from YAML) or a path string
                    if isinstance(args.sim_config_2, dict):
                        sim_config_2 = args.sim_config_2
                        print("Using sim_config_2 from YAML config")
                    elif isinstance(args.sim_config_2, str) and Path(args.sim_config_2).exists():
                        # It's a path to a JSON file
                        with open(args.sim_config_2, "r") as f:
                            sim_config_2 = json.load(f)
                        print(f"Loaded simulation config for run2 from {args.sim_config_2}")
                    else:
                        # Try to load from run_1/store_1 (from a previous run)
                        output_path = Path(args.output) if args.output else Path("results/classification")
                        store_config_path = output_path / "run_1" / "store_1" / "sim_config.json"
                        
                        if store_config_path.exists():
                            print(f"Found simulation config in {store_config_path}")
                            with open(store_config_path, "r") as f:
                                sim_config_2 = json.load(f)
                        else:
                            print("Warning: No simulation config provided for run2. New simulations will not be generated.")
                else:
                    # Check if config exists in run_1/store_1 (from a previous run)
                    output_path = Path(args.output) if args.output else Path("results/classification")
                    store_config_path = output_path / "run_1" / "store_1" / "sim_config.json"
                    
                    if store_config_path.exists():
                        print(f"Found simulation config in {store_config_path}")
                        with open(store_config_path, "r") as f:
                            sim_config_2 = json.load(f)
                    else:
                        print("Warning: No simulation config provided for run2. New simulations will not be generated.")
        
        run_classification(
            realali=args.real_align,
            simali=args.sim_align,
            output=args.output,
            config=args.config,
            tools=args.tools,
            two_iterations=args.two_iterations,
            threshold=args.threshold,
            sim_config_2=sim_config_2
        )

        print("\nClassification pipeline (iterations) completed.")

        # Generate interactive Plotly visualizations (always run)
        print("\n[2/4] Generating interactive Plotly visualizations...")
        try:
            generate_plotly_plots()
        except Exception as e:
            print(f"⚠️  Warning: Could not generate Plotly plots: {e}")

        # Optional post-processing (only if user requests a report)
        if args.report_output:
            print("\n[3/4] Generating logistic regression history...")
            generate_logreg_train_history(args.output)

            print("\n[4/4] Generating final report PDF...")
            process_classification_results(
                base_dir=args.output,
                output_pdf=args.report_output
            )
            print("Report generated.")

        log_step("classify_pipeline", vars(args), "success", start)
        print("\nClassification stage completed successfully!")

    except Exception as e:
        log_step("classify_pipeline", vars(args), f"error: {e}", start)
        print(f"\nClassification pipeline failed: {e}")
        raise


# === MAIN ENTRYPOINT ===
def main():
    parser = argparse.ArgumentParser(description="Unified bioinformatics pipeline.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- SIMULATE ---
    p_sim = subparsers.add_parser("simulate", help="Run simulation pipeline")
    p_sim.add_argument("--yaml", type=str, help="Path to YAML config file")
    p_sim.add_argument("--pre-input", required=False)
    p_sim.add_argument("--pre-output", required=False)
    p_sim.add_argument("--minseq", type=int, required=False)
    p_sim.add_argument("--maxsites", type=int, required=False)
    p_sim.add_argument("--minsites", type=int, required=False)
    p_sim.add_argument("--alphabet", choices=["aa","dna"], required=False)
    p_sim.add_argument("--align", "-a", required=False)
    p_sim.add_argument("--tree", "-t", required=False)
    p_sim.add_argument("--config", "-c", required=False)
    p_sim.add_argument("--sim-output", required=False)
    p_sim.add_argument("--ext_rate", "-e", required=False)
    p_sim.add_argument("--tree-output", required=False)
    p_sim.add_argument("--metrics-output", required=False, help="Output directory for metrics (optional)")
    p_sim.set_defaults(func=simulate_cmd)


    # --- METRICS ---
    p_met = subparsers.add_parser("metrics", help="Compute MPD between real and simulated sequences")
    p_met.add_argument("--empirical", help="Empirical FASTA directory")
    p_met.add_argument("--simulation", help="Simulated FASTA directory")
    p_met.add_argument("--output", default="results", help="Output directory")
    p_met.add_argument("--threads", type=int, default=4)
    p_met.set_defaults(func=run_metrics)


    # --- CLASSIFY ---
    p_cls = subparsers.add_parser("classify", help="Run classification pipeline")
    p_cls.add_argument("--yaml", type=str, help="Path to YAML config file")
    p_cls.add_argument("--real-align", required=False, default=None)
    p_cls.add_argument("--sim-align", required=False, default=None)
    p_cls.add_argument("--output", required=False, default=None)
    p_cls.add_argument("--config", required=False, default=None)
    p_cls.add_argument("--tools", required=False, default=None)
    p_cls.add_argument("--report-output", required=False, default=None, help="Optional PDF report output path")
    p_cls.add_argument("--two-iterations", action="store_true", default=False, help="Enable Run1 + Run2 refinement")
    p_cls.add_argument("--threshold", type=float, default=None, help="Threshold to classify sims as REAL")
    p_cls.add_argument("--2simconfig", dest="sim_config_2", type=str, required=False, default=None,
                       help="Path to simulation config JSON for run2. Required if not stored in run_1/store_1.")
    p_cls.add_argument("--2simtree", dest="sim_tree_2", type=str, required=False, default=None,
                       help="Path to tree directory for run2 simulation")
    p_cls.add_argument("--2simextrate", dest="sim_ext_rate_2", type=float, required=False, default=None,
                       help="External branch rate for run2 simulation")
    p_cls.add_argument("--2simalphabet", dest="sim_alphabet_2", choices=["aa", "dna"], required=False, default=None,
                       help="Alphabet for run2 simulation")
    p_cls.set_defaults(func=classify_cmd)
    
    # --- Visualisation Dashboard ---
    p_dash = subparsers.add_parser("visualisation", help="Launch the Dash dashboard")
    p_dash.set_defaults(func=lambda args: run_dashboard())

    args = parser.parse_args()

    # Handle YAML configuration
    if hasattr(args, 'yaml') and args.yaml:
        # Get the appropriate subparser for conflict checking
        subparser = None
        if args.command == "simulate":
            subparser = p_sim
        elif args.command == "classify":
            subparser = p_cls
        
        if subparser:
            check_yaml_conflict(args, subparser)
        
        config = load_yaml_config(args.yaml)
        apply_yaml_config(args, config, args.command)
    else:
        # Validate that required arguments are present when not using YAML
        if args.command == "simulate":
            required = ['pre_input', 'pre_output', 'minseq', 'maxsites', 'minsites', 
                       'alphabet', 'align', 'tree', 'config', 'sim_output', 'ext_rate', 
                       'tree_output']
            # metrics_output is optional
            missing = [arg for arg in required if getattr(args, arg, None) is None]
            if missing:
                print(f"ERROR: Missing required arguments: {', '.join(missing)}")
                print("Use --yaml config.yaml or provide all required flags.")
                sys.exit(1)
        elif args.command == "classify":
            required = ['real_align', 'sim_align', 'output', 'config', 'tools']
            missing = [arg for arg in required if getattr(args, arg, None) is None]
            if missing:
                print(f"ERROR: Missing required arguments: {', '.join(missing)}")
                print("Use --yaml config.yaml or provide all required flags.")
                sys.exit(1)
        
        # Set default threshold if not provided
        if args.command == "classify" and args.threshold is None:
            args.threshold = 0.5
    
    args.func(args)


if __name__ == "__main__":
    main()


# === COMMANDES EXEMPLES (TEST QUICK-START) ===
"""
# --- AVEC YAML ---
python3 scripts/main2.py simulate --yaml config/yaml/simulate.yaml
python3 scripts/main2.py classify --yaml config/yaml/classify.yaml
python3 scripts/main2.py classify --yaml config/yaml/classify_full.yaml


# --- SIMULATION ---
python3 scripts/main2.py simulate \
    --pre-input data/prot_mammals \
    --pre-output results/preprocessed \
    --minseq 5 \
    --maxsites 2000 \
    --minsites 100 \
    --alphabet aa \
    --align results/preprocessed/clean_data \
    --tree data/prot_mammals/trees \
    --config config/bpp/aa/WAG_frequencies.bpp \
    --sim-output results/simulations \
    --ext_rate 0.3 \
    --tree-output results/trees

    
# --- METRICS ---
python3 scripts/main2.py metrics \
    --empirical results/preprocessed/clean_data \
    --simulation results/simulations/ \
    --output results \
    --threads 8                            

    
# --- CLASSIFY : RUN 1 SEULEMENT ---
python3 scripts/main2.py classify \
    --real-align results/preprocessed/clean_data \
    --sim-align results/simulations \
    --output results/classification \
    --config config/config_template.json \
    --tools backup/


# --- CLASSIFY : RUN 1 + RUN 2 (refinement) ---
python3 scripts/main2.py classify \
    --real-align results/preprocessed/clean_data \
    --sim-align results/simulations \
    --output results/classification \
    --config config/config_template.json \
    --tools backup/ \
    --two-iterations


# --- CLASSIFY : RUN 1 + RUN 2 + PDF REPORT ---
python3 scripts/main2.py classify \
    --real-align results/preprocessed/clean_data \
    --sim-align results/simulations \
    --output results/classification \
    --config config/config_template.json \
    --tools backup/ \
    --two-iterations \
    --report-output results/classification/final_report.pdf

    
# --- VISUALISATION ---
python3 scripts/main2.py visualisation
"""