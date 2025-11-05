"""
main.py
========
Point d’entrée principal du pipeline.  
Gère toutes les sous-commandes (prétraitement, simulation, classification, arbres, etc.) via argparse.  
Orchestre le flux complet d’analyse, enregistre les logs dans un CSV et déclenche la génération automatique de rapports PDF.

Ce script centralise toutes les interfaces en ligne de commande des sous-modules :
- preprocess : nettoie et filtre les alignements de séquence
- simuler : exécute des simulations de séquences empiriques ou basées sur des modèles
- tree : calcule les arbres phylogénétiques
- métriques : calcule les métriques phylogénétiques (par exemple, MPD)
- rapport : génère un rapport de synthèse PDF

Chaque étape est enregistrée au format CSV (logs/pipeline_log.csv)
et peut être géré de manière indépendante.
"""

import argparse
import time
import csv
from pathlib import Path
import sys

from classification import run_classification
from compute_tree import ComputingTrees
from data_description import describe_data
from filter_mono import copy_mono_files
from preprocess import Preprocess
from simulation import ESMsimulator, BppSimulator
from report import generate_pdf_report
from phylo_metrics import tree_summary

# === LOGGING ===
def log_step(step_name, args_dict, status, start_time):
    """
    Log each pipeline step into CSV file.
    Args:
        step_name (str): Name of the pipeline step.
        args_dict (dict): Arguments used for the step.
        status (str): Status of the step (e.g., "success", "error").
        start_time (float): Start time of the step.
    Returns:
        None
    """
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


# === PIPELINE STEPS ===
def preprocess_cmd(args):
    """
    Launch preprocessing step.
    Args:
        args: Parsed command-line arguments.
    Returns:
        None
    """
    start = time.time()
    try:
        pr = Preprocess(
            input_dir=args.input,
            output_dir=args.output,
            minseq=args.minseq,
            maxsites=args.maxsites,
            minsites=args.minsites,
            alphabet=args.alphabet,
        )
        pr.preprocessing()
        pr.remove_gaps()
        pr.remove_ambig_sites("gapless")
        pr.remove_ambig_sites("clean")
        log_step("preprocess", vars(args), "success", start)
    except Exception as e:
        log_step("preprocess", vars(args), f"error: {e}", start)
        raise


def simulate_cmd(args):
    """
    Launch full pipeline automatically starting from simulation.
    Args:
        args: Parsed command-line arguments.
    Returns:
        None
    """
    global_start = time.time()

    try:
        # === PREPROCESS ===
        print("\n[1/6] Preprocessing input alignments...") 
        pre_args = argparse.Namespace( 
            input=args.pre_input, 
            output=args.pre_output, 
            minseq=args.minseq, 
            maxsites=args.maxsites, 
            minsites=args.minsites, 
            alphabet=args.alphabet 
            )
        preprocess_cmd(pre_args)
        clean_align_dir = Path(args.align)  # récupère le dossier final nettoyé

        # === SIMULATION ===
        print("\n[2/6] Running simulations...")
        config_file = args.config[0] if isinstance(args.config, list) else args.config

        bpp_sim = BppSimulator(
            align=str(clean_align_dir),
            tree=str(args.tree),
            config=str(config_file),
            output=str(args.sim_output),
            ext_rate=args.ext_rate
        )
        bpp_sim.simulate()

        # === TREE COMPUTATION ===
        print("\n[3/6] Computing trees...")
        tree_cp = ComputingTrees(args.sim_output, args.tree_output, args.alphabet)
        tree_cp.compute_all_trees()

        # === METRICS (MPD) ===
        print("\n[4/6] Computing phylogenetic metrics (MPD)...")
        metrics_dir = Path(args.metrics_output)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        out_csv = metrics_dir / "phylo_metrics.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["tree", "MPD", "n_leaves"])
            writer.writeheader()
            for tree_file in Path(args.tree_output).glob("*.nw*"):
                m = tree_summary(tree_file)
                writer.writerow({"tree": tree_file.name, **m})

        # === CLASSIFICATION ===
        print("\n[5/6] Running classification between real and simulated alignments...")
        try:
            run_classification(
                realali=pre_args.output,
                simali=args.sim_output,
                output=args.class_output,
                config=args.class_config,
                tools=args.tools
            )
            print("✅ Classification done.")
        except Exception as e:
            print(f"❌ Classification failed: {e}")

        # === REPORT ===
        print("\n[6/6] Generating final report PDF...")
        generate_pdf_report(args.sim_output, args.report_output)

        log_step("simulate_pipeline", vars(args), "success", global_start)
        print("\n✅ Pipeline completed successfully!")

    except Exception as e:
        log_step("simulate_pipeline", vars(args), f"error: {e}", global_start)
        print(f"\n❌ Pipeline failed: {e}")
        raise



def tree_cmd(args):
    start = time.time()
    from compute_tree import Computing_trees
    try:
        tree_cp = Computing_trees(args.input, args.output, args.alphabet, args.only)
        tree_cp.compute_all_trees()
        log_step("compute_tree", vars(args), "success", start)
    except Exception as e:
        log_step("compute_tree", vars(args), f"error: {e}", start)
        raise


def metrics_cmd(args):
    start = time.time()
    try:
        from phylo_metrics import tree_summary
        input_dir = Path(args.input)
        out_csv = Path(args.output) / "phylo_metrics.csv"
        out_csv.parent.mkdir(exist_ok=True)

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["tree", "MPD", "n_leaves"])
            writer.writeheader()
            for tree_file in input_dir.glob("*.nw*"):
                m = tree_summary(tree_file)
                writer.writerow({"tree": tree_file.name, **m})
                print(f"{tree_file.name}: MPD={m['MPD']:.4f}, n_leaves={m['n_leaves']}")

        log_step("metrics", vars(args), "success", start)
    except Exception as e:
        log_step("metrics", vars(args), f"error: {e}", start)
        raise


def report_cmd(args):
    start = time.time()
    try:
        from report import generate_report
        generate_report(args.input, args.output)
        log_step("report", vars(args), "success", start)
    except Exception as e:
        log_step("report", vars(args), f"error: {e}", start)
        raise


# === MAIN ENTRYPOINT ===
def main():
    parser = argparse.ArgumentParser(
        description="Unified bioinformatics pipeline. Launch full workflow with 'simulate'."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- SIMULATE (FULL PIPELINE) ---
    p_sim = subparsers.add_parser("simulate", help="Run full pipeline (preprocess + simulate + tree + metrics + report).")

    # PREPROCESS ARGS
    p_sim.add_argument("--pre-input", required=True, help="Raw alignments directory.")
    p_sim.add_argument("--pre-output", required=True, help="Output directory for cleaned alignments.")
    p_sim.add_argument("--minseq", type=int, required=True, help="Minimum number of sequences.")
    p_sim.add_argument("--maxsites", type=int, required=True, help="Maximum number of sites.")
    p_sim.add_argument("--minsites", type=int, required=True, help="Minimum number of sites.")
    p_sim.add_argument("--alphabet", choices=["aa", "dna"], required=True, help="Alphabet type.")

    # SIMULATION ARGS
    p_sim.add_argument("--align", "-a", required=True, help="Directory containing alignments (used in simulation).")
    p_sim.add_argument("--tree", "-t", required=True, help="Directory containing phylogenetic trees.")
    p_sim.add_argument("--config", "-c", type=str, required=True, help="BPP configuration files.")
    p_sim.add_argument("--sim-output", required=True, help="Output directory for simulated data.")
    p_sim.add_argument("--ext_rate", "-e", help="External branch rate (for BPP).")

    # CLASSIFICATION ARGS
    p_sim.add_argument("--class-config", required=True, help="Classification config JSON template.")
    p_sim.add_argument("--class-output", required=True, help="Output directory for classification results.")
    p_sim.add_argument("--tools", required=True, help="Tools root directory (contient simulations-classifiers).")

    # TREE & METRICS OUTPUTS
    p_sim.add_argument("--tree-output", required=True, help="Output directory for generated trees.")
    p_sim.add_argument("--metrics-output", required=True, help="Output directory for metrics (MPD).")
    p_sim.add_argument("--report-output", required=True, help="Output directory for PDF report.")
    p_sim.set_defaults(func=simulate_cmd)

    # --- TREE (optional direct call) ---
    p_tree = subparsers.add_parser("tree", help="Compute trees independently.")
    p_tree.add_argument("--input", "-i", required=True)
    p_tree.add_argument("--output", "-o", required=True)
    p_tree.add_argument("--alphabet", "-a", choices=["nt", "aa"], required=True)
    p_tree.add_argument("--only", help="List of files to process.")
    p_tree.set_defaults(func=tree_cmd)

    # --- METRICS (optional direct call) ---
    p_mpd = subparsers.add_parser("metrics", help="Compute MPD metrics on trees.")
    p_mpd.add_argument("--input", "-i", required=True)
    p_mpd.add_argument("--output", "-o", required=True)
    p_mpd.set_defaults(func=metrics_cmd)

    # --- REPORT (optional direct call) ---
    p_report = subparsers.add_parser("report", help="Generate PDF report manually.")
    p_report.add_argument("--input", "-i", required=True)
    p_report.add_argument("--output", "-o", required=True)
    p_report.set_defaults(func=report_cmd)

    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

# === COMMANDE QUI MARCHE ===
"""
python3 scripts/main.py simulate \
 --pre-input data/prot_mammals \
 --pre-output results/preprocessed/clean_data \
 --minseq 5 --maxsites 2000 --minsites 100 \
 --alphabet aa \
 --align results/preprocessed \
 --tree data/prot_mammals/trees \
 --config backup/config/bpp/aa/WAG_frequencies.bpp \
 --sim-output results/simulations \
 --ext_rate 0.3 \
 --tree-output results/trees \
 --metrics-output results/metrics \
 --class-config backup/config_template.json \
 --class-output results/classification \
 --tools backup/ \
 --report-output results/report
 
 """