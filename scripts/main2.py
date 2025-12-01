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
from pathlib import Path

from preprocess import Preprocess
from simulation import BppSimulator
from compute_tree import ComputingTrees
from phylo_metrics import tree_summary
from classification import run_classification
from analyse_classif import process_classification_results
from fix_logreg_history import generate_logreg_train_history
from dashboard import run_dashboard



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


# === SIMULATION ===
def simulate_cmd(args):
    global_start = time.time()

    try:
        print("\n[1/4] Preprocessing input alignments...")
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

        print("\n[2/4] Running simulations...")
        config_file = args.config[0] if isinstance(args.config, list) else args.config
        bpp_sim = BppSimulator(
            align=str(clean_align_dir),
            tree=str(args.tree),
            config=str(config_file),
            output=str(args.sim_output),
            ext_rate=args.ext_rate
        )
        bpp_sim.simulate()

        print("\n[3/4] Computing trees...")
        tree_cp = ComputingTrees(args.sim_output, args.tree_output, args.alphabet)
        tree_cp.compute_all_trees()

        print("\n[4/4] Computing phylogenetic metrics (MPD)...")
        metrics_dir = Path(args.metrics_output)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        out_csv = metrics_dir / "phylo_metrics.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["tree", "MPD", "n_leaves"])
            writer.writeheader()
            for tree_file in Path(args.tree_output).glob("*.nw*"):
                m = tree_summary(tree_file)
                writer.writerow({"tree": tree_file.name, **m})
        print(f"Metrics saved to {out_csv}")

        log_step("simulate_pipeline", vars(args), "success", global_start)
        print("\nSimulation pipeline completed successfully!")

    except Exception as e:
        log_step("simulate_pipeline", vars(args), f"error: {e}", global_start)
        print(f"\nSimulation pipeline failed: {e}")
        raise


# === CLASSIFICATION ===
def classify_cmd(args):
    start = time.time()

    try:
        print("\n[1/3] Running classification pipeline...")
        
        run_classification(
            realali=args.real_align,
            simali=args.sim_align,
            output=args.output,
            config=args.config,
            tools=args.tools,
            two_iterations=args.two_iterations,
            threshold=args.threshold
        )

        print("\nClassification pipeline (iterations) completed.")

        # Optional post-processing (only if user requests a report)
        if args.report_output:
            print("\n[2/3] Generating logistic regression history...")
            generate_logreg_train_history(args.output)

            print("\n[3/3] Generating final report PDF...")
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
    p_sim.add_argument("--pre-input", required=True)
    p_sim.add_argument("--pre-output", required=True)
    p_sim.add_argument("--minseq", type=int, required=True)
    p_sim.add_argument("--maxsites", type=int, required=True)
    p_sim.add_argument("--minsites", type=int, required=True)
    p_sim.add_argument("--alphabet", choices=["aa","dna"], required=True)
    p_sim.add_argument("--align", "-a", required=True)
    p_sim.add_argument("--tree", "-t", required=True)
    p_sim.add_argument("--config", "-c", required=True)
    p_sim.add_argument("--sim-output", required=True)
    p_sim.add_argument("--ext_rate", "-e", required=True)
    p_sim.add_argument("--tree-output", required=True)
    p_sim.add_argument("--metrics-output", required=True)
    p_sim.set_defaults(func=simulate_cmd)

    # --- CLASSIFY ---
    p_cls = subparsers.add_parser("classify", help="Run classification pipeline")
    p_cls.add_argument("--real-align", required=True)
    p_cls.add_argument("--sim-align", required=True)
    p_cls.add_argument("--output", required=True)
    p_cls.add_argument("--config", required=True)
    p_cls.add_argument("--tools", required=True)
    p_cls.add_argument("--report-output", required=False, help="Optional PDF report output path")
    p_cls.add_argument("--two-iterations", action="store_true", help="Enable Run1 + Run2 refinement")
    p_cls.add_argument("--threshold", type=float, default=0.5, help="Threshold to classify sims as REAL")
    p_cls.set_defaults(func=classify_cmd)

    
    # --- Visualisation Dashboard ---
    p_dash = subparsers.add_parser("visualisation", help="Launch the Dash dashboard")
    p_dash.set_defaults(func=lambda args: run_dashboard())

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


# === COMMANDES EXEMPLES (TEST QUICK-START) ===
"""
# --- SIMULATION ---
python3 scripts/main2.py simulate \
    --pre-input data/prot_mammals \
    --pre-output results/preprocessed \
    --minseq 5 --maxsites 2000 --minsites 100 \
    --alphabet aa \
    --align results/preprocessed/clean_data \
    --tree data/prot_mammals/trees \
    --config backup/config/bpp/aa/WAG_frequencies.bpp \
    --sim-output results/simulations \
    --ext_rate 0.3 \
    --tree-output results/trees \
    --metrics-output results/metrics


# --- CLASSIFY : RUN 1 SEULEMENT ---
python3 scripts/main2.py classify \
    --real-align results/preprocessed/clean_data \
    --sim-align results/simulations \
    --output results/classification \
    --config backup/config_template.json \
    --tools backup/


# --- CLASSIFY : RUN 1 + RUN 2 (refinement) ---
python3 scripts/main2.py classify \
    --real-align results/preprocessed/clean_data \
    --sim-align results/simulations \
    --output results/classification \
    --config backup/config_template.json \
    --tools backup/ \
    --two-iterations


# --- CLASSIFY : RUN 1 + RUN 2 + PDF REPORT ---
python3 scripts/main2.py classify \
    --real-align results/preprocessed/clean_data \
    --sim-align results/simulations \
    --output results/classification \
    --config backup/config_template.json \
    --tools backup/ \
    --two-iterations \
    --report-output results/classification/final_report.pdf
"""
