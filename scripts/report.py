"""
report.py
==========
Génère un rapport PDF complet pour une simulation.  
Inclut :
 - tableau des statistiques d’alignements,
 - histogrammes de longueurs et de gaps,
 - représentation graphique des arbres phylogénétiques annotés avec MPD.  
Utilise matplotlib et BioPython pour la visualisation.
"""
import matplotlib
matplotlib.use("Agg")  # Backend non graphique pour les serveurs
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pandas as pd
from Bio import SeqIO, Phylo
from pathlib import Path
from datetime import datetime
from scripts.phylo_metrics import tree_summary


def generate_pdf_report(simulation_folder, output_dir):
    """
    Génère un PDF avec statistiques, histogrammes et arbres pour une simulation.
    Args:
        simulation_folder (str): Dossier contenant les résultats de la simulation.
        output_dir (str): Dossier où enregistrer le rapport PDF.
    Returns:
        str: Chemin vers le fichier PDF généré.
    """
    simulation_folder = Path(simulation_folder)
    output_dir = Path(output_dir)
    if output_dir.exists() and output_dir.is_file():
        output_dir.unlink()  # supprime le fichier s’il bloque la création du dossier
    output_dir.mkdir(parents=True, exist_ok=True)


    # nom final : report_simulation_YYYYMMDD_HHMM.pdf
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    pdf_path = output_dir / f"report_simulation_{timestamp}.pdf"

    pdf = PdfPages(pdf_path)

    fasta_files = list(simulation_folder.glob("*.fasta"))
    tree_files = list(simulation_folder.glob("*.nw")) + list(simulation_folder.glob("*.nwk"))

    # --- Statistiques sur les alignements ---
    stats_list = []
    for f in fasta_files:
        sequences = list(SeqIO.parse(f, "fasta"))
        nseq = len(sequences)
        lengths = [len(seq.seq) for seq in sequences]
        mean_length = sum(lengths)/nseq if nseq > 0 else 0
        gaps = sum(str(seq.seq).count('-') for seq in sequences)

        stats_list.append({
            "file": f.name,
            "nseq": nseq,
            "mean_length": mean_length,
            "n_gaps": gaps
        })

    stats_df = pd.DataFrame(stats_list)

    # --- Page tableau ---
    fig, ax = plt.subplots(figsize=(10, len(stats_list)*0.5 + 1))
    ax.axis('off')

    if stats_df.empty:
        print("⚠️ Aucun résultat disponible pour le rapport PDF")
        ax.text(0.5, 0.5, "Aucune donnée disponible", ha='center', va='center')
    else:
        tbl = ax.table(cellText=stats_df.values, colLabels=stats_df.columns, loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.2)

    pdf.savefig()
    plt.close()

    # --- Histogrammes ---
    if not stats_df.empty:
        for col in ["mean_length", "n_gaps"]:
            plt.figure()
            plt.hist(stats_df[col], bins=20, color='skyblue', edgecolor='black')
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            pdf.savefig()
            plt.close()

    # --- Arbres et métriques phylogénétiques ---
    for tfile in tree_files:
        plt.figure(figsize=(8,6))
        try:
            tree = Phylo.read(tfile, "newick")
            Phylo.draw(tree, do_show=False)
            metrics = tree_summary(tfile)
            plt.title(f"{tfile.name}\nMPD={metrics['MPD']:.3f}, n_leaves={metrics['n_leaves']}")
        except Exception as e:
            plt.title(f"{tfile.name} (error reading tree: {e})")

        pdf.savefig()
        plt.close()

    pdf.close()
    print(f"PDF report generated: {pdf_path}")

    return pdf_path
