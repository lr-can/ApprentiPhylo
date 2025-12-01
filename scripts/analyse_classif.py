import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from fpdf import FPDF
import shutil
import numpy as np


def load_and_concat_parquets(base_dir, classifiers, filename):
    """
    Load and concatenate the files . parquet of a given type for all classifiers.
    Cherche dans run_1/ et run_2/ et aussi directement dans les dossiers des classifieurs.
    Charge les fichiers des deux runs s'ils existent.
    """

    dfs = []
    for clf in classifiers:
        found_in_runs = False
        # Chercher dans run_1 et run_2 - charger les deux s'ils existent
        for run_dir in ["run_1", "run_2"]:
            path = base_dir / run_dir / clf / filename
            if path.exists():
                df = pd.read_parquet(path)
                df["classifier"] = clf
                if "run" not in df.columns:
                    df["run"] = run_dir
                elif "run" in df.columns and df["run"].isna().all():
                    df["run"] = run_dir
                dfs.append(df)
                found_in_runs = True
                # Copier aussi vers le dossier du classifieur pour compatibilité (seulement depuis run_1 pour éviter d'écraser)
                if run_dir == "run_1":
                    clf_dir = base_dir / clf
                    clf_dir.mkdir(exist_ok=True)
                    shutil.copy(path, clf_dir / filename)
        
        # Si pas trouvé dans run_1/run_2, chercher directement dans le dossier du classifieur
        if not found_in_runs:
            path = base_dir / clf / filename
            if path.exists():
                df = pd.read_parquet(path)
                df["classifier"] = clf
                if "run" not in df.columns:
                    df["run"] = "unknown"
                dfs.append(df)
            else:
                print(f"Fichier manquant : {base_dir}/(run_1|run_2)/{clf}/{filename}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()  # dataframe vide


def plot_learning_curves(df, output_dir):
    """
    Generates the learning curves for each classifier.
    Tries to guess the correct name of the loss columns.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = []

    # Vérifier si le DataFrame est vide ou n'a pas la colonne "classifier"
    if df.empty or "classifier" not in df.columns:
        print("Aucune donnée d'entraînement disponible pour générer les courbes d'apprentissage.")
        return plots

    for clf, subdf in df.groupby("classifier"):
        plt.figure()

        if "epoch" not in subdf.columns:
            print(f"Pas de colonne 'epoch' dans {clf}, courbe ignorée.")
            continue

        grouped = subdf.groupby("epoch").mean(numeric_only=True)

        # Détection automatique du nom des colonnes de perte
        possible_train_loss = [c for c in grouped.columns if "loss" in c.lower() and "val" not in c.lower()]
        possible_val_loss = [c for c in grouped.columns if "val" in c.lower() and "loss" in c.lower()]

        if not possible_train_loss:
            print(f"Pas de colonne de loss trouvée pour {clf}, colonnes = {list(grouped.columns)}")
            plt.close()
            continue

        train_loss_col = possible_train_loss[0]
        plt.plot(grouped.index, grouped[train_loss_col], label=f"{train_loss_col}")

        if possible_val_loss:
            val_loss_col = possible_val_loss[0]
            plt.plot(grouped.index, grouped[val_loss_col], label=f"{val_loss_col}")

        plt.title(f"Courbe d'apprentissage - {clf}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plot_path = output_dir / f"{clf}_learning_curve.png"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        plots.append(plot_path)

    return plots


def plot_roc_curves(base_dir, output_dir):
    """
    Génère les courbes ROC à partir des fichiers CSV exportés par le pipeline,
    ou calcule les courbes ROC à partir des fichiers best_preds.parquet si les CSV n'existent pas.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = []
    
    base_dir = Path(base_dir)
    classifiers = [
        "AACnnClassifier",
        "DenseMsaClassifier",
        "DenseSiteClassifier",
        "LogisticRegressionClassifier"
    ]
    
    # D'abord, chercher les fichiers ROC CSV existants
    roc_csv_found = set()
    for run_dir in ["run_1", "run_2"]:
        roc_dir = base_dir / run_dir / "roc_data"
        if not roc_dir.exists():
            continue
            
        for roc_file in roc_dir.glob("*_roc.csv"):
            try:
                df = pd.read_csv(roc_file)
                
                if df.empty or "fpr" not in df.columns or "tpr" not in df.columns:
                    continue
                
                classifier = roc_file.stem.replace("_roc", "")
                roc_csv_found.add((classifier, run_dir))
                
                plt.figure(figsize=(8, 6))
                plt.plot(df["fpr"], df["tpr"], linewidth=2, label=f"{classifier} ({run_dir})")
                plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"Courbe ROC - {classifier} ({run_dir})")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Calculer l'AUC si possible
                if "tpr" in df.columns and "fpr" in df.columns:
                    try:
                        from sklearn.metrics import auc
                        auc_score = auc(df["fpr"], df["tpr"])
                        plt.text(0.6, 0.2, f"AUC = {auc_score:.3f}", 
                                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
                    except:
                        pass
                
                plot_path = output_dir / f"{classifier}_{run_dir}_roc.png"
                plt.savefig(plot_path, bbox_inches="tight", dpi=150)
                plt.close()
                plots.append(plot_path)
            except Exception as e:
                print(f"Erreur lors de la génération de la courbe ROC pour {roc_file}: {e}")
                continue
    
    # Ensuite, calculer les courbes ROC à partir de best_preds.parquet pour les classifieurs manquants
    for classifier in classifiers:
        for run_dir in ["run_1", "run_2"]:
            # Skip si on a déjà un CSV ROC pour ce classifieur
            if (classifier, run_dir) in roc_csv_found:
                continue
            
            preds_path = base_dir / run_dir / classifier / "best_preds.parquet"
            if not preds_path.exists():
                continue
            
            try:
                preds_df = pd.read_parquet(preds_path)
                
                if preds_df.empty:
                    continue
                
                # Identifier les colonnes de probabilité et de label
                prob_col = None
                if "prob_real" in preds_df.columns:
                    prob_col = "prob_real"
                elif "prob" in preds_df.columns:
                    prob_col = "prob"
                else:
                    print(f"Pas de colonne de probabilité trouvée pour {classifier} ({run_dir})")
                    continue
                
                label_col = None
                if "target" in preds_df.columns:
                    label_col = "target"
                elif "true_label" in preds_df.columns:
                    label_col = "true_label"
                else:
                    print(f"Pas de colonne de label trouvée pour {classifier} ({run_dir})")
                    continue
                
                # Extraire les données
                y_score = preds_df[prob_col].values
                y_true = preds_df[label_col].values
                
                # Vérifier qu'on a les deux classes
                unique_labels = np.unique(y_true)
                if len(unique_labels) < 2:
                    print(f"Seulement {len(unique_labels)} classe(s) trouvée(s) pour {classifier} ({run_dir}), impossible de calculer ROC")
                    continue
                
                # Calculer la courbe ROC
                from sklearn.metrics import roc_curve, auc
                # pos_label=1 pour LABEL_SIMULATED (selon le code du pipeline)
                fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
                
                # Ajouter le point (0,0) pour complétude
                fpr = np.concatenate([[0.0], fpr])
                tpr = np.concatenate([[0.0], tpr])
                
                # Calculer l'AUC
                auc_score = auc(fpr, tpr)
                
                # Générer le plot
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, linewidth=2, label=f"{classifier} ({run_dir})")
                plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"Courbe ROC - {classifier} ({run_dir})")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.text(0.6, 0.2, f"AUC = {auc_score:.3f}", 
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
                
                plot_path = output_dir / f"{classifier}_{run_dir}_roc.png"
                plt.savefig(plot_path, bbox_inches="tight", dpi=150)
                plt.close()
                plots.append(plot_path)
                
                print(f"Courbe ROC générée pour {classifier} ({run_dir}) à partir de best_preds.parquet")
                
            except Exception as e:
                print(f"Erreur lors de la génération de la courbe ROC pour {classifier} ({run_dir}): {e}")
                continue
    
    if not plots:
        print("Aucune courbe ROC disponible.")
    else:
        print(f"{len(plots)} courbe(s) ROC générée(s).")
    
    return plots


def generate_pdf_report(train_df, pred_df, plots, output_pdf):
    """
    Generate a PDF report combining:
      - the first lines of the concatenated tables
      - the learning curves
    """

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def safe_text(text: str) -> str:
        """Convertit le texte en latin-1 compatible."""
        return text.encode("latin-1", "replace").decode("latin-1")

    # Titre
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, safe_text("Rapport de classification"), ln=True, align="C")
    pdf.ln(10)

    # Section : Train history
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, safe_text("Aperçu des données d'entraînement :"), ln=True)
    pdf.set_font("Helvetica", size=10)
    if train_df.empty:
        pdf.multi_cell(0, 5, safe_text("Aucune donnée d'entraînement disponible."))
    else:
        pdf.multi_cell(0, 5, safe_text(train_df.head().to_string(index=False)))
    pdf.ln(8)

    # Section : Best predictions
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, safe_text("Aperçu des meilleures prédictions :"), ln=True)
    pdf.set_font("Helvetica", size=10)
    if pred_df.empty:
        pdf.multi_cell(0, 5, safe_text("Aucune prédiction disponible."))
    else:
        pdf.multi_cell(0, 5, safe_text(pred_df.head().to_string(index=False)))
    pdf.ln(8)

    # Section : Courbes d’apprentissage
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, safe_text("Courbes d’apprentissage :"), ln=True)

    for plot in plots:
        pdf.add_page()
        pdf.image(str(plot), x=15, y=25, w=180)
        pdf.set_y(270)
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 5, safe_text(f"{plot.name}"), align="C")

    pdf.output(str(output_pdf))
    print(f"\nRapport PDF généré : {output_pdf}")



def process_classification_results(base_dir="results/classification", output_pdf="results/classification/report.pdf", iteration=1):
    """
    Main function: load, concatenate, trace and export everything in a PDF.
    Easily callable from main.py.
    Args:
        base_dir (str): Base directory where classification results are stored.
        output_pdf (str): Path to the output PDF report.
        iteration (int): Iteration number (for future extensions).
    Returns:
        None
    """

    base_dir = Path(base_dir)
    output_dir = base_dir / "plots"
    output_pdf = Path(output_pdf)

    classifiers = [
        "AACnnClassifier",
        "DenseMsaClassifier",
        "DenseSiteClassifier",
        "LogisticRegressionClassifier"
    ]

    print("Lecture des fichiers parquet...")
    train_df = load_and_concat_parquets(base_dir, classifiers, "train_history.parquet")
    pred_df = load_and_concat_parquets(base_dir, classifiers, "best_preds.parquet")

    # Sauvegarde des tables concaténées
    train_concat_path = base_dir / "all_train_history.parquet"
    pred_concat_path = base_dir / "all_best_preds.parquet"

    # Sauvegarder seulement si les DataFrames ne sont pas vides
    if not train_df.empty:
        #  Forcer un typage homogène avant écriture Parquet
        if "best" in train_df.columns:
        # 🔍 Convertir tout en texte, puis mapper vers bool
            train_df["best"] = (
                train_df["best"]
                .astype(str)  # tout en texte pour éviter les erreurs
                .str.strip()
                .str.lower()
                .map({"true": True, "false": False, "1": True, "0": False, "nan": pd.NA})
                .astype("boolean")
            )
        train_df.to_parquet(train_concat_path)
        print(f"Table sauvegardée : {train_concat_path}")
    else:
        print("Aucune donnée d'entraînement à sauvegarder.")

    if not pred_df.empty:
        pred_df.to_parquet(pred_concat_path)
        print(f"Table sauvegardée : {pred_concat_path}")
    else:
        print("Aucune prédiction à sauvegarder.")

    # Génération des plots
    print("\nGénération des courbes d'apprentissage...")
    plots = plot_learning_curves(train_df, output_dir)
    
    print("\nGénération des courbes ROC...")
    roc_plots = plot_roc_curves(base_dir, output_dir)
    plots.extend(roc_plots)

    print(f"[{3*iteration}/6] PDF rendering...")
    # Génération du PDF
    generate_pdf_report(train_df, pred_df, plots, output_pdf)



# INTERPRETATION
"""
#### Table 1: all_train_history.parquet
Comes from the concatenation of the train_history.parquet files of each classifier.
Each line corresponds to an epoch for a given model.
columns:
epoch = training iteration number
train_loss = loss (error) on training data
val_loss = loss on validation data (more reliable to compare)
val_acc = precision on validation
f1 = F1 score (harmonic mean between precision and recall)
lr = learning rate
best = boolean indicating if it is the best registered model

train_loss and val_loss decrease when it stabilizes -> model learns.
val_loss increases while train_loss decreases -> overfitting.
val_acc or f1 -> final performance indicators.
The higher val_acc and f1 are, the more generalized the model.


##### Table 2: all_best_preds.parquet:
Contains the best predictions made by each classifier on the test data.
Each line corresponds to an alignment on which the model makes a prediction (simulation vs real, or internal classes according to your setup).
columns:
file or id = identifier of the sample
true_label = real class
pred_label = predicted class (model prediction)
prob = probability of prediction (model confidence)
correct = boolean (1 if well ranked)
classifier = model name

The proportion of correct = 1 rate of good prediction.
If the model is always wrong about the same classes -> it lacks generalization.
You can group by classifier to calculate an average rate of precision per model.


#### Learning curves:
learning models drawn from training data.
X-Axis = epochs (training iterations)
Y-axis = loss

The two curves must descend and then stabilize together -> good learning.
If the validation curve (val_loss) remains high -> the model generalizes poorly.
If the losses are very unstable -> non-convergent learning or insufficient data.


#### Which one is the best classifier?
F1-mean score (f1) = Good compromise between accuracy and recall
Validation accuracy (val_acc) = Higher => more reliable
Stability (val_loss) = robust model
Convergence time (number of epochs before stabilization) = Faster => more efficient


#### example:
The DenseMsaClassifier model reaches an average F1 of 0.91 with a stable val_loss,
while AACnnClassifier is capping at 0.84 and showing overfitting after 15 epochs.
Logistic regression, although simpler, obtains 0.78 but converges very quickly.
Overall, DenseMsaClassifier is the most efficient and generalizes best.
"""