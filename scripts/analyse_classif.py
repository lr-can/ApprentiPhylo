import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from fpdf import FPDF


def load_and_concat_parquets(base_dir, classifiers, filename):
    """
    Load and concatenate the files . parquet of a given type for all classifiers.
    """

    dfs = []
    for clf in classifiers:
        path = base_dir / clf / filename
        if path.exists():
            df = pd.read_parquet(path)
            df["classifier"] = clf
            dfs.append(df)
        else:
            print(f"Fichier manquant : {path}")
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
    pdf.multi_cell(0, 5, safe_text(train_df.head().to_string(index=False)))
    pdf.ln(8)

    # Section : Best predictions
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, safe_text("Aperçu des meilleures prédictions :"), ln=True)
    pdf.set_font("Helvetica", size=10)
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



def process_classification_results(base_dir="results/classification", output_pdf="results/classification/report.pdf"):
    """
    Main function: load, concatenate, trace and export everything in a PDF.
    Easily callable from main.py.
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
    pred_df.to_parquet(pred_concat_path)
    print(f"Tables sauvegardées dans :\n - {train_concat_path}\n - {pred_concat_path}")

    # Génération des plots
    print("\nGénération des courbes d'apprentissage...")
    plots = plot_learning_curves(train_df, output_dir)

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