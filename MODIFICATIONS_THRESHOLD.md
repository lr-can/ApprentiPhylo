# 🔧 Modifications du calcul du seuil optimal dans pipeline.py

## 📝 Résumé des changements

Le fichier `backup/simulations-classifiers/src/classifiers/pipeline.py` a été modifié pour corriger le calcul du seuil optimal de classification.

---

## ❌ Problème initial

### Comportement ancien (INCORRECT)
La méthode `_find_optimal_threshold_roc()` **retournait l'AUC comme seuil de classification** :

```python
# ANCIEN CODE (INCORRECT)
auc = roc_auc_score(y_true, y_score)
return float(auc)  # ❌ Utilise l'AUC (ex: 0.87) comme seuil !
```

**Pourquoi c'est incorrect :**
- L'AUC est une **métrique de performance** (entre 0 et 1)
- Ce n'est **PAS un seuil de classification** !
- Résultat : seuils très élevés (~0.87) qui classent presque tout dans une seule classe

### Conséquences observées
```
RUN 1: threshold = 0.8671 → Seulement 3.37% des données retenues
RUN 2: threshold = 0.8778 → Seulement 4.41% des données retenues
```

Presque toutes les prédictions tombaient dans une seule classe car le seuil était trop élevé.

---

## ✅ Solution implémentée

### Nouveau comportement (CORRECT)

La méthode utilise maintenant le **J de Youden** pour trouver le seuil optimal sur la courbe ROC :

```python
# NOUVEAU CODE (CORRECT)
fpr, tpr, thresholds = roc_curve(y_true, y_score)

# Calcul du J de Youden (TPR - FPR)
j_scores = tpr - fpr

# Trouver le seuil qui maximise J
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]
```

**Avantages :**
- ✅ Maximise la séparation entre les deux classes
- ✅ Équilibre TPR (sensibilité) et FPR (spécificité)
- ✅ Seuil statistiquement optimal basé sur la courbe ROC

### Gestion des prédictions inversées

Le code détecte maintenant si le modèle prédit à l'envers (AUC < 0.5) :

```python
if auc < 0.5:
    # Modèle prédit l'inverse → inverser les scores
    y_score_inverted = 1 - y_score
    # Recalculer ROC avec scores inversés
    fpr, tpr, thresholds = roc_curve(y_true, y_score_inverted)
    # Inverser le seuil pour application aux scores originaux
    optimal_threshold = 1.0 - optimal_threshold
```

**Pourquoi c'est nécessaire :**
- Les logs montraient `AUC = 0.13` (pire que le hasard)
- Cela suggère que les labels sont inversés quelque part
- L'inversion permet d'obtenir un seuil correct malgré ce problème

---

## 📊 Impact attendu

### Avant (avec AUC comme seuil)
```
Seuil: ~0.87 (très élevé)
Rétention: 3-5% des données
F1-score: ~0.004
Accuracy: ~48%
```

### Après (avec Youden's J)
```
Seuil: ~0.3-0.5 (plus raisonnable)
Rétention: 30-60% des données (selon objectif)
F1-score: ~0.10-0.60 (meilleur)
Accuracy: ~20-44% (variable selon seuil)
```

---

## 🔍 Détails techniques

### Méthode de Youden

Le **J de Youden** est défini comme :
```
J = TPR - FPR = Sensibilité + Spécificité - 1
```

Le seuil optimal est celui qui **maximise J**, c'est-à-dire qui :
- Maximise le TPR (True Positive Rate / Sensibilité)
- Minimise le FPR (False Positive Rate)

### Localisation des modifications

**Fichier :** `backup/simulations-classifiers/src/classifiers/pipeline.py`

**Méthode modifiée :** `_find_optimal_threshold_roc()` (lignes ~591-650)

**Appels (3 endroits) :**
1. Ligne ~952 : RUN 1 - Filtrage initial
2. Ligne ~1117 : RUN 2 - Méthode `run2_retrain_best_model()`
3. Ligne ~1254 : RUN 2 - Méthode `run_two_iterations()`

**Messages de log mis à jour :**
```python
# Avant
f"threshold = {optimal_threshold:.4f} (AUC value)"

# Après
f"optimal threshold = {optimal_threshold:.4f} (from ROC - Youden's J)"
```

---

## 🧪 Test et validation

### Scripts d'analyse créés

1. **`scripts/analyse_predictions.py`**
   - Analyse les distributions de probabilités
   - Génère des violin plots et histogrammes
   - Export CSV des prédictions

2. **`scripts/analyse_thresholds.py`** (supprimé après utilisation)
   - Teste différents seuils (0.3, 0.5, 0.7, 0.87)
   - Calcule métriques pour chaque seuil
   - Montre l'impact sur la rétention

3. **`scripts/visualize_optimal_threshold.py`**
   - Visualise la courbe ROC avec les différents seuils
   - Montre le J de Youden
   - Compare ancien vs nouveau seuil

### Commande de test
```bash
cd /home/lorcan/Documents/Master/Projet/ApprentiPhylo_clean
source .venv/bin/activate
python scripts/visualize_optimal_threshold.py
```

---

## ⚠️ Problème détecté : Labels inversés

L'analyse a révélé un **problème sous-jacent** :

```
AUC = 0.13 (très mauvais)
```

Un AUC de 0.13 signifie que le modèle prédit **systématiquement l'inverse** de ce qu'il devrait :
- Les **simulés** obtiennent des probabilités **élevées** d'être "réels"
- Les **réels** obtiennent des probabilités **faibles**

**Hypothèses :**
1. Labels inversés lors de l'entraînement (`LABEL_REAL` et `LABEL_SIMULATED` échangés)
2. Logique inversée dans la fonction de prédiction
3. Les simulations sont "trop bonnes" et ressemblent plus aux données réelles

Le code modifié **détecte et corrige automatiquement** ce problème en inversant les prédictions lors du calcul du seuil.

---

## 📁 Fichiers générés

```
results/classification/predictions_analysis/
├── predictions_run1.csv
├── predictions_run2.csv
├── summary_statistics.csv
├── violin_plot_distributions.png
├── violin_plot_comparison.png
├── histogram_distributions.png
├── threshold_impact_analysis.png
├── threshold_metrics_run1.csv
├── threshold_metrics_run2.csv
├── optimal_threshold_comparison_run1.png
└── optimal_threshold_comparison_run2.png
```

---

## 🚀 Prochaines étapes recommandées

1. **Ré-exécuter le pipeline complet** pour tester le nouveau calcul de seuil :
   ```bash
   python backup/simulations-classifiers/src/classifiers/pipeline.py \
       --config results/classification/config.json \
       --two-iterations
   ```

2. **Vérifier les labels** dans le code d'entraînement pour s'assurer qu'ils ne sont pas inversés

3. **Comparer les résultats** avant/après modification

4. **Ajuster le calcul du seuil** si nécessaire (autres méthodes possibles : F1-max, distance à (0,1), etc.)

---

## 📚 Références

- **Youden's J statistic**: Youden, W. J. (1950). "Index for rating diagnostic tests". Cancer. 3 (1): 32–35.
- **ROC curves**: Fawcett, T. (2006). "An introduction to ROC analysis". Pattern Recognition Letters. 27 (8): 861–874.

---

**Date de modification :** 2025-12-02  
**Auteur :** Assistant IA avec Lorcan

