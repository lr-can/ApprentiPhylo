# 🔧 Correction de l'inversion des labels de prédiction

## 🐛 Problème détecté

### Symptômes initiaux
- **AUC = 0.13** (bien pire que le hasard de 0.5)
- Les alignements **simulés** obtenaient des probabilités **élevées** (> 0.5)
- Les alignements **réels** obtenaient des probabilités **faibles** (< 0.5)
- Les prédictions semblaient inversées

### Cause racine identifiée

Incohérence entre la définition des labels et le code de prédiction :

**Dans `utils.py` (AVANT) :**
```python
LABEL_REAL = 0
LABEL_SIMULATED = 1
```

**Dans `deep_classifier.py` et `pipeline.py` :**
```python
# Pour les modèles avec 2 sorties (softmax)
probs = torch.softmax(logits, dim=1)
prob_real = probs[:, 1]  # ← Prend l'index 1 !
```

**Le problème :** Le code prenait `probs[:, 1]` (probabilité de la classe 1) pour `prob_real`, mais la classe 1 était `LABEL_SIMULATED`, pas `LABEL_REAL` !

Donc **`prob_real` contenait en fait P(SIMULATED)**, pas P(REAL) ! 😱

---

## ✅ Solution implémentée

### 1. Correction dans `utils.py`

Inversion des définitions des labels pour correspondre au code existant :

```python
# AVANT (INCORRECT)
LABEL_REAL = 0
LABEL_SIMULATED = 1

# APRÈS (CORRECT)
# Label definitions (aligned with model output indices)
# Model outputs: [prob_class_0, prob_class_1]
# prob_real = probs[:, 1] → so LABEL_REAL must be 1
LABEL_SIMULATED = 0
LABEL_REAL = 1
```

**Justification :** Puisque le code prend `probs[:, 1]` pour `prob_real`, alors `LABEL_REAL` doit être 1 pour que le nom corresponde à la réalité.

### 2. Correction dans `pipeline.py`

Mise à jour du calcul de la courbe ROC pour utiliser le bon label positif :

```python
# AVANT
fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=LABEL_SIMULATED)

# APRÈS
# pos_label=LABEL_REAL because prob_real represents P(REAL)
fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=LABEL_REAL)
```

### 3. Correction des scripts d'analyse

Pour les **prédictions déjà existantes** (faites avec l'ancienne convention), ajout d'une inversion dans les scripts d'analyse :

**Dans `analyse_predictions.py` et `visualize_optimal_threshold.py` :**
```python
# ⚠️ Inverser les prédictions existantes (ancienne convention)
df = df.with_columns([
    (1.0 - pl.col("prob_real")).alias("prob_real")
])
```

---

## 📊 Impact des corrections

### Avant correction (données brutes)
```
RUN 1: AUC = 0.13 ❌
       Simulés: prob_mean = 0.58 (> 0.5) ❌
       Réels:   prob_mean = 0.40 (< 0.5) ❌
```

### Après correction
```
RUN 1: AUC = 0.87 ✅
       Simulés: prob_mean = 0.41 (< 0.5) ✅
       Réels:   prob_mean = 0.60 (> 0.5) ✅

RUN 2: AUC = 0.87 ✅
       Simulés: prob_mean = 0.44 (< 0.5) ✅
       Réels:   prob_mean = 0.70 (> 0.5) ✅
```

### Seuils optimaux calculés (Youden's J)
```
RUN 1: threshold = 0.4907 (au lieu de 0.8671)
       TPR = 90.13%, FPR = 26.71%, J = 0.6341

RUN 2: threshold = 0.5691 (au lieu de 0.8778)
       TPR = 82.81%, FPR = 22.69%, J = 0.6012
```

### Taux de rétention (avec seuil 0.5)
```
RUN 1: 25% des simulés flaggés comme REAL (vs 75% avant)
RUN 2: 39% des simulés flaggés comme REAL (vs 63% avant)
Rétention globale: 27.87% (vs 45% avant)
```

---

## 📁 Fichiers modifiés

### Modifications permanentes (pour futurs entraînements)
1. **`backup/simulations-classifiers/src/classifiers/utils.py`**
   - Ligne 12-13 : Inversion des labels

2. **`backup/simulations-classifiers/src/classifiers/pipeline.py`**
   - Ligne 574 : Correction du `pos_label` pour ROC
   - Lignes 591-651 : Amélioration du calcul du seuil optimal (Youden's J)

### Scripts d'analyse (pour données existantes)
3. **`scripts/analyse_predictions.py`**
   - Ajout de l'inversion des prédictions existantes

4. **`scripts/visualize_optimal_threshold.py`**
   - Ajout de l'inversion des prédictions existantes

---

## 🔄 Prochaines étapes

### Pour utiliser les corrections

1. **Pour les nouveaux entraînements** : Les corrections dans `utils.py` et `pipeline.py` seront appliquées automatiquement

2. **Pour réentraîner avec les bons labels** :
   ```bash
   cd /home/lorcan/Documents/Master/Projet/ApprentiPhylo_clean
   source .venv/bin/activate
   
   # Nettoyer les anciens résultats (optionnel)
   rm -rf results/classification/run_*
   
   # Relancer le pipeline
   python backup/simulations-classifiers/src/classifiers/pipeline.py \
       --config results/classification/config.json \
       --two-iterations
   ```

3. **Pour analyser les nouveaux résultats** :
   - Retirer l'inversion dans `analyse_predictions.py` (lignes ~53-56)
   - Retirer l'inversion dans `visualize_optimal_threshold.py` (lignes ~34-38)

---

## 🎯 Résumé des changements

| Aspect | Avant | Après |
|--------|-------|-------|
| **LABEL_REAL** | 0 ❌ | 1 ✅ |
| **LABEL_SIMULATED** | 1 ❌ | 0 ✅ |
| **prob_real signifie** | P(SIMULATED) ❌ | P(REAL) ✅ |
| **AUC** | 0.13 ❌ | 0.87 ✅ |
| **Seuil optimal** | 0.87 (AUC) ❌ | 0.49-0.57 (Youden) ✅ |
| **Réels > Simulés** | Non ❌ | Oui ✅ |

---

## 📚 Détails techniques

### Pourquoi cette incohérence existait ?

L'erreur provient probablement d'une convention initiale où :
- Les labels étaient définis comme `REAL=0, SIMULATED=1`
- Mais le code de prédiction a été écrit en supposant que `prob_real = probs[:, 1]`

Cette incohérence n'a pas été détectée initialement car :
1. Le modèle s'entraînait correctement (peu importe la convention tant qu'elle est cohérente)
2. Mais les **interprétations** des prédictions étaient inversées
3. L'ancien code utilisait l'AUC comme seuil, ce qui masquait le problème

### Vérification de la cohérence

Après correction, vérifiez que :
```python
# Dans le code de prédiction
prob_real = probs[:, 1]  # ou probs[:, LABEL_REAL]

# Dans utils.py
LABEL_REAL == 1  # Cohérent ! ✅
```

---

## ⚠️ Important

- Les **modèles déjà entraînés** fonctionnent toujours correctement
- Seule l'**interprétation** des prédictions était inversée
- Les scripts d'analyse **inversent temporairement** les anciennes prédictions
- Pour une solution complète : **réentraîner les modèles** avec la nouvelle convention

---

**Date de correction :** 2025-12-02  
**Auteur :** Assistant IA avec Lorcan  
**Fichiers de référence :**
- `MODIFICATIONS_THRESHOLD.md` (corrections du calcul du seuil)
- `LABEL_INVERSION_FIX.md` (ce document - correction des labels)

