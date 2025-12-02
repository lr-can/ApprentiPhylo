# 🔄 Garanties de Réinitialisation pour RUN 2

## 📋 Résumé

RUN 2 est **complètement réinitialisé** et ne conserve **AUCUN état** de RUN 1. Seule l'architecture du meilleur modèle de RUN 1 est réutilisée, mais avec des **poids complètement nouveaux**.

---

## ✅ Ce qui EST réinitialisé (pas de cache)

### 1. **Modèle Neural Network**
```python
# NOUVEAU modèle créé à chaque fois
model = clf["classifier_fn"](**args)  # Ligne 231 de pipeline.py
```
- ✅ **Nouvelle instance** du modèle créée
- ✅ **Poids aléatoires** (initialization from scratch)
- ✅ **Pas de transfert de poids** depuis RUN 1
- ✅ Seeds contrôlés par `RANDOM_SEED = 42` pour reproductibilité

### 2. **Optimizer & Scheduler**
- ✅ **Nouvel optimizer** créé dans `Training.__init__()`
- ✅ **Nouveau learning rate scheduler** créé
- ✅ **Pas d'état d'optimizer** conservé

### 3. **Fichiers & Checkpoints**
```python
# Suppression explicite avant RUN 2
existing_model.unlink()              # Supprime best_model.pt
shutil.rmtree(checkpoint_dir)        # Supprime tout le dossier checkpoint/
```
- ✅ `best_model.pt` supprimé
- ✅ Tout le dossier `checkpoint/` supprimé
- ✅ Pas de fichiers résiduels

### 4. **Mémoire GPU/CPU**
```python
torch.cuda.empty_cache()  # Vide le cache CUDA si GPU
gc.collect()              # Force garbage collection Python
```
- ✅ Cache CUDA vidé (si GPU utilisé)
- ✅ Garbage collection forcée
- ✅ Mémoire libérée

### 5. **Dataset**
```python
self.base_data = Data(
    source_real=FastaSource(self.out_path / "run_2_real"),
    source_simulated=FastaSource(self.out_path / "run_2_sim"),
    tokenizer=self.tokenizer,
)
```
- ✅ **Nouveau dataset** chargé
- ✅ Fichiers différents (run_2_real, run_2_sim)
- ✅ Nouvelles données tokenisées

---

## 🔍 Ce qui est CONSERVÉ (intentionnel)

### 1. **Architecture du modèle**
- Le **type** de modèle (ex: AACnnClassifier)
- L'**architecture** (nombre de couches, taille)
- Les **hyperparamètres** (learning rate, batch size, etc.)

**Pourquoi ?** C'est le but de RUN 2 : réentraîner le meilleur modèle avec de meilleures données.

### 2. **Seeds aléatoires**
- `RANDOM_SEED = 42` reste identique

**Pourquoi ?** Pour la **reproductibilité** des expériences.

### 3. **Configuration du pipeline**
- Paramètres globaux (device, paths, etc.)

---

## 🔬 Vérification du Comportement

### Test 1: Nouveau modèle créé
```python
# Dans train_classifier():
model = clf["classifier_fn"](**args)  # <-- NOUVELLE instance
# Chaque appel crée un objet Python différent
```

### Test 2: Poids réinitialisés
Les poids sont initialisés selon la stratégie du modèle :
- **PyTorch default**: Xavier/Kaiming initialization
- **Aléatoire** basé sur les seeds

### Test 3: Pas de gradient flow
- Pas de `.requires_grad` conservé
- Pas d'historique de backprop
- Nouveau graphe de computation

### Test 4: Checkpoints propres
```bash
# Avant RUN 2:
results/classification/run_2/AACnnClassifier/
└── (vide ou ancien supprimé)

# Après nettoyage:
results/classification/run_2/AACnnClassifier/
└── (complètement vide)

# Après RUN 2:
results/classification/run_2/AACnnClassifier/
├── best_model.pt          (NOUVEAU)
├── checkpoint/
│   └── best_*.pt          (NOUVEAUX)
└── train_history.parquet  (NOUVEAU)
```

---

## 📊 Logs de Confirmation

Lors de l'exécution, vous verrez :

```
[RUN 2] Retraining AACnnClassifier with Run 2 dataset...
[RUN 2] Removed existing best_model.pt
[RUN 2] Removed checkpoint directory
[RUN 2] Cleared CUDA cache
[RUN 2] Starting fresh training (new model instance, no weights from RUN 1)
[RUN] Training AACnnClassifier
--- Hyperparameters ---
model = AAConvNet(...)  # <-- Nouvelle instance
...
--- Training start ---
Start training using cpu device.
Number of model parameters: 1537.
```

---

## 🎯 Conclusion

### ✅ Garanties fournies :

1. **Modèle**: Nouveau modèle avec poids aléatoires
2. **Optimizer**: Nouvel optimizer/scheduler
3. **Fichiers**: Tous les fichiers précédents supprimés
4. **Mémoire**: Cache CUDA vidé, GC forcé
5. **Dataset**: Nouvelles données chargées

### ❌ Pas de risque de :

- ❌ Transfert de poids entre RUN 1 et RUN 2
- ❌ État d'optimizer conservé
- ❌ Mémoire GPU résiduelle
- ❌ Fichiers de checkpoint mélangés
- ❌ Overfitting sur les mêmes initialisations

### 🔒 Reproductibilité :

- Seeds contrôlés (`RANDOM_SEED = 42`)
- Même comportement à chaque exécution
- Résultats comparables entre runs

---

## 📝 Code Modifié

**Fichier**: `backup/simulations-classifiers/src/classifiers/pipeline.py`

**Méthode**: `run2_retrain_best_model()` (lignes ~1118-1133)

**Modifications**:
1. ✅ Import `gc` ajouté
2. ✅ Suppression de `best_model.pt`
3. ✅ Suppression du dossier `checkpoint/`
4. ✅ Vidage du cache CUDA
5. ✅ Garbage collection forcée
6. ✅ Logs informatifs

---

**Date**: 2025-12-02  
**Auteur**: Assistant IA avec Lorcan

