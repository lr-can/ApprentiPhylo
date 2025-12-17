# Dashboard des Modèles de Classification

## Installation

Pour utiliser le dashboard interactif, installez les dépendances nécessaires :

```bash
pip install dash plotly pandas numpy
```

ou

```bash
pip install -r requirements.txt
```

## Utilisation

Lancez l'application Dash :

```bash
python soutenance/models_dashboard.py
```

L'application sera accessible à l'adresse : http://127.0.0.1:8050

## Fonctionnalités

- **Sélection interactive** : Choisissez un modèle dans le menu déroulant
- **Informations détaillées** : Visualisez les paramètres, performances et hyperparamètres de chaque modèle
- **Diagramme d'architecture** : Visualisation interactive de l'architecture de chaque modèle
- **Comparaisons** : Graphiques comparatifs des performances et de la complexité des modèles

## Modèles disponibles

1. **AACnnClassifier** - CNN (meilleur modèle, F1: 0.838)
2. **DenseMsaClassifier** - Réseau dense MSA
3. **DenseSiteClassifier** - Réseau dense Site
4. **LogisticRegressionClassifier** - Régression logistique

