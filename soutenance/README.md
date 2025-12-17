# Installation des polices Atkinson Hyperlegible

Ce dossier contient les polices **Atkinson Hyperlegible** développées par le Braille Institute of America, Inc. pour améliorer la lisibilité.

Les polices se trouvent dans le dossier `./fonts`.

## Polices incluses

- `AtkinsonHyperlegible-Bold.ttf`
- `AtkinsonHyperlegible-BoldItalic.ttf`
- `AtkinsonHyperlegible-Italic.ttf`
- `AtkinsonHyperlegible-Regular.ttf`

## Licence

Ces polices sont distribuées sous la licence **SIL Open Font License, Version 1.1**. Voir le fichier `fonts/OFL.txt` pour plus de détails.

---

## Installation

### Windows

#### Méthode 1 : Installation via l'interface graphique

1. Ouvrez l'Explorateur de fichiers et naviguez vers le dossier `fonts`
2. Sélectionnez tous les fichiers `.ttf` (maintenez `Ctrl` et cliquez sur chaque fichier)
3. Cliquez avec le bouton droit sur la sélection
4. Choisissez **"Installer"** ou **"Installer pour tous les utilisateurs"**
5. Les polices seront installées et disponibles immédiatement

#### Méthode 2 : Installation via PowerShell

Ouvrez PowerShell en tant qu'administrateur et exécutez :

```powershell
$fontPath = "C:\chemin\vers\soutenance\fonts"
Get-ChildItem -Path $fontPath -Filter "*.ttf" | ForEach-Object {
    $fontFile = $_.FullName
    $destination = "C:\Windows\Fonts\$($_.Name)"
    Copy-Item -Path $fontFile -Destination $destination -Force
    New-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts" -Name $_.Name.Replace('.ttf',' (TrueType)') -Value $_.Name -Force | Out-Null
}
```

Remplacez `C:\chemin\vers\soutenance\fonts` par le chemin absolu vers le dossier `fonts`.

#### Méthode 3 : Installation via l'invite de commande

```cmd
xcopy /Y "C:\chemin\vers\soutenance\fonts\*.ttf" "C:\Windows\Fonts\"
```

Remplacez `C:\chemin\vers\soutenance\fonts` par le chemin absolu vers le dossier `fonts`.

---

### macOS

#### Méthode 1 : Installation via le Finder

1. Ouvrez le Finder et naviguez vers le dossier `fonts`
2. Double-cliquez sur chaque fichier `.ttf`
3. Dans la fenêtre d'aperçu qui s'ouvre, cliquez sur **"Installer la police"**
4. Répétez pour tous les fichiers `.ttf`

#### Méthode 2 : Installation via Terminal

```bash
# Depuis le dossier courant (soutenance), copier toutes les polices vers le répertoire utilisateur
cp ./fonts/*.ttf ~/Library/Fonts/

# Ou pour tous les utilisateurs (nécessite les droits administrateur)
sudo cp ./fonts/*.ttf /Library/Fonts/
```

#### Méthode 3 : Installation via Font Book

1. Ouvrez **Font Book** (Livre de polices)
2. Glissez-déposez tous les fichiers `.ttf` du dossier `fonts` dans Font Book
3. Les polices seront automatiquement installées

---

### Linux

#### Méthode 1 : Installation pour l'utilisateur courant (recommandé)

```bash
# Depuis le dossier courant (soutenance), créer le répertoire des polices utilisateur s'il n'existe pas
mkdir -p ~/.local/share/fonts

# Copier toutes les polices
cp ./fonts/*.ttf ~/.local/share/fonts/

# Actualiser le cache des polices
fc-cache -f -v
```

#### Méthode 2 : Installation système (pour tous les utilisateurs)

```bash
# Depuis le dossier courant (soutenance), copier toutes les polices vers le répertoire système (nécessite sudo)
sudo cp ./fonts/*.ttf /usr/share/fonts/truetype/

# Actualiser le cache des polices
sudo fc-cache -f -v
```

#### Méthode 3 : Installation dans un répertoire personnalisé

```bash
# Créer un répertoire personnalisé
sudo mkdir -p /usr/local/share/fonts/atkinson-hyperlegible

# Depuis le dossier courant (soutenance), copier les polices
sudo cp ./fonts/*.ttf /usr/local/share/fonts/atkinson-hyperlegible/

# Actualiser le cache
sudo fc-cache -f -v
```

#### Vérification de l'installation (Linux)

Pour vérifier que les polices sont bien installées :

```bash
# Lister les polices installées contenant "Atkinson"
fc-list | grep -i atkinson
```

---

## Vérification de l'installation

Après l'installation, redémarrez vos applications si nécessaire pour que les polices soient disponibles. Les polices devraient apparaître sous le nom **"Atkinson Hyperlegible"** dans la liste des polices de vos applications.

---

## Désinstallation

### Windows
Supprimez les fichiers `.ttf` du dossier `C:\Windows\Fonts\` et les entrées correspondantes dans le registre.

### macOS
Supprimez les fichiers depuis Font Book ou du dossier `~/Library/Fonts/` ou `/Library/Fonts/`.

### Linux
```bash
# Pour l'utilisateur
rm ~/.local/share/fonts/AtkinsonHyperlegible*.ttf
fc-cache -f -v

# Pour le système
sudo rm /usr/share/fonts/truetype/AtkinsonHyperlegible*.ttf
sudo fc-cache -f -v
```

---

## Dashboard de Classification

Le fichier `classifier_dashboard.py` est un dashboard interactif créé avec Streamlit pour visualiser et présenter les résultats du pipeline de classification.

### Installation des dépendances

```bash
pip install streamlit plotly pandas polars biopython
```

### Utilisation

Pour lancer le dashboard :

```bash
streamlit run classifier_dashboard.py
```

Le dashboard s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`.

### Fonctionnalités

Le dashboard permet de visualiser :

1. **Vue d'ensemble des données** : Statistiques sur les alignements réels et simulés
2. **Classificateurs** : Informations détaillées sur chaque classificateur et leur processus d'embedding
3. **Résultats d'entraînement** : Courbes d'apprentissage, métriques de performance
4. **Prédictions** : Distribution des probabilités, meilleures prédictions
5. **Courbes ROC** : Performance de classification
6. **Comparaison** : Comparaison des performances entre différents classificateurs

### Configuration

Dans la barre latérale du dashboard, vous pouvez configurer :

- **Chemin données réelles** : Répertoire contenant les alignements réels (format FASTA)
- **Chemin données simulées** : Répertoire contenant les alignements simulés (format FASTA)
- **Chemin résultats classification** : Répertoire contenant les résultats de classification (généré par le pipeline)
- **Run** : Sélectionner le run 1 ou 2 à visualiser

### Structure des résultats attendue

Le dashboard s'attend à trouver les résultats dans la structure suivante :

```
results/classification/
├── run_1/
│   ├── AACnnClassifier/
│   │   ├── train_history.parquet
│   │   └── best_preds.parquet
│   ├── LogisticRegressionClassifier/
│   │   └── ...
│   └── roc_data/
│       ├── AACnnClassifier_roc.csv
│       └── ...
└── run_2/
    └── ...
```

### Utilisation pour présentation

Le dashboard est conçu pour être utilisé lors de présentations PowerPoint. Vous pouvez :

1. Lancer le dashboard en mode plein écran
2. Naviguer entre les différentes sections
3. Capturer des captures d'écran des visualisations pour les inclure dans votre présentation
4. Utiliser les graphiques interactifs Plotly qui peuvent être exportés en images haute résolution

### Export des graphiques

Les graphiques Plotly peuvent être exportés en cliquant sur l'icône de caméra dans la barre d'outils de chaque graphique, ou en utilisant le menu contextuel (clic droit) pour exporter en PNG, SVG ou HTML.

---

## Ressources

- [Braille Institute](https://www.brailleinstitute.org/freefont)
- [SIL Open Font License](https://openfontlicense.org)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)

