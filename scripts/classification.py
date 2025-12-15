"""
classification.py
==================
Creates the config.json for the simulations-classifiers pipeline
and launches it (single iteration or two-iteration mode).
"""

import json
from pathlib import Path
import subprocess


def run_classification(realali, simali, output, config, tools, two_iterations=False, threshold=0.5, sim_config_2=None):
    """
    Prépare le fichier de configuration JSON et lance le pipeline.

    Args:
        realali (str): Chemin vers les alignements réels.
        simali (str): Chemin vers les alignements simulés.
        output (str): Dossier de sortie pour les résultats.
        config (str): Chemin vers le fichier de configuration template JSON.
        tools (str): Dossier des outils nécessaires.
        two_iterations (bool): Si True, active Run1 + Run2 automatiquement.
        threshold (float): Seuil de décision pour flagger les simulations "réelles".
        sim_config_2 (dict, optional): Configuration pour générer de nouvelles simulations entre run1 et run2.
    """
    output = Path(output)
    real_ali = Path(realali)
    sim_ali = Path(simali)
    tools = Path(tools)

    output.mkdir(parents=True, exist_ok=True)

    # Charger le template
    template_path = Path(config)
    if not template_path.exists():
        raise FileNotFoundError(f"Classification config not found: {template_path}")

    with open(template_path, "r") as f:
        data = json.load(f)

    # Injecter les chemins réels dans le config
    data["out_path"] = str(output)
    data["real_path"] = str(real_ali)
    data["sim_path"] = str(sim_ali)
    
    # Ajouter sim_config_2 si fourni (pour génération de nouvelles simulations entre run1 et run2)
    if sim_config_2 is not None:
        data["sim_config_2"] = sim_config_2
        # Sauvegarder aussi dans run_1/store_1 pour référence future
        store_dir = output / "run_1" / "store_1"
        store_dir.mkdir(parents=True, exist_ok=True)
        with open(store_dir / "sim_config.json", "w") as f:
            json.dump(sim_config_2, f, indent=4)

    # Sauvegarder le config.json final
    final_config = output / "config.json"
    with open(final_config, "w") as f:
        json.dump(data, f, indent=4)

    # Déterminer la commande
    pipeline_script = tools / "simulations-classifiers" / "src" / "classifiers" / "pipeline.py"

    if not pipeline_script.exists():
        raise FileNotFoundError(f"Pipeline script not found: {pipeline_script}")

    command = [
        "uv", "run", "python",
        str(pipeline_script),
        "--config", str(final_config),
        "--threshold", str(threshold),
        "--no-progress"
    ]

    # Activer run1 + run2
    if two_iterations:
        command.append("--two-iterations")

    print("\nLaunching classification pipeline…")
    print(f"→ script: {pipeline_script}")
    print(f"→ mode: {'TWO-ITERATIONS' if two_iterations else 'RUN1 ONLY'}")
    subprocess.run(command, check=True)
