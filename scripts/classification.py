"""
classification.py
==================
Compare et classe les alignements simulés vs réels.
Permet d’évaluer la fidélité des simulations par rapport aux données expérimentales.
Utilise une configuration externe et des outils de scoring/statistiques.
"""

import json
from pathlib import Path
import subprocess

def run_classification(realali, simali, output, config, tools):  
    """
    Exécute la pipeline de classification en préparant la configuration JSON.
    Args:
        realali (str): Chemin vers les alignements réels.
        simali (str): Chemin vers les alignements simulés.
        output (str): Dossier de sortie pour les résultats de classification.
        config (str): Chemin vers le fichier de configuration template JSON.
        tools (str): Chemin vers le dossier des outils nécessaires.
    Returns:
        None
    """
    output = Path(output)
    real_ali = Path(realali)
    sim_ali = Path(simali)
    tools = Path(tools)

    output.mkdir(parents=True, exist_ok=True)

    # Charger le fichier de config template
    input_file = Path(config)
    if not input_file.exists():
        raise FileNotFoundError(f"Classification config not found: {input_file}")

    config_path = output / "config.json"

    with open(input_file, 'r') as file:
        data = json.load(file)

    # Mise à jour du JSON avec les chemins réels
    data["out_path"] = str(output)
    data["real_path"] = str(real_ali)
    data["sim_path"] = str(sim_ali)

    # Sauvegarde du nouveau fichier de config
    with open(config_path, 'w') as file:
        json.dump(data, file, indent=4)

    # Exécution du pipeline
    pipeline_script = tools / "simulations-classifiers" / "src" / "classifiers" / "pipeline.py"

    if not pipeline_script.exists():
        raise FileNotFoundError(f"Pipeline script not found: {pipeline_script}")

    command = [
        "uv", "run", "python",
        str(pipeline_script),
        "--config", str(config_path),
        "--no-progress"
    ]

    print(f"\n🚀 Launching classification pipeline: {pipeline_script}")
    subprocess.run(command, check=True)
