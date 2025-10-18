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
    """
    output = Path(output)
    real_ali = Path(realali)
    sim_ali = Path(simali)
    tools = Path(tools)

    output.mkdir(parents=True, exist_ok=True)

    # Chargement de la config
    input_file = Path(config)
    config_path = output / "config.json"

    with open(input_file, 'r') as file:
        data = json.load(file)

    # Mise à jour du JSON
    data["out_path"] = str(output)
    data["real_path"] = str(real_ali)
    data["sim_path"] = str(sim_ali)

    # Sauvegarde du nouveau fichier
    with open(config_path, 'w') as file:
        json.dump(data, file, indent=4)

    # Exécution du pipeline
    command = [
        "uv", "run", "python",
        str(tools / "simulations-classifiers" / "src" / "classifiers" / "pipeline.py"),
        "--config", str(config_path),
        "--no-progress"
    ]
    subprocess.run(command, check=True)