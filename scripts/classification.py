import json
import argparse 
from pathlib import Path
import os

if __name__ == "__main__":
    #Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--realali', help ='path to the directory containing the real alignments to be classified.')
    parser.add_argument('--simali', help='path to the directory containing the simulated alignments to be classified.')
    parser.add_argument('--output', help='path to directory where results and updated configuration will be stored.')
    parser.add_argument('--config', help='path to JSON configuration file containing initial parameters for classification pipeline')
    parser.add_argument('--tools', help= 'path to the directory containing the a clone of the gitlab : https://gitlab.in2p3.fr/jbarnier/simulations-classifiers.' )
    args = parser.parse_args()

    #Repositories
    output = Path(args.output)
    real_ali = Path(args.realali)
    sim_ali = Path(args.simali)
    tools = Path(args.tools)
    output.mkdir(parents=True, exist_ok=False)

    #Loading config
    input_file = Path(args.config)
    config = output / "config"  

    with open(input_file, 'r') as file:
        data = json.load(file)

    #Update config
    data["out_path"] = f"{str(output)}"
    data["real_path"] = f"{str(real_ali)}"
    data["sim_path"] = f"{str(sim_ali)}"

    #Save config 
    with open(config, 'w') as file:
        json.dump(data, file, indent=4)

    #Run classification
    command = f"uv run python {tools / 'simulations-classifiers' / 'src' / 'classifiers' / 'pipeline.py'} --config {config} --no-progress"
    os.system(command)