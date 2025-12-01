#!/bin/bash

cd "$(dirname "$0")"

python3 -m venv Project_environment

source Project_environment/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# echo "Installation complete. To activate the virtual environment, run 'source install.sh'."