
#Il faut activer l'environnements virtuel :
#source path_to_tools/simulations-classifiers/.venv/bin/activate
realali="../data/preprocess/clean_data"
realsimdoss="../results/simulations/BPP"
output="../results/classification"
config="../config/classifiers/sample_conf_dna.json"
tools="../tools"

for simdoss in "$realsimdoss"/*; do
    if [[ -d "$simdoss" ]]; then
        # Extraire uniquement le nom du dossier
        sim_name=$(basename "$simdoss")
        classif_output="$output/$sim_name"
        python3 ../scripts/classification.py \
        --realali $realali \
        --simali $simdoss \
        --output $classif_output \
        --config $config \
        --tools $tools
    fi
done