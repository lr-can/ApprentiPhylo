mkdir -p ../results/classification_results

python3 ../scripts/classification_results.py \
--input ../data/preprocess/gap_and_ambigless \
--output ../results/classification_results/accuracy.res \
--metric accuracy

python3 ../scripts/classification_results.py \
--input ../data/preprocess/gap_and_ambigless \
--output ../results/classification_results/f1_score.res \
--metric f1_score

python3 ../scripts/conf_matrix.py \
--input ../data/preprocess/gap_and_ambigless \
--output ../results/classification_results/conf_matrix.png \
--cols 4

python3 ../scripts/training_loss.py \
--input ../data/preprocess/gap_and_ambigless \
--output ../results/classification_results/training_loss.png \
--cols 2

python3 ../scripts/plot_accuracies.py \
--input ../results/classification_results/accuracy.res \
--output ../results/classification_results/plot_accuracies.png