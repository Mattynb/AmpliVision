#! /bin/bash

RESULTS_FOLDER_PATH="data/results/06-27-2024/"
SAVE_PATH="data/generated_results"

for i in {1..3}
do
    n=$((10**i))
    python DataGenerator -n ${n} -s ${SAVE_PATH} -r ${RESULTS_FOLDER_PATH}
    sleep 1
    python SampleClassifier
done