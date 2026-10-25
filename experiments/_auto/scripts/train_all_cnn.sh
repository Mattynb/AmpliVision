#!/bin/bash

# --- Configuration ---\
#MODELS=("ConvNeXtTiny")
#"MobileNetV3Small" "LENET" "DenseNet121" "MobileNet" "Xception" )

MODELS=("MobileNetV2" "Xception" "MobileNet"  "MobileNetV3Small" "MobileNetV3Large" "EfficientNetB0" "EfficientNetB1" "EfficientNetB2" "EfficientNetV2B0" "EfficientNetV2B1" "DenseNet121" "NASNetMobile" "EfficientNetB3" "EfficientNetB4" "EfficientNetB5" "EfficientNetB6" "EfficientNetV2B2" "EfficientNetV2B3" "EfficientNetV2S" "ConvNeXtTiny" "ResNet50" "ResNet50V2" "InceptionV3" "EfficientNetB7" "EfficientNetV2M" "EfficientNetV2L" "ConvNeXtSmall" "ConvNeXtBase" "ConvNeXtLarge" "ConvNeXtXLarge" "VGG16" "VGG19" "NASNetLarge" "InceptionResNetV2")

JOB="FULL"
PREFIX="NOCROP_DENV_100" #"224s120e10b150ts4vs_LOADtr" #"512s64b2e150ts42vs" # image size, bach, epochs, training steps, validation steps

SLURM_TEMPLATE=$(<_experiments/scripts/slurm_template.txt)

# Submit jobs
for MODEL in "${MODELS[@]}"; do
    
    # replace variables
    #| sed "s/mem=1600gb/mem=900gb/g"  \
    #| sed "s/mem=1600gb/mem=1900gb/g"  \
    MODEL_SLURM_SCRIPT=$(
        echo "$SLURM_TEMPLATE"        \
        | sed "s/MODEL_NAME/$MODEL/g" \
        | sed "s/PREFIX/$PREFIX/g"    \
        | sed "s/JOB/$JOB/g"          \
        | sed "s/task=16/task=32/g"  \
        
    )

    SLURM_FILE="slurm_${MODEL}.sh"
    echo -e "${MODEL_SLURM_SCRIPT}" > "${SLURM_FILE}"
    sbatch "${SLURM_FILE}"
    echo "Submitted SLURM job for: ${MODEL}"
    rm "${SLURM_FILE}"
done

# echo "All model jobs have been submitted."
