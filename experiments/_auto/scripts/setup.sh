JOB="GENERATE_DATA" #"GENERATE_DATA"
PREFIX="setup"

echo "Setting up"
MODELS=("MobileNetV2") 

# generating data
SLURM_TEMPLATE=$(cat _experiments/scripts/slurm_template.txt)

# Submit jobs
for MODEL in "${MODELS[@]}"; do

    # replace variables
    #| sed "s/task=16/task=128/g"  \
    #| sed "s/mem=1600gb/mem=1900gb/g"  \
    MODEL_SLURM_SCRIPT=$(
        echo "$SLURM_TEMPLATE"        \
        | sed "s/MODEL_NAME/$MODEL/g" \
        | sed "s/PREFIX/$PREFIX/g"    \
        | sed "s/JOB/$JOB/g"          \
        | sed "s/#SBATCH --gres/##SBATCH --gres/g" \
        | sed "s/scanned_MARKER/scanned_DENV/g"
    )
    SLURM_FILE="slurm_${MODEL}.sh"
    echo -e "${MODEL_SLURM_SCRIPT}" > "${SLURM_FILE}"
    
    sbatch "${SLURM_FILE}"
    echo "Submitted SLURM job for: ${MODEL}"
    
    rm "${SLURM_FILE}"
done

echo "done"
