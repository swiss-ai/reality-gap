#!/bin/bash
# SLURM batch evaluation for EuroSpeech languages using XCodec2 tokenizer

LANGUAGES=(
    bosnia-herzegovina bulgaria croatia denmark estonia finland 
    france germany greece iceland italy latvia lithuania malta 
    norway portugal serbia slovakia slovenia sweden uk ukraine
)

mkdir -p /users/${USER}/benchmark-audio-tokenizer/logs

for lang in "${LANGUAGES[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=xcodec2_${lang}
#SBATCH --output=/users/${USER}/benchmark-audio-tokenizer/logs/%j_xcodec2_${lang}.out
#SBATCH --error=/users/${USER}/benchmark-audio-tokenizer/logs/%j_xcodec2_${lang}.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=normal


export OPENBLAS_NUM_THREADS=72
export OMP_NUM_THREADS=72

source /users/${USER}/benchmark-audio-tokenizer/.venv-xcodec2/bin/activate
cd ${PROJECT_DIR}

python scripts/tokenizer_evaluation.py --tokenizer xcodec2 --language ${lang}
EOF
done

echo "Submitted ${#LANGUAGES[@]} XCodec2 evaluation jobs to SLURM"


