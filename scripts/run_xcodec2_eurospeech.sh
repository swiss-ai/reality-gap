#!/bin/bash
# SLURM batch evaluation for EuroSpeech languages using XCodec2 tokenizer

LANGUAGES=(
    bosnia-herzegovina bulgaria croatia denmark estonia finland 
    france germany greece iceland italy latvia lithuania malta 
    norway portugal serbia slovakia slovenia sweden uk ukraine
)

PROJECT_DIR=/users/lmantel/benchmark-audio-tokenizer
LOG_DIR=${PROJECT_DIR}/logs
VENVSRC=${PROJECT_DIR}/.venv-xcodec2/bin/activate

mkdir -p "${LOG_DIR}"

for lang in "${LANGUAGES[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=xcodec2_${lang}
#SBATCH --environment=ngc-24.11
#SBATCH --output=${LOG_DIR}/xcodec2_${lang}_%j.out
#SBATCH --error=${LOG_DIR}/xcodec2_${lang}_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=normal


export OPENBLAS_NUM_THREADS=72
export OMP_NUM_THREADS=72

source ${VENVSRC}
cd ${PROJECT_DIR}

python scripts/tokenizer_evaluation.py --tokenizer xcodec2 --language ${lang}
EOF
done

echo "Submitted ${#LANGUAGES[@]} XCodec2 evaluation jobs to SLURM"


