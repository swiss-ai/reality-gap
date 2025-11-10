#!/bin/bash
# submit_all_languages.sh

LANGUAGES=(
    bosnia-herzegovina bulgaria croatia denmark estonia finland 
    france germany greece iceland italy latvia lithuania malta 
    norway portugal serbia slovakia slovenia sweden uk ukraine
)

for lang in "${LANGUAGES[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --account=root
#SBATCH --job-name=neucodec_${lang}
#SBATCH --output=/users/mrieff/benchmark-audio-tokenizer/logs/neucodec_${lang}_%j.out
#SBATCH --error=/users/mrieff/benchmark-audio-tokenizer/logs/neucodec_${lang}_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu
#SBATCH --partition=normal

export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

source /users/mrieff/benchmark-audio-tokenizer/neucodec-venv/bin/activate
cd /users/mrieff/benchmark-audio-tokenizer

python scripts/neucodec_evaluation.py --language ${lang}
EOF
done