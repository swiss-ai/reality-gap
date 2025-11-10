#!/bin/bash
LANGUAGES=(
    ast_es ca_es nl_nl en_us gl_es hu_hu ga_ie kea_cv lb_lu oc_fr es_419 cy_gb
    hy_am be_by cs_cz ka_ge mk_mk pl_pl ro_ro ru_ru
    cmn_hans_cn yue_hant_hk ja_jp ko_kr
    hi_in bn_in ta_in te_in
    th_th vi_vn id_id
    af_za sw_ke am_et yo_ng
    ar_eg tr_tr he_il fa_ir
)

for lang in "${LANGUAGES[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=xcodec2_${lang}
#SBATCH --output=/users/$USER/benchmark-audio-tokenizer/logs/%j_xcodec2_${lang}.out
#SBATCH --error=/users/$USER/benchmark-audio-tokenizer/logs/%j_xcodec2_${lang}.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu
#SBATCH --partition=normal

export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

source /users/$USER/benchmark-audio-tokenizer/.venv-xcodec2/bin/activate
cd /users/$USER/benchmark-audio-tokenizer

python scripts/tokenizer_evaluation.py --tokenizer xcodec2 --language ${lang}
EOF
done

