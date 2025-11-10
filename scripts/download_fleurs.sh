#!/bin/bash
# download_all_fleurs.sh - Batch download FLEURS languages using SLURM

# FLEURS language codes (40 languages - matching download_fleurs.py)
LANGUAGES=(
    # Western Europe
    ast_es ca_es nl_nl en_us gl_es hu_hu ga_ie kea_cv lb_lu oc_fr es_419 cy_gb
    
    # Eastern Europe
    hy_am be_by cs_cz ka_ge mk_mk pl_pl ro_ro ru_ru
    
    # Asia - CJK
    cmn_hans_cn yue_hant_hk ja_jp ko_kr
    
    # Asia - South Asia
    hi_in bn_in ta_in te_in
    
    # Asia - Southeast Asia
    th_th vi_vn id_id
    
    # Africa
    af_za sw_ke am_et yo_ng
    
    # Middle East
    ar_eg tr_tr he_il fa_ir
)


for lang in "${LANGUAGES[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=download_fleurs_${lang}
#SBATCH --output=$HOME/benchmark-audio-tokenizer/logs/%j_download_fleurs_${lang}.out
#SBATCH --error=$HOME/benchmark-audio-tokenizer/logs/%j_download_fleurs_${lang}.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=normal

source /users/$USER/benchmark-audio-tokenizer/.venv/bin/activate
cd /users/$USER/benchmark-audio-tokenizer

python scripts/download_fleurs.py --language ${lang}
EOF
done

echo "Submitted ${#LANGUAGES[@]} FLEURS download jobs to SLURM"