#!/usr/bin/env bash
# File: run_all_jobs.sh
# Purpose: drive every experiment in one go
# Usage:   bash run_all_jobs.sh
# Notes:   -e  -> stop the whole run if any command fails
#          -u  -> treat unset variables as errors
#          -o pipefail -> catch errors inside pipelines
set -euo pipefail

echo "🐍 Activating virtual environment..."
source /research/cvl-lehoang2/uv_envs/nerf2physv2/bin/activate

###############################################################################
# Helper: run a generator Python script and then execute the .sh it writes
# Args:
#   $1  = path of the small script to be generated
#   $@  = the rest of the arguments are passed to pipeline_generation_script.py
###############################################################################
run_job () {
    local OUT_SH="$1"
    shift

    # 1.  Launch Python to *create* the shell file
    python pipeline_generation_script.py "$@" --output_sh "$OUT_SH"

    # 2.  Make sure the new script is executable (first run only)
    chmod +x "$OUT_SH"

    # 3.  Run the script right away
    bash "$OUT_SH"
}

###############################################################################
# ▶▶▶  LIST YOUR JOBS BELOW  ◀◀◀
###############################################################################

# Job 1 – good_volume_test with 3D volume
run_job scripts/evaluate_3dvol.sh \
    --split good_volume_test \
    --all_combi \
    --ks 1 \
    --source_point_strides 10 \
    --dense_point_strides 5 \
    --correction_factors 100 \
    --property_name density \
    --combine_dict viz/viz_combine_dicts/viz_combine_dict_gt.json \
    --feature_save_name 2d_patch \
    --feature_load_name 2d_patch \
    --predict \
    --evaluate \
    --overwrite \
    --device cuda:2 \
    --result_path log/results_all_nerfvol.csv


echo "[✅] Done: evaluate_row17_2.sh"

# Job 2 – 17.3
run_job scripts/evaluate_row17_3.sh \
    --split good_volume_test \
    --all_combi \
    --ks 1 \
    --source_point_strides 10 \
    --dense_point_strides 5 \
    --correction_factors 80 \
    --property_name density \
    --combine_dict viz/viz_combine_dicts/viz_combine_dict_gt.json \
    --feature_save_name 2d_patch \
    --feature_load_name 2d_patch \
    --predict \
    --evaluate \
    --overwrite \
    --device cuda:2 \
    --result_path log/results_all_nerfvol.csv

echo "[✅] Done: scripts/evaluate_row17_3.sh"

# Job 3 – 17.4
run_job scripts/evaluate_row17_4.sh \
    --split good_volume_test \
    --all_combi \
    --ks 1 \
    --source_point_strides 0 \
    --dense_point_strides 5 \
    --correction_factors 80 \
    --property_name density \
    --combine_dict viz/viz_combine_dicts/viz_combine_dict_gt.json \
    --feature_save_name 2d_patch \
    --feature_load_name 2d_patch \
    --use_sam \
    --mask_prior_lambda 2 \
    --predict \
    --evaluate \
    --overwrite \
    --device cuda:2 \
    --result_path log/results_all_nerfvol.csv

echo "[✅] Done: scripts/evaluate_row17_4.sh"

# Job 4 – 17.5
run_job scripts/evaluate_row17_5.sh \
    --split good_volume_test \
    --all_combi \
    --ks 1 \
    --source_point_strides 0 \
    --dense_point_strides 5 \
    --correction_factors 80 \
    --property_name density \
    --combine_dict viz/viz_combine_dicts/viz_combine_dict_gt.json \
    --feature_save_name 2d_patch \
    --feature_load_name 2d_patch \
    --use_sam \
    --top_material \
    --mask_prior_lambda 2 \
    --predict \
    --evaluate \
    --overwrite \
    --device cuda:2 \
    --result_path log/results_all_nerfvol.csv

echo "[✅] Done: scripts/evaluate_row17_5.sh"

# Job 5 – 17.6
python predict_property.py \
    --split good_volume_test \
    --property_name density \
    --dense_point_stride 5 \
    --feature_load_name 2d_patch \
    --feature_save_name 2d_patch \
    --overwrite \
    --testing \
    --model mlp \
    --mlp_checkpoint checkpoints/mlp_contrastive_stride10_one_mask_20250608_034944_95.pth \
    --correction_factor 80 \
    --evaluate \
    --result_path log/results_all_nerfvol.csv

echo "[✅] Done: scripts/evaluate_row17_6.sh"

# Job 6 – 17.7
run_job scripts/evaluate_row17_7_attention_grid.sh \
    --split good_volume_test \
    --all_combi \
    --ks 5 \
    --source_point_strides 10 \
    --dense_point_strides 5 \
    --correction_factors 60 \
    --property_name density \
    --combine_dict viz/viz_combine_dicts/viz_combine_dict_attention_testing.json \
    --feature_save_name 2d_patch \
    --feature_load_name 2d_patch \
    --predict \
    --evaluate \
    --overwrite \
    --device cuda:2 \
    --result_path log/results_all_nerfvol.csv

echo "[✅] Done: scripts/evaluate_row17_7.sh"

# Job 7 – 17.8
run_job scripts/evaluate_row17_8_random.sh \
    --split good_volume_test \
    --all_combi \
    --ks 5 \
    --source_point_strides 10 \
    --dense_point_strides 4 \
    --correction_factors 60 \
    --property_name density \
    --combine_dict viz/viz_combine_dicts/viz_combine_dict_gt.json \
    --feature_save_name 2d_patch \
    --feature_load_name 2d_patch \
    --randomize_prob \
    --predict \
    --evaluate \
    --overwrite \
    --device cuda:2 \
    --result_path log/results_all_nerfvol.csv

echo "[✅] Done: scripts/evaluate_row17_8.sh"

# Job 8 – 17.9
run_job scripts/evaluate_row17_9_onemask_alt.sh \
    --split good_volume_test \
    --all_combi \
    --ks 10 \
    --source_point_strides 10 \
    --dense_point_strides 5 \
    --correction_factors 60 \
    --property_name density \
    --combine_dict viz/viz_combine_dicts/viz_combine_dict_gt.json \
    --feature_save_name 2d_patch \
    --feature_load_name 2d_patch \
    --one_mask \
    --alternate_l2_gt our22 \
    --predict \
    --evaluate \
    --overwrite \
    --device cuda:3 \
    --result_path log/results_all_nerfvol.csv

echo "[✅] Done: scripts/evaluate_row17_9.sh"

# Job 9 – 17.10
python predict_property.py \
    --split good_volume_test \
    --property_name density \
    --feature_load_name 2d_patch \
    --feature_save_name 2d_patch \
    --overwrite \
    --one_mask \
    --materials_existed_name info_orig \
    --evaluate \
    --gt_json storage/MaterialPrediction5/train/_annotations.coco.json\
    --result_path log/results_all_nerfvol.csv

echo "[✅] Done: scripts/evaluate_row17_10.sh"
