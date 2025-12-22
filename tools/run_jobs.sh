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

# ────────────────────────────────────────────────────────────────────────────
# Job 1 – density / good_volume_test / cuda:2
# ────────────────────────────────────────────────────────────────────────────
run_job scripts/evaluate_row2_2d_patch_nerfvol.sh \
    --split good_volume_test \
    --end_idx 1 \
    --all_combi \
    --ks 1 \
    --source_point_strides 0 \
    --dense_point_strides 5 \
    --correction_factors 60 \
    --property_name density \
    --combine_dict viz/viz_combine_dicts/viz_combine_dict_gt.json \
    --feature_save_name 2d_patch \
    --feature_load_name 2d_patch \
    --predict \
    --evaluate \
    --overwrite \
    --device cuda:2 \
    --result_path log/results_all_nerfvol_testing.csv 

echo "Done job 1"
# ────────────────────────────────────────────────────────────────────────────
# Job 2 – another split on cuda:1 (example, duplicate/edit as you wish)
# ────────────────────────────────────────────────────────────────────────────
run_job scripts/evaluate_row3_2d_patch_nerfvol.sh \
    --split good_volume_test \
    --end_idx 1 \
    --all_combi \
    --ks 1 \
    --source_point_strides 10 \
    --dense_point_strides 5\
    --correction_factors 80\
    --property_name density \
    --combine_dict viz/viz_combine_dicts/viz_combine_dict_gt.json \
    --feature_save_name 2d_patch \
    --feature_load_name 2d_patch \
    --predict \
    --evaluate \
    --overwrite \
    --device cuda:2 \
    --result_path log/results_all_nerfvol_testing.csv 

# ────────────────────────────────────────────────────────────────────────────
# Job 3 – another split on cuda:1 (example, duplicate/edit as you wish)
# ────────────────────────────────────────────────────────────────────────────
run_job scripts/evaluate_row4_2d_patch_nerfvol.sh \
    --split good_volume_test \
    --end_idx 1 \
    --all_combi \
    --ks 1 \
    --source_point_strides 0 \
    --dense_point_strides 5\
    --correction_factors 80\
    --property_name density \
    --combine_dict viz/viz_combine_dicts/viz_combine_dict_gt.json \
    --feature_save_name 2d_patch \
    --feature_load_name 2d_patch \
    --use_sam \
    --mask_prior_lambda 2\
    --predict \
    --evaluate \
    --overwrite \
    --device cuda:2 \
    --result_path log/results_all_nerfvol_testing.csv 


echo "Done job 3"

# ────────────────────────────────────────────────────────────────────────────
# Job 4 – another split on cuda:1 (example, duplicate/edit as you wish)
# ────────────────────────────────────────────────────────────────────────────
run_job scripts/evaluate_row5_2d_patch_nerfvol.sh \
    --split good_volume_test \
    --end_idx 1 \
    --all_combi \
    --ks 1 \
    --source_point_strides 0 \
    --dense_point_strides 5\
    --correction_factors 80\
    --property_name density \
    --combine_dict viz/viz_combine_dicts/viz_combine_dict_gt.json \
    --feature_save_name 2d_patch \
    --feature_load_name 2d_patch \
    --use_sam \
    --top_material\
    --mask_prior_lambda 2\
    --predict \
    --evaluate \
    --overwrite \
    --device cuda:2 \
    --result_path log/results_all_nerfvol_testing.csv 


echo "Done job 4"

# ────────────────────────────────────────────────────────────────────────────
# Job 4 – another split on cuda:1 (example, duplicate/edit as you wish)
# ────────────────────────────────────────────────────────────────────────────
run_job scripts/evaluate_row5_2d_patch_nerfvol.sh \
    --split good_volume_test \
    --end_idx 1 \
    --all_combi \
    --ks 1 \
    --source_point_strides 0 \
    --dense_point_strides 5\
    --correction_factors 80\
    --property_name density \
    --combine_dict viz/viz_combine_dicts/viz_combine_dict_gt.json \
    --feature_save_name 2d_patch \
    --feature_load_name 2d_patch \
    --use_sam \
    --top_material\
    --mask_prior_lambda 2\
    --predict \
    --evaluate \
    --overwrite \
    --device cuda:2 \
    --result_path log/results_all_nerfvol_testing.csv 


echo "Done job 4"

# Add as many blocks as you need. Every block:
#   run_job  <output-script>  <arguments passed to pipeline_generation_script.py>
###############################################################################
echo "✅ All jobs finished successfully."
