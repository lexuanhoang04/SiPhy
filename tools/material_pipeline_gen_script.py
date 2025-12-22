#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.getcwd())

from arguments import get_args

def generate_script(args, output_script="run_physx_pipeline.sh"):
    script_content = f"""#!/bin/bash
# Auto-generated PhysX pipeline runner
set -euo pipefail

echo "[0.2] Feature fusion"
python feature_fusion.py \\
    --general_config {args.general_config} \\
    --data_dir {args.data_dir} \\
    --split {args.split} \\
    --start_idx {args.start_idx} \\
    --end_idx {args.end_idx} \\
    --device {args.device} \\

echo "[0.3] Captioning"
python captioning.py \\
    --general_config {args.general_config} \\
    --start_idx {args.start_idx} \\
    --end_idx {args.end_idx} \\
    --split {args.split} \\
    --device {args.device} \\

echo "[0.4] Material proposal"
python material_proposal.py \\
    --general_config {args.general_config}\\
    --start_idx {args.start_idx} \\
    --end_idx {args.end_idx} \\
    --split {args.split} \\
    --device {args.device} \\

echo "[0.4.1] Sanity check"
python {args.split}1/sanity_check_info.py \\
    --general_config {args.general_config} \\
    --start_idx {args.start_idx} \\
    --end_idx {args.end_idx} \\
    --split {args.split} \\

echo "[0.5] Collect new materials"
python misc/collect_materials.py \\
    --general_config {args.general_config} \\
    --split {args.split}

echo "✅ Pipeline finished."
"""
    with open(output_script, "w") as f:
        f.write(f"source /egr/research-zijunlab/lehoang2/uv_envs/nerf2phys5/bin/activate\n\n")
        f.write(script_content)
    os.chmod(output_script, 0o755)
    print(f"Shell script written to {output_script}")

if __name__ == "__main__":
    args = get_args()
    generate_script(args, args.output_sh)
