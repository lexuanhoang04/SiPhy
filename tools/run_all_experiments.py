import os
import subprocess

scene_name = "SOME_SCENE_NAME"  # ← Change this to the actual scene you want to visualize

patch_modes = ['one_img', 'multiview']
source_modes = ['ours_random', 'ours_complement', 'nerf2phys_first_view']
query_modes = ['ours_mask', 'nerf2phys_dense']

for patch in patch_modes:
    for source in source_modes:
        for query in query_modes:
            if patch == "multiview" and source.startswith("ours"):
                continue  # Invalid combination

            viz_name = f"{patch}_{source}_{query}"

            # Decide script to run
            script = "visualization_2d.py" if patch == "one_img" else "visualization.py"

            cmd = [
                "python", script,
                "--scene_name", scene_name,
                "--patch_features_mode", patch,
                "--source_points_mode", source,
                "--query_points_mode", query,
                "--viz_save_name", viz_name
            ]

            print(f"Running: {viz_name}")
            subprocess.run(cmd)
