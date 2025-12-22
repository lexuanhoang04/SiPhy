#!/usr/bin/env python3
import os, sys

sys.path.insert(0, os.getcwd())

import json
import argparse

from arguments import get_args
from gpt_inference import parse_material_list, parse_material_hardness
from visualization import make_legend

from utils import read_json, parse_material_gt  # <-- only what you need
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.draw import polygon

def main():
    args = get_args()

    # Load COCO JSON
    with open(args.gt_json, "r") as f:
        coco = json.load(f)

    # Build category_id → name map
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}

    # Group annotations by image_id
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    if args.viz_only:
        material_to_supercat = read_json(args.material_to_supercat_path)
        supercat_to_idx = read_json(args.supercat_to_idx_path)

    # Process each image as a "scene"
    for img in coco["images"]:
        img_id = img["id"]
        # Derive scene_name from extra.name or file_name
        scene_name = os.path.splitext(
            img.get("extra", {}).get("name", img["file_name"])
        )[0]
        if args.debug_scene_name != "None" and scene_name != args.debug_scene_name:
            continue
        #print('scene_name:', scene_name)

        scene_dir = os.path.join(args.data_dir, 'scenes', scene_name)
        #os.makedirs(scene_dir, exist_ok=True)

        gt_path = os.path.join(scene_dir, f"{args.materials_existed_name}_gt_sim_vecs.json")
        if os.path.exists(gt_path) and not args.overwrite:
            print(f"Skipping scene '{scene_name}' as {args.materials_existed_name}_gt_sim_vecs.json already exists.")
            if args.viz_only:
                   # ---- viz_only branch ----
                with open(gt_path, "r") as f:
                    gt_vecs = json.load(f)

                # rebuild material list
                if args.materials_existed_name == "info_orig":
                    info_file = os.path.join(
                        scene_dir,
                        f"{args.mats_load_name}.json"
                    )
                    with open(info_file, "r") as f:
                        info = json.load(f)
                    mat_names, _ = parse_material_list(
                        info[f"candidate_materials_{args.property_name}"]
                    )
                else:
                    mats_path = os.path.join(
                        scene_dir,
                        f"{args.materials_existed_name}.json"
                    )
                    with open(mats_path, "r") as f:
                        caption = json.load(f)["caption"]
                    mat_names = [m.strip() for m in caption.split(",")]

                # prepare colors and legend
                cmap = mpl.colormaps["tab10"]
                colors = [cmap(i) for i in range(len(mat_names))]

                # plot
                fig, ax = plt.subplots(figsize=(8,8))
                img_file = os.path.join(scene_dir, "images", f"{scene_name}_00.png")
                ax.imshow(plt.imread(img_file))
                anns_sorted = sorted(anns_by_img.get(img_id, []), key=lambda a: a["id"])
                preds = []
                for idx, ann in enumerate(anns_sorted):
                    seg = ann["segmentation"][0]
                    coords = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                    vec = gt_vecs[f"mask_{idx}"]
                    pred = max(range(len(vec)), key=lambda j: vec[j])
                    poly = mpl.patches.Polygon(
                        coords,
                        closed=True,
                        edgecolor=colors[pred],
                        facecolor=colors[pred],
                        linewidth=2
                    )
                    ax.add_patch(poly)
                    preds.append(pred)

                # 2) Now build your global_map *using* those preds
                H, W = img['height'], img['width']
                global_map = np.full((H, W), fill_value=-1, dtype=np.int32)

                for ann, pred_local in zip(anns_sorted, preds):
                    # map pred_local → mat_name → supercat → global_idx
                    mat_name   = mat_names[pred_local]
                    #print('mat_name:', mat_name)
                    supercat   = material_to_supercat.get(mat_name, "unknown")
                    global_idx = supercat_to_idx.get(supercat, -1)

                    # rasterize all parts
                    for seg in ann['segmentation']:
                        xs = np.array(seg[0::2], dtype=np.int32)
                        ys = np.array(seg[1::2], dtype=np.int32)
                        rr, cc = polygon(ys, xs, shape=(H, W))
                        global_map[rr, cc] = global_idx

                # 3) Save it once
                segmap_path = os.path.join(scene_dir, f"global_material_map.npy")
                np.save(
                    segmap_path,
                    global_map
                )
                print(f" → saved global map: {global_map.shape} → {segmap_path}")

                make_legend(
                    colors,
                    mat_names,
                    savefile=os.path.join(scene_dir, f"{args.materials_existed_name}_legend.png"),
                    show=args.show
                )
                fig.savefig(
                    os.path.join(scene_dir, f"{args.materials_existed_name}_viz_masks.png"),
                    bbox_inches="tight"
                )
                print(f'saved to {os.path.join(scene_dir, f"{args.materials_existed_name}_viz_masks.png")}')
                plt.close(fig)

            continue

        anns = anns_by_img.get(img_id, [])
        if not anns:
            continue

        # 1. Sort masks by annotation id for fixed order
        anns_sorted = sorted(anns, key=lambda x: x["id"])

        # 2. Unique materials in this scene, sorted by category_id
        unique_cids = sorted({ann["category_id"] for ann in anns_sorted})
        materials = [cat_map[cid] for cid in unique_cids]
        caption = ", ".join(materials)

        # 3. Save materials_existed file
        mats_path = os.path.join(
            scene_dir, f"{args.materials_existed_name}.json"
        )
        # Check if file exists and if caption key is already there
        existing_data = {}
        if os.path.exists(mats_path):
            with open(mats_path, "r") as f:
                existing_data = json.load(f)
        
        # Only update if 'caption' doesn't exist
        if 'caption' not in existing_data:
            existing_data['caption'] = caption
            with open(mats_path, "w") as f:
                json.dump(existing_data, f, indent=2)

        # 4. Build one-hot vector per mask
        if args.materials_existed_name != 'info_orig':
            k = len(unique_cids)
            gt_vecs = {}
            for mask_idx, ann in enumerate(anns_sorted):
                vec = [0] * k
                mat_idx = unique_cids.index(ann["category_id"])
                vec[mat_idx] = 1
                gt_vecs[f"mask_{mask_idx}"] = vec
        else:
            #print('hehe')
            info_file = os.path.join(scene_dir, '%s.json' % args.mats_load_name)

            with open(info_file, "r") as f:
                info = json.load(f)

            #preparing material info
            mat_val_list = info['candidate_materials_%s' % args.property_name]
            mat_names, mat_vals = parse_material_list(mat_val_list)
            k = len(mat_names)
            gt_vecs = {}
            for mask_idx, ann in enumerate(anns_sorted):
                vec = [0] * k
                mats = cat_map[ann["category_id"]]
                mats_dict = parse_material_gt(mats)
                if args.debug_scene_name != "None":
                    print('mats:', mats)
                    print('mats_dict:', mats_dict)
                for idx, (mat_name, mat_val) in enumerate(mats_dict):
                    try:
                        vec[idx] = mat_val
                    except IndexError:
                        print(f"IndexError: {idx} for material '{mat_name}' in scene '{scene_name}'")
                        continue
                gt_vecs[f"mask_{mask_idx}"] = vec

        # Save gt_sim_vecs.json
        gt_path = os.path.join(scene_dir, f"{args.materials_existed_name}_gt_sim_vecs.json")
        with open(gt_path, "w") as f:
            json.dump(gt_vecs, f, indent=2)

        print(f"[✓] Processed scene '{scene_name}': {k} materials, {len(anns_sorted)} masks")

        if args.debug_scene_name != "None":
            break
        
if __name__ == "__main__":
    main()
