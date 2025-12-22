#!/usr/bin/env python3
"""
organize_data.py

Convert an MVImgNET dump into nerf2phys-style data:
- creates data_custom/scenes/<scene>/images/<scene>_00.png (RGBA with object alpha)
- writes data_custom/splits.json listing all scenes under "train"
"""
import os
import argparse
import json
from PIL import Image
from collections import Counter, defaultdict
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(
        description="Organize MVImgNET into nerf2phys data_custom format"
    )
    p.add_argument(
        "--input_dir",
        help="Path to MVImgNET folder containing .jpg, *_pseudo.png, etc."
    )
    p.add_argument(
        "--mode",
        choices=["gp_gt"], default="gp_gt",
        help="Mode of operation (currently only 'gp_gt' supported)"
    )
    p.add_argument(
        "--output_dir",
        default="data_custom",
        help="Where to write the nerf2phys-formatted data"
    )
    p.add_argument(
        "--debug_scene_name",
        default='None',
        help="If set, only process this scene and print debug info"
    )

    return p.parse_args()

def make_rgba(rgb_path, pseudo_path):
    """
    Loads an RGB image and its colored-pseudo mask,
    builds an alpha channel (255 where pseudo != black),
    and returns a PIL RGBA Image.
    """
    rgb = Image.open(rgb_path).convert("RGB")
    pseudo = Image.open(pseudo_path).convert("RGB")
    # ensure same size
    if pseudo.size != rgb.size:
        pseudo = pseudo.resize(rgb.size, Image.NEAREST)

    w, h = rgb.size
    alpha = Image.new("L", (w, h))
    px_p, px_a = pseudo.load(), alpha.load()

    for y in range(h):
        for x in range(w):
            px_a[x, y] = 255 if px_p[x, y] != (0, 0, 0) else 0

    rgba = rgb.copy()
    rgba.putalpha(alpha)
    return rgba

def count_mvimgnet_types(input_dir):
    """
    Scan `input_dir` and count:
      - original RGB .jpg files
      - *_cutout.png files
      - *_label.png files
      - *_pseudo.png files
    Returns a Counter with keys: "jpg", "cutout", "label", "pseudo", "other".
    """
    counts = Counter()
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(".jpg"):
            counts["jpg"] += 1
        elif fname.lower().endswith("_cutout.png"):
            counts["cutout"] += 1
        elif fname.lower().endswith("_label.png"):
            counts["label"] += 1
        elif fname.lower().endswith("_pseudo.png"):
            counts["pseudo"] += 1
        else:
            counts["other"] += 1
    return counts

def scan_mvimgnet_coverage(input_dir):
    """
    Returns a dict:
      scene_id → set of { 'jpg', 'cutout', 'label', 'pseudo' }
    """
    coverage = defaultdict(set)
    for fname in os.listdir(input_dir):
        lower = fname.lower()
        stem, ext = os.path.splitext(fname)
        if ext == ".jpg":
            scene = stem
            coverage[scene].add("jpg")
        elif stem.endswith("_cutout"):
            scene = stem[:-len("_cutout")]
            coverage[scene].add("cutout")
        elif stem.endswith("_label"):
            scene = stem[:-len("_label")]
            coverage[scene].add("label")
        elif stem.endswith("_pseudo"):
            scene = stem[:-len("_pseudo")]
            coverage[scene].add("pseudo")
        # else: ignore
    return coverage

def report_coverage(coverage):
    scenes = sorted(coverage.keys())
    print(f"→ Found {len(scenes)} unique scene IDs\n")

    # counts per type
    for t in ["jpg","cutout","label","pseudo"]:
        ct = sum(1 for s in scenes if t in coverage[s])
        print(f"  {t:6s}: {ct}")

    # missing lists
    print()
    for t in ["cutout","label","pseudo"]:
        missing = [s for s in scenes if t not in coverage[s]]
        print(f"  Missing {t:6s}: {len(missing)} scenes")
        if missing:
            print("    ", ", ".join(missing))
    print()

def main():
    args = parse_args()

    scenes_root = os.path.join(args.output_dir, "scenes")
    os.makedirs(scenes_root, exist_ok=True)

    scene_names = []
    # cov = scan_mvimgnet_coverage(args.input_dir)
    # report_coverage(cov)

    # counts = count_mvimgnet_types(args.input_dir)
    # print('counts:', counts)

    # print('len intput dir:', len(os.listdir(args.input_dir)))

    for fname in sorted(os.listdir(args.input_dir)):
        if not fname.lower().endswith(".jpg"):
            continue

        # 1. raw_id = remove only the final .jpg
        raw_id = Path(fname).stem                # e.g. "3d001df1.png" or "40000a2d"

        # 2. scene = raw_id without a trailing ".png"
        if raw_id.lower().endswith(".png"):
            scene = raw_id[:-4]
        else:
            scene = raw_id
        
        if args.debug_scene_name != 'None' and scene != args.debug_scene_name:
            continue
        # record it for splits


        jpg_path   = os.path.join(args.input_dir, fname)
        pseudo_path = os.path.join(args.input_dir, f"{raw_id}_pseudo.png")
        label_path  = os.path.join(args.input_dir, f"{raw_id}_label.png")

        if args.debug_scene_name != 'None' and scene == args.debug_scene_name:
            print('label_path:', label_path)
            break

        # skip if missing either mask or label
        if not os.path.exists(pseudo_path):
            print(f"[WARN] skipping {scene}: missing {raw_id}_pseudo.png")
            continue
        if not os.path.exists(label_path):
            print(f"[WARN] skipping {scene}: missing {raw_id}_label.png")
            continue
        
        # check if jpg and label file has same shape
        jpg_img = Image.open(jpg_path)
        label_img = Image.open(label_path)
        if jpg_img.size != label_img.size:
            print(f"[WARN] skipping {scene}: jpg and label size mismatch ({jpg_img.size} vs {label_img.size})")
            break
        
        scene_names.append(scene)

        rgba = make_rgba(jpg_path, pseudo_path)

        out_img_dir = os.path.join(scenes_root, scene, "images")
        os.makedirs(out_img_dir, exist_ok=True)

        out_path = os.path.join(out_img_dir, f"{scene}_00.png")
        rgba.save(out_path)
        print(f"[✓] Wrote {out_path}")


    # simple splits.json (all scenes in train)
    splits = {
        "train": [],
        "val":   [],
        "test":  scene_names
    }
    splits_path = os.path.join(args.output_dir, "splits.json")
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"[✓] Wrote splits file: {splits_path}")

if __name__ == "__main__":
    main()
