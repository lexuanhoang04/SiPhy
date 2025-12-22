import sys, os 
sys.path.insert(0, os.getcwd())

from utils import *
from arguments import get_args  

from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt 


def main():
    args = get_args()
    scenes = get_scenes_list(args)
    image_processor = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"
    )

    for scene in scenes:
        image_path = os.path.join(args.data_dir, 'scenes', scene, 'images', f"{scene}_00.png")
        image = Image.open(image_path).convert("RGB")

        # prepare image for the model
        inputs = image_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        # interpolate to original size and visualize the prediction
        post_processed_output = image_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )

        predicted_depth = post_processed_output[0]["predicted_depth"]
        save_path = os.path.join(args.data_dir, 'scenes', scene, 'depth_pred', f"{scene}_00.pt")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(predicted_depth, save_path)
        print(f"Saved depth map for scene {scene} to {save_path}")
if __name__ == "__main__":
    main()