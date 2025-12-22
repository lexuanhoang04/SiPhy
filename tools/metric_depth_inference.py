from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt 

import os 

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str)
parser.add_argument('--save_path', type=str)

args = parser.parse_args()

# image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
# model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")

image_processor = AutoImageProcessor.from_pretrained(
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"
)
model = AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"
)

#image_path = 'data/abo_500/scenes/B00DIHVMEA_ATVPDKIKX0DER/images/B00DIHVMEA_ATVPDKIKX0DER_00.png'
image_path = args.image_path

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
#depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
depth = predicted_depth
#depth = depth.detach().cpu().numpy() * 255
#depth = Image.fromarray(depth.astype("uint8"))

# save depth map with colormap
#save_path = 'data/HO3D_processed_manual/scenes/AP10/depth_pred/AP10_00.png'
save_path = args.save_path
os.makedirs(os.path.dirname(save_path), exist_ok=True)
#plt.imsave(save_path, depth, cmap='magma', vmin=0, vmax=5)

#print('dpeth shape', depth)

#print(f'saved to {save_path}')

# save as numpy array
np.save(save_path.replace('.png', '.npy'), depth.numpy())
print('saved to', save_path.replace('.png', '.npy'))

# save as torch tensor
torch.save(depth, save_path.replace('.png', '.pt'))