import sys, os
sys.path.insert(0, os.getcwd())

from utils import *
from arguments import get_args 

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def scene_depth_inference(scene_dir, image_processor, model, image_path):
    scene_name = os.path.basename(scene_dir)

    #print('image path', image_path)

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
    depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())

    viz_save_path = os.path.join(scene_dir, "depth_viz", f"{scene_name}_00.png")
    save_path = os.path.join(scene_dir, "depth_pred", f"{scene_name}_00.pt")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(viz_save_path), exist_ok=True)

    #print('aver depth', depth.median())
    depth_np = depth.cpu().numpy()
    #print('median', np.median(depth_np[depth_np > 0]))
    plt.imsave(viz_save_path, depth, cmap='magma', vmin=0, vmax=5)

    torch.save(depth, save_path)

    #print(f'saved to {save_path}')
def main():
    args = get_args()

    image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")

    scenes = get_scenes_list(args)

    split_map = {"train": "train", "test": "evaluation"}

    #ho3d_image_dir = os.path.join(args)
    for scene in scenes:
        ho3d_image_path = os.path.join(args.ho3d_dir, split_map[args.split], scene, 'rgb', '0000.jpg')
        scene_dir = os.path.join(args.data_dir, 'scenes', scene)
        scene_depth_inference(scene_dir, image_processor, model, ho3d_image_path)

if __name__ == "__main__":
    main()