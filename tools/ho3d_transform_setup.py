import sys, os 
sys.path.insert(0, os.getcwd())

from utils import * 
from utils_camera import *
from arguments import get_args

def main():
    args = get_args()
    scenes = get_scenes_list(args)
    conversion_map_path = os.path.join(args.data_dir, 'map_to_orig.json')
    conversion_map = read_json(conversion_map_path)
    
    for scene in scenes:
        meta_path = conversion_map[scene]
        image_path = os.path.join('images', f"{scene}_00.png")
        full_image_path = os.path.join(args.data_dir, 'scenes', scene, image_path)
        out_json_path = os.path.join(args.data_dir, 'scenes', scene, 'transforms.json')
        #print('meta_path', meta_path)
        ho3d_pkl_to_transforms(
            pkl_path=meta_path,
            image_path=image_path,
            full_image_path=full_image_path,
            out_json_path=out_json_path
        )
    
if __name__ == "__main__":
    main()