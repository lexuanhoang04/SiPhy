import sys, os 
#sys.path.insert(0, os.getcwd())

import numpy as np
import json
import torch
import open_clip
from PIL import Image

from utils import *
try:
    from arguments import get_args
except:
    from Nerf2Physic_Generalization.arguments import get_args
from tqdm import tqdm
import matplotlib.pyplot as plt

CLIP_BACKBONE = 'ViT-B-16'
CLIP_CHECKPOINT = 'datacomp_xl_s13b_b90k'
CLIP_INPUT_SIZE = 224
CLIP_OUTPUT_SIZE = 512

def plot_sampled_points(image, points, save_path='sampled_points_overlay.png'):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.scatter(points[:, 0], points[:, 1], c='red', s=2, marker='o')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def get_patch_features_from_2d(img, pts_2d, model, preprocess_fn,
                               patch_size=56, batch_size=8, device='cuda'):
    
    # print('get_patch_features_from_2d')
    # print('pts_2d', len(pts_2d))
    if args.verbose:
        print('starting get_patch_features_from_2d, num points:', len(pts_2d))
        
    h, w, c = img.shape
    half_patch = patch_size // 2
    n_pts = len(pts_2d)

    # Debug visualization
    plot_sampled_points(img, np.array(pts_2d), save_path=f'viz/{args.start_idx}_{args.end_idx}_{args.feature_save_name}_debug_sampled_points.png')

    patch_features = torch.zeros(n_pts, CLIP_OUTPUT_SIZE, device=device)
    is_valid = torch.zeros(n_pts, dtype=torch.bool, device=device)
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        model.to(device)
        for batch_start in range(0, n_pts, batch_size):
            curr_batch_size = min(batch_size, n_pts - batch_start)
            batch_patches = torch.zeros(curr_batch_size, 3, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE, device=device)

            for j in range(curr_batch_size):
                x, y = pts_2d[batch_start + j]

                if x < half_patch or x >= w - half_patch or y < half_patch or y >= h - half_patch:
                    continue
                
                #Get RGB and alpha separately
                patch_rgba = img[y - half_patch:y + half_patch, x - half_patch:x + half_patch, :]
                patch_rgb = patch_rgba[:, :, :3]
                patch_alpha = patch_rgba[:, :, 3]

                # Generate binary mask: foreground = alpha > 0
                patch_mask = patch_alpha > 0
                
                if args.reject_background_thr > 0:
                        
                    # Reject patch if too much background (less than 70% foreground)
                    if np.mean(patch_mask) < args.reject_background_thr / 100:
                        continue

                #patch_rgb = img[y - half_patch:y + half_patch, x - half_patch:x + half_patch, :3]

                # Convert to PIL and preprocess
                patch_rgb = Image.fromarray(patch_rgb)
                patch_tensor = preprocess_fn(patch_rgb).unsqueeze(0).to(device)

                batch_patches[j] = patch_tensor
                is_valid[batch_start + j] = True


            valid_mask = is_valid[batch_start:batch_start + curr_batch_size]
            if valid_mask.any():
                feats = model.encode_image(batch_patches)
                patch_features[batch_start:batch_start + curr_batch_size] = feats

    return patch_features.unsqueeze(0), is_valid.unsqueeze(0)


def get_patch_features(pts, imgs, depths, w2cs, K, model, preprocess_fn, occ_thr,
                       patch_size=56, batch_size=8, device='cuda', scene_name='None', masks=None):
    n_imgs = len(imgs)
    n_pts = len(pts)

    patch_features = torch.zeros(n_imgs, n_pts, CLIP_OUTPUT_SIZE, device=device, requires_grad=False)
    is_visible = torch.zeros(n_imgs, n_pts, device=device, dtype=torch.bool, requires_grad=False)
    half_patch_size = patch_size // 2

    K = np.array(K)
    with torch.no_grad(), torch.cuda.amp.autocast():
        model.to(device)

        visible_pts_3d_view0 = []
        visible_pts_2d_view0 = []
        pts_2d_first_view = project_3d_to_2d(pts, w2cs[0], K)
        for i in tqdm(range(n_imgs), desc="Processing images", leave=False):
            if '2d' in args.feature_save_name and 'multi' not in args.feature_save_name and i != 0:
                continue
            h, w, c = imgs[i].shape
            if len(K.shape) == 3:
                curr_K = K[i]
            else:
                curr_K = K
            pts_2d, dists = project_3d_to_2d(pts, w2cs[i], curr_K, return_dists=True)

            
            if args.feature_save_name == 'physx_aligned' and masks is not None:
                #print('hehehe')
                viz_savepath = None
                if args.viz_mode == 'viz_align':
                    viz_savepath = f'viz/{scene_name}/aligned_view{i:02d}.png'
                pts_2d_aligned, _ = align_points_to_foreground(imgs[i], masks[i], pts_2d, viz_savepath=viz_savepath)
                #pts_2d_best, _ = align_points_to_foreground_fine(imgs[i], masks[i], pts_2d, viz_savepath=viz_savepath)
                pts_2d = pts_2d_aligned
            
            pts_2d = np.round(pts_2d).astype(np.int32)
            observed_dists = depths[i]
        
            # loop through pts in batches
            for batch_start in range(0, n_pts, batch_size):
                curr_batch_size = min(batch_size, n_pts - batch_start)
                batch_patches = torch.zeros(curr_batch_size, 3, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE, device=device)
                
                for j in range(curr_batch_size):
                    x, y = pts_2d[batch_start + j]

                    if x >= half_patch_size and x < w - half_patch_size and \
                       y >= half_patch_size and y < h - half_patch_size:
                        is_occluded = dists[batch_start + j] > observed_dists[y, x] + occ_thr
                        if not is_occluded:
                            if i == 0:
                                visible_pts_3d_view0.append(pts[batch_start + j])  # 3D point
                                visible_pts_2d_view0.append([x, y])  

                            patch = imgs[i][y - half_patch_size:y + half_patch_size, x - half_patch_size:x + half_patch_size]
                            patch = Image.fromarray(patch)
                            
                            patch = preprocess_fn(patch).unsqueeze(0).to(device)
                            batch_patches[j] = patch
                            is_visible[i, batch_start + j] = True


                if is_visible[i, batch_start:batch_start + batch_size].any():
                    patch_features[i, batch_start:batch_start + curr_batch_size] = model.encode_image(batch_patches)
            if args.plot_source_points:
                print(f"Image {i}: {is_visible[i].sum().item()} visible points")

        if 'first_view_2d' in args.feature_save_name:
            patch_features = patch_features[0].unsqueeze(0)
            is_visible = is_visible[0].unsqueeze(0)
    return patch_features, is_visible, visible_pts_3d_view0, visible_pts_2d_view0, pts_2d_first_view


def process_scene(args, scene_dir, model, preprocess_fn):
    
    scene_name = os.path.basename(scene_dir)

    depth_dir = os.path.join(scene_dir, 'ns', 'renders', 'depth')
    
    
    if args.feature_save_name != '2d_patch':
        dt_file = os.path.join(scene_dir, 'ns', 'dataparser_transforms.json')
        t_file = os.path.join(scene_dir, 'transforms.json')
        w2cs, K = parse_transforms_json(t_file, return_w2c=True, different_Ks=args.different_Ks)

        if not 'physx' in args.feature_save_name:
            pcd_file = os.path.join(scene_dir, 'ns', 'point_cloud.ply')
            pts = load_ns_point_cloud(pcd_file, dt_file, ds_size=args.feature_voxel_size)
        else:
            pcd_file = os.path.join(scene_dir, 'physx', f'point_cloud_ds_{args.translate_method}.ply')
            pts = load_ply_point_cloud(pcd_file, ds_size=None)

        if args.verbose:
            print('source point shape:', pts.shape)

        
        #if args.feature_save
        
        #pts = load_ns_point_cloud(pcd_file, dt_file, ds_size=None)
        ns_transform, scale = parse_dataparser_transforms_json(dt_file)

        depths = load_depths(depth_dir, Ks=None)

        occ_thr = args.occ_thr * scale

    img_dir = os.path.join(scene_dir, 'images')


    imgs, masks = load_images(img_dir, return_masks=True)



    #print('preprocess list', args.preprocess_list)
    
    if 'clahe' in args.preprocess_list:
        imgs = [clahe_equalize(img) for img in imgs]
    
    #preprocess_name = name_creation(args.preprocess_list)
    
    names_dict = name_creation(args)

    if '2d_patch' in args.feature_save_name:
        print(f"[2D MODE] Processing scene: {scene_name}")

        # img_dir = os.path.join(scene_dir, 'images')
        # img_files = sorted(os.listdir(img_dir))
        # img_path = os.path.join(img_dir, img_files[0])
        # img = np.array(Image.open(img_path))
        # mask = img[:, :, 3] > 0
        mask, img = get_mask_abo(
            scene_dir=scene_dir,
            scene_name=scene_name,
        )
        # Mask: non-black pixels
        if args.feature_save_name == '2d_patch':
            ys, xs = np.where(mask)
            num_pts = min(1300, len(xs))
            indices = np.random.choice(len(xs), size=num_pts, replace=False)
            pts = list(zip(xs[indices], ys[indices]))


            dense_pts = downsample_mask(mask, args.dense_point_stride)
            
            if args.source_point_stride > 0:
                pts = downsample_mask(mask, args.source_point_stride)


        elif args.feature_save_name == '2d_patch_3d':
            ns_transform, scale = parse_dataparser_transforms_json(dt_file)
            depth_map_path = os.path.join(depth_dir, f'{scene_name}_00.npy.gz')
            with gzip.open(depth_map_path, 'rb') as f:
                depth_map = np.load(f, allow_pickle=True)
                # convert to h w 
                #depth_map = np.transpose(depth_map, (1, 2, 0))  # Assuming depth_map is in (H, W, C) format
                depth_map = depth_map[:, :, 0]
            #print('depth_map shape:', depth_map.shape)
            dense_pts = sample_pixels_fixed_world_spacing(
                mask=mask,
                depth=depth_map,
                K=K,
                voxel_size=args.sample_voxel_size / scale
            )

            #print('num of dense points:', len(dense_pts))
            #print('hehehe')
            pts = dense_pts.copy()
        
        patch_features, is_visible = get_patch_features_from_2d(
            img, pts, model, preprocess_fn,
            patch_size=args.patch_size, batch_size=args.batch_size, device=args.device
        )

        out_dir_fusion = os.path.join(scene_dir, 'features')
        os.makedirs(out_dir_fusion, exist_ok=True)

        torch.save(
            torch.tensor(dense_pts),
            os.path.join(out_dir_fusion, f'dense_pts_{args.feature_save_name}{names_dict["dense_point_name"]}.pt')
        )

        with open(os.path.join(out_dir_fusion, 'voxel_size_%s.json' % args.feature_save_name), 'w') as f:
            json.dump({'voxel_size': args.feature_voxel_size}, f, indent=4)
        
        torch.save(torch.tensor(pts), os.path.join(out_dir_fusion, f'source_pts_{args.feature_save_name}{names_dict["source_point_name"]}.pt'))
        #return pts, patch_features, is_visible
        
    else:
        print('scene: %s, points: %d, scale: %.4f' % (scene_name, len(pts), scale))

        scene_name = os.path.basename(scene_dir)

        with torch.no_grad():
            try:
                patch_features, is_visible, visible_pts_3d, visible_pts_2d, pts_2d = get_patch_features(pts, imgs, depths, w2cs, K, 
                                                                model, preprocess_fn, 
                                                                occ_thr, patch_size=args.patch_size, batch_size=args.batch_size, 
                                                                device=args.device, scene_name=scene_name, masks=masks)
            except Exception as e:
                print(f"Error for scene {scene_name}: {e}")
                print("Skipping this scene.")
                return None, None, None
                
            patch_features_view0 = patch_features[0]
        # Saving
        out_dir_fusion = os.path.join(scene_dir, 'features')
        os.makedirs(out_dir_fusion, exist_ok=True)
        if not 'physx' in args.feature_save_name:
            np.savez(os.path.join(out_dir_fusion, f"visible_points_view0_{args.feature_save_name}.npz"),
                    pts_3d=np.stack(visible_pts_3d),
                    pts_2d=np.stack(visible_pts_2d))
        #torch.save(patch_features[0], os.path.join(out_dir_fusion, f'patch_features_view0_{args.feature_save_name}.pt'))
        
        with open(os.path.join(out_dir_fusion, 'voxel_size_%s.json' % args.feature_save_name), 'w') as f:
            json.dump({'voxel_size': args.feature_voxel_size}, f, indent=4)

        torch.save(torch.tensor(pts_2d), os.path.join(out_dir_fusion, f'source_pts_{args.feature_save_name}{names_dict["source_point_name"]}.pt'))
        #print('source point path', os.path.join(out_dir_fusion, f'source_pts_{args.feature_save_name}{names_dict["source_point_name"]}.pt'))
        
    torch.save(patch_features, os.path.join(out_dir_fusion, f'patch_features_{args.feature_save_name}{names_dict["patch_feature_name"]}.pt'))
    torch.save(is_visible, os.path.join(out_dir_fusion, f'is_visible_{args.feature_save_name}{names_dict["is_visible_name"]}.pt'))
    #print('is visible path', os.path.join(out_dir_fusion, f'is_visible_{args.feature_save_name}{names_dict["is_visible_name"]}.pt'))
    
    return pts, patch_features, is_visible
    
    
if __name__ == '__main__':   

    args = get_args()

    scenes_dir = os.path.join(args.data_dir, 'scenes')
    scenes = get_scenes_list(args)

    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_BACKBONE, pretrained=CLIP_CHECKPOINT)
    model.to(args.device)

    for j, scene in enumerate(scenes):
        pts, patch_features, is_visible = process_scene(args, os.path.join(scenes_dir, scene), model, preprocess)