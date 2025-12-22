
import os, glob, sys
sys.path.insert(0, os.getcwd())

import json
import torch
import open_clip
import numpy as np
import open3d as o3d
import matplotlib as mpl
import gzip
from sklearn.decomposition import PCA

from feature_fusion import CLIP_BACKBONE, CLIP_CHECKPOINT
from gpt_inference import parse_material_list, parse_material_hardness
from carving import get_carved_pts
from utils import *
from utils import _infer_device, _to_device, _to_numpy
from plot.plot_utils import plot_segmaps, plot_np_array
from predict_utils import count_bad_cases
from utils_dataset.ho3d_utils import load_ho3d_depth_as_meters

try:
    from arguments import get_args
except:
    from Nerf2Physic_Generalization.arguments import get_args
from external.sam2.utils_sam import masks_related_names
#from external.PUGS.predict_property import material_seg_evaluation, evaluate_material_segmentation

from tqdm import tqdm
import subprocess
import copy

from models.mlp_contrastive import MLPContrastive, MaskAwareMLPContrastive

from typing import List

from tools.material_seg.octopi_model import get_octopi_iou_per_class, evaluate_other_prop_octopi

from utils_pixel_level import other_prop_error
from utils_point_cloud import volume_reconstructed

def get_result_query(
    args, scene_dir, clip_model, clip_tokenizer
):
    t_file = os.path.join(scene_dir, 'transforms.json')
    pcd_file = os.path.join(scene_dir, 'ns', 'point_cloud.ply')
    dt_file = os.path.join(scene_dir, 'ns', 'dataparser_transforms.json')

    _, result, _, _= predict_physical_property_integral(
        args, scene_dir, clip_model, clip_tokenizer, preprocess=None
    )

    query_pts = result['dense_pts']
    query_pts = torch.Tensor(query_pts).to(args.device)
    return result, query_pts, t_file, pcd_file

def evaluate_material_segmentation(
    segmentation_input_dict,
    clip_model,
    clip_tokenizer,
    args,
    scene_name=None,
    scene_dir=None,
    plotting_related_dict=None,
    iou_per_class_all=None,
    out_dir=None,
    model = 'our'
):
    """
    Evaluate material segmentation and optionally generate debug plots.
    
    Parameters:
    -----------
    segmentation_input_dict : dict
        Dictionary containing segmentation input data
    clip_model : model
        CLIP model for evaluation
    clip_tokenizer : tokenizer
        CLIP tokenizer
    args : argparse.Namespace
        Arguments containing configuration settings
    scene_name : str, optional
        Name of the scene being processed
    scene_dir : str, optional
        Directory path of the scene
    plotting_related_dict : dict, optional
        Dictionary containing data for plotting (required if args.debug_plot is True)
    iou_per_class_all : list, optional
        List to append IoU per class results to
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'mIoU': Mean Intersection over Union
        - 'segmap_pred': Predicted segmentation map
        - 'segmap_gt': Ground truth segmentation map
        - 'iou_per_class': IoU values per class
        - 'iou_per_class_dict': Dictionary of IoU per class
        - 'out_path': Path to debug plot (if generated)
    """
    
    # Perform material segmentation evaluation
    segmentation_evaluation = material_seg_evaluation(
        segmentation_input_dict, 
        clip_model, 
        clip_tokenizer, 
        args,
        model=model
    )
    
    # Extract evaluation metrics
    miou = segmentation_evaluation['mIoU']
    segmap_pred = segmentation_evaluation['segmap_pred']
    segmap_gt = segmentation_evaluation['segmap_gt']
    iou_per_class = segmentation_evaluation['iou_per_class']
    
    # # Determine output directory
    # if scene_name:
    #     out_dir = os.path.join('viz', scene_name)
    # else:
    #     out_dir = segmentation_input_dict.get('out_dir', 'viz/default')
    
    # Append IoU per class if list is provided
    if iou_per_class_all is not None:
        iou_per_class_all.append(iou_per_class)
    
    # Generate debug plots if requested
    out_path = None
    if args.debug_plot:
        if plotting_related_dict is None:
            raise ValueError("plotting_related_dict is required when debug_plot is True")
        
        if scene_dir is None or scene_name is None:
            raise ValueError("scene_dir and scene_name are required when debug_plot is True")
        
        
        out_path = plot_segmaps(
            segmap_pred=segmap_pred,
            segmap_gt=segmap_gt,
            orig_image_path=os.path.join(scene_dir, 'images', f'{scene_name}_00.png'),
            supercat_to_idx = read_json(args.supercat_to_idx_path),
            out_dir=out_dir,
            iou_per_class_dict=segmentation_evaluation['iou_per_class_dict'],
            conversion_to_supercategories=read_json(args.material_to_supercat_path),
            mat_names=segmentation_input_dict['mat_names'],
            args=args
        )
        # Log the output path if log file is specified
        if hasattr(args, 'log_file') and args.log_file:
            log_paths(args.log_file, [out_path, f'orig image path {os.path.join(scene_dir, "images", f"{scene_name}_00.png")}'])
    
    # Return results dictionary
    results = {
        'mIoU': miou,
        'segmap_pred': segmap_pred,
        'segmap_gt': segmap_gt,
        'iou_per_class': iou_per_class,
        'iou_per_class_dict': segmentation_evaluation.get('iou_per_class_dict'),
        'out_path': out_path
    }
    
    return results
    
def material_seg_evaluation(segmentation_input_dict, clip_model, clip_tokenizer, args, model='our'):
    # pull what we need (keep scene_dir!)
    scene_dir  = segmentation_input_dict['scene_dir']
    scene_name = segmentation_input_dict['scene_name']

    source_pts = segmentation_input_dict['source_pts']

    if 'abo' in args.data_dir.lower() or 'physx' in args.data_dir.lower() or 'ho3d' in args.data_dir.lower():
        w2cs       = segmentation_input_dict['w2cs']
        K          = segmentation_input_dict['K']

    # choose a single target device and move ONLY tensors there
    target_device = _infer_device(args, clip_model, segmentation_input_dict)
    segm_dict_dev = _to_device(segmentation_input_dict, target_device)

    # use moved tensors from the device-synced dict
    source_pts_dev = segm_dict_dev['source_pts']  # tensor on target_device
    if 'abo' in args.data_dir.lower() or 'physx' in args.data_dir.lower():
        w2cs_dev       = segm_dict_dev['w2cs']        # could be list[tensor] or numpy
        K_dev          = segm_dict_dev['K']

        if 'abo' in args.data_dir:
            # load sim map
            # downstream utils often expect numpy for projection; convert just for this call
            sim_coco = load_sim_blip2(scene_dir, args, mode='coco')

            source_pts_np = project_3d_to_2d(
                source_pts_dev.detach().cpu().numpy(),
                w2c=_to_numpy(w2cs_dev[0]),  # first view
                K=_to_numpy(K_dev),
            )

            mask_ids_gt_check = build_mask_ids(
                source_pts_np,
                mask_dir=args.gt_json,
                one_mask=args.one_mask,
                mode='coco',
                scene_name=scene_name,
            )

            manu_gt_sim = expand_sim_map(sim_coco, mask_ids_gt_check, args.one_mask)
            manu_gt_sim=_to_device(manu_gt_sim, target_device) if isinstance(manu_gt_sim, (torch.Tensor, list, dict, tuple)) else manu_gt_sim,

    # Pack kwargs: start from the device-synced dict; add extras (also put on device if tensors)
    eval_kwargs = dict(
        segm_dict_dev,
        args=args,
        clip_model=clip_model,
        clip_tokenizer=clip_tokenizer,
        w2cs=w2cs_dev if 'abo' in args.data_dir.lower() or 'physx' in args.data_dir.lower() else None,
        K=K_dev if 'abo' in args.data_dir.lower() or 'physx' in args.data_dir.lower() else None,
        manu_gt_sim=manu_gt_sim if 'abo' in args.data_dir else None,
        model=model
    )

    if 'abo' in args.data_dir:
        eval_kwargs['manu_gt_sim'] = manu_gt_sim
    
 
    # Important: ensure anything used in torch ops inside segmentation_eval is now on the same device.
    return segmentation_eval(**eval_kwargs)


@torch.no_grad()
def get_interpolated_values(source_pts, source_vals, inner_pts, batch_size=2048, k=1):
    """Interpolate values by k nearest neighbor."""
    n_inner = len(inner_pts)
    inner_vals = torch.zeros(n_inner, source_vals.shape[1], device=inner_pts.device)
    for batch_start in range(0, n_inner, batch_size):
        curr_batch_size = min(batch_size, n_inner - batch_start)
        curr_inner_pts = inner_pts[batch_start:batch_start + curr_batch_size]

        dists = torch.cdist(curr_inner_pts, source_pts)  # (B, N)
        _, idxs = torch.topk(dists, k=k, dim=1, largest=False)
        curr_inner_vals = source_vals[idxs].mean(1)

        inner_vals[batch_start:batch_start + curr_batch_size] = curr_inner_vals
    return inner_vals

def features_to_colors(features):
    """Convert feature vectors to RGB colors using PCA."""
    pca = PCA(n_components=3)
    pca.fit(features)
    transformed = pca.transform(features)
    q1, q99 = np.percentile(transformed, [1, 99])
    feature_pca_postprocess_sub = q1
    feature_pca_postprocess_div = (q99 - q1)
    transformed = (transformed - feature_pca_postprocess_sub) / feature_pca_postprocess_div
    colors = np.clip(transformed, 0, 1)
    return colors

@torch.no_grad()
def get_text_features(texts, clip_model, clip_tokenizer, prefix='', suffix='', device='cuda'):
    """Get CLIP text features, optionally with a fixed prefix and suffix."""
    extended_texts = [prefix + text + suffix for text in texts]
    tokenized = clip_tokenizer(extended_texts).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = clip_model.encode_text(tokenized)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

    return text_features


@torch.no_grad()
def get_agg_patch_features(patch_features, is_visible, args=None):
    """Get aggregated patch features by averaging over visible patches.
    Params:
        patch_features: (N_img, N_pts, clip_output_size)
        is_visible: (N_imgs, N_pts)
    Returns:
        avg_visible_patch_features: (N_pts, clip_output_size)
        is_valid: (N_pts,) (is_valid[i] = True if i-th point is visible in at least one view)
        avg_visible_patch_features_first_view: (N1_pts, clip_output_size): the patch features of the points visible in the first view
        is_valid_first_view : (N_pts,) (is_valid_first_view[i] = True if i-th point is visible in the first view)

    """
    n_visible = is_visible.sum(0) # (N_pts,) (n_visible[i] = number of visible patches for i-th point)
    is_valid = n_visible > 0 # (N_pts,) (is_valid[i] = True if i-th point is visible in at least one view)

    visible_patch_features = patch_features * is_visible.unsqueeze(-1)
    if args.feature_load_name != '2d_patch':
        avg_visible_patch_features = visible_patch_features.sum(0) / n_visible.unsqueeze(-1)
    else:
        # Select only valid points before averaging
        valid_patch_features = visible_patch_features[:, is_valid, :]  # (N_views, N_valid_pts, D)
        avg_visible_patch_features = valid_patch_features.mean(0)   # (N_valid_pts, D)

    avg_visible_patch_features = avg_visible_patch_features / avg_visible_patch_features.norm(dim=1, keepdim=True) # (N_pts, clip_output_size)

    # First view
    first_view_mask = is_visible[0].bool()  # (N_pts,)

    if args.feature_load_name == 'ps56_multi_view_2d':
        is_valid = first_view_mask

    if not args.feature_load_name == '2d_patch':
        return avg_visible_patch_features[is_valid], is_valid
    else:
        return avg_visible_patch_features, is_valid
    
def process_segmentation_maps_mvimg(args, scene_name, segmap_pred, dense_pts):
    """
    Process segmentation maps by loading ground truth, masking with dense points,
    and padding to match shapes.
    
    Args:
        args: Arguments object containing configuration parameters
        scene_name: Name of the scene being processed
        segmap_pred: Predicted segmentation map
        dense_pts: Dense points tensor for masking
    
    Returns:
        tuple: (segmap_pred, segmap_gt) - processed segmentation maps
    """
    # Load ground truth segmentation map
    segmap_gt = load_mvimg_segmap_gt(args, scene_name)
    segmap_gt = mask_gt_with_dense_pts(
        segmap_gt, dense_pts.cpu().numpy() if type(dense_pts) is torch.Tensor else dense_pts
    )

    if args.verbose:
        print('shape before pad:', segmap_pred.shape, segmap_gt.shape)
        # plot segmap pred and gt before padding
        debug_save_path = os.path.join('viz', scene_name, f'{args.viz_save_name}', f'{scene_name}_segmap_before_pad.png')
        plot_np_array(segmap_pred, segmap_gt, save_path=debug_save_path)

    # Pad arrays to match shapes
    segmap_pred, segmap_gt = pad_to_match_shape(
        segmap_pred, segmap_gt, pad_value=-1)
    
    if args.verbose:
        print('shape after pad:', segmap_pred.shape, segmap_gt.shape)
        debug_save_path = os.path.join('viz', scene_name, f'{args.viz_save_name}', f'{scene_name}_segmap_after_pad.png')
        plot_np_array(segmap_pred, segmap_gt, save_path=debug_save_path)
    
    return segmap_pred, segmap_gt
    
def segmentation_eval(
    args, scene_dir, scene_name, clip_model, clip_tokenizer,
    dense_pred_probs, dense_pts, w2cs, K, mat_names,
    manu_gt_sim=None, source_pts=None, masks_dir=None, mat_names_gt=None, H_cap=None, W_cap=None, model='our', dense_pred_vals=None
):

    if mat_names_gt is None:
        mat_names_gt = mat_names
    
    if args.verbose:
        print('mat names', mat_names)
        print('mat names gt', mat_names_gt)
    #print('ahahahahhaha')
#if args.evaluate_segmentation == 'intensive':
    if 'abo' in args.data_dir.lower():
        dense_pred_probs_gt = get_interpolated_values(source_pts, manu_gt_sim, dense_pts, batch_size=2048, k=args.k)
    material_to_supercat = read_json(args.material_to_supercat_path)
    supercat_to_idx = read_json(args.supercat_to_idx_path)

    orig_img_path = os.path.join(scene_dir, 'images', f'{scene_name}_00.png')
    orig_img = mpl.image.imread(orig_img_path)
    H_cap, W_cap = orig_img.shape[:2]

    if args.feature_load_name == '2d_patch':

        segmap_pred = dense_prob_to_segmentation(
            dense_probs = dense_pred_probs.cpu().numpy() if type(dense_pred_probs) is torch.Tensor else dense_pred_probs,
            dense_pts = dense_pts.cpu().numpy() if type(dense_pts) is torch.Tensor else dense_pts,
            material_to_supercat = read_json(args.material_to_supercat_path),
            supercat_to_idx = read_json(args.supercat_to_idx_path),
            mat_names = mat_names,
            H_cap = H_cap,
            W_cap = W_cap
        )

        #print('segmap_pred shape:', np.unique(segmap_pred))
        
        if 'abo' in args.data_dir:
            segmap_gt = dense_prob_to_segmentation(
                dense_probs = dense_pred_probs_gt.cpu().numpy(),
                dense_pts = dense_pts.cpu().numpy(),
                material_to_supercat = material_to_supercat,
                supercat_to_idx = supercat_to_idx,
                mat_names = mat_names_gt
            )
        # mvimgnet
        elif 'mvimgnet' in args.data_dir.lower() or 'physx' in args.data_dir.lower() or 'ho3d' in args.data_dir.lower() or 'arctic' in args.data_dir.lower():
            segmap_pred, segmap_gt = process_segmentation_maps_mvimg(
                args=args,
                scene_name=scene_name,
                segmap_pred=segmap_pred,
                dense_pts=dense_pts
            )
    elif args.feature_load_name == 'default':
        # result, query_pts, t_file, pcd_file = get_result_query(
        #     args, scene_dir, clip_model, clip_tokenizer
        # )

        #print('hahahahah')
        imgs = load_images(os.path.join(scene_dir, 'images'))
        orig_img = imgs[0] / 255.

        w2c = w2cs[0]  # Use the first view's camera
        if 'abo' in args.data_dir.lower():
            w2c[[1,2]] *= -1 # convert from nerfstudio to open3d format
        # render = material_seg_render(
        #     result=result,
        #     query_pts=query_pts,
        #     orig_img=orig_img,
        #     out_dir = out_dir,
        #     w2c=w2c,  # Use the first view's camera
        #     K=K,
        #     args=args
        # )
        dense_pts_2d = project_3d_to_2d(
            dense_pts.cpu().numpy() if type(dense_pts) is torch.Tensor else dense_pts,
            w2cs[0],  # Use the first view's camera
            K
        )

        if args.verbose:
            print('dense pts 2d', dense_pts_2d.shape)

        #print('hahahahah')
        segmap_pred = dense_prob_to_segmentation(
            dense_probs = dense_pred_probs.cpu().numpy() if type(dense_pred_probs) is torch.Tensor else dense_pred_probs,
            dense_pts = dense_pts_2d,
            material_to_supercat=read_json(args.material_to_supercat_path),
            supercat_to_idx=read_json(args.supercat_to_idx_path),
            mat_names= mat_names,
            H_cap = H_cap,
            W_cap = W_cap
        )

        if 'abo' in args.data_dir.lower():
            segmap_gt = dense_prob_to_segmentation(
                dense_probs = dense_pred_probs_gt.cpu().numpy(),
                dense_pts = dense_pts_2d,
                material_to_supercat=material_to_supercat,
                supercat_to_idx=supercat_to_idx,
                mat_names= mat_names_gt,
                H_cap = H_cap,
                W_cap = W_cap
            )
        elif 'mvimgnet' in args.data_dir.lower() or 'physx' in args.data_dir.lower() or 'ho3d' in args.data_dir.lower() or 'arctic' in args.data_dir.lower():
            segmap_pred, segmap_gt = process_segmentation_maps_mvimg(
                args=args,
                scene_name=scene_name,
                segmap_pred=segmap_pred,
                dense_pts=torch.Tensor(dense_pts_2d)
            )
        # create segmap_pred

    miou, iou_per_class, wrong_gt_check_bool = scene_iou(
        pred_map=segmap_pred,
        gt_map=segmap_gt,
        supercat_to_idx=supercat_to_idx,
        masks_dir = masks_dir
    )

    iou_per_class_dict = {}
    for i, supercat in enumerate(supercat_to_idx):
        iou_per_class_dict[supercat] = iou_per_class[i]
    
    #print('heheheheh')
    return {
        'mIoU': miou,
        'iou_per_class_dict': iou_per_class_dict,
        'wrong_gt_check_bool': wrong_gt_check_bool,
        'segmap_pred': segmap_pred,
        'segmap_gt': segmap_gt,
        'iou_per_class': iou_per_class,
        'dense_pts_2d': dense_pts_2d if args.feature_load_name == 'default' else dense_pts,
        'dense_pred_probs_gt': dense_pred_probs_gt if 'abo' in args.data_dir else None,
        'manu_gt_sim': manu_gt_sim if 'abo' in args.data_dir else None,
    }



def run_feature_fusion(scene_dir, args):
    print(f"⚙️ Auto-running feature fusion for scene: {scene_dir} with feature save name: {args.feature_save_name} patch size: {args.patch_size} source point stride: {args.source_point_stride} dense point stride: {args.dense_point_stride}")

    cmd = [
        "python", "feature_fusion.py",
        "--split", args.split,
        "--start_idx", str(args.start_idx),
        "--end_idx", str(args.end_idx),
        "--feature_save_name", args.feature_save_name,
        "--feature_load_name", args.feature_load_name,
        "--source_point_stride", str(args.source_point_stride),
        "--dense_point_stride", str(args.dense_point_stride),
        "--patch_size", str(args.patch_size),
        "--device", args.device
    ]
    subprocess.run(cmd, check=True)

    #subprocess.run(cmd, check=True)

def check_required_feature_files(scene_dir, args, names_dict):
    feature_dir = os.path.join(scene_dir, 'features')
    required_files = [
        os.path.join(feature_dir, f'source_pts_{args.feature_load_name}{names_dict["source_point_name"]}.pt'),
        os.path.join(feature_dir, f'patch_features_{args.feature_save_name}{names_dict["patch_feature_name"]}.pt'),
        os.path.join(feature_dir, f'is_visible_{args.feature_load_name}{names_dict["is_visible_name"]}.pt')
    ]

    #print('required files:', required_files)
    #print('required files:', required_files)
    return all(os.path.exists(f) for f in required_files)

def apply_sam_prior_refinement(
    logits: torch.Tensor,               # (N, K)
    source_coords: np.ndarray,          # (N, 2) int, in (x, y) format
    masks: List[np.ndarray],            # each (H, W) binary numpy mask
    mask_lambda: float,                 # nudging strength
) -> torch.Tensor:
    """
    Apply SAM-based prior refinement to logits using mask-wise material priors.

    Returns:
        refined_probs: torch.Tensor of shape (N, K)
    """
    print('[🧠] Applying SAM-based mask prior refinement...')

    N, K = logits.shape
    #H, W = image_size

    mask_to_indices = {}
    index_to_mask = np.full(len(source_coords), -1)
    
    for mask_idx, mask in enumerate(masks):
        #assert mask.shape == (H, W)
        inside = []
        for i, (x, y) in enumerate(source_coords):
            if mask[y, x]:
                inside.append(i)
        if inside:
            mask_to_indices[mask_idx] = inside
            for i in inside:
                index_to_mask[i] = mask_idx

    # source_pred_probs = torch.softmax(logits, dim=1)
    source_pred_probs = logits
    mask_material_prior = {
        mask_idx: source_pred_probs[indices].mean(0)
        for mask_idx, indices in mask_to_indices.items()
    }

    biased_logits = logits.clone()
    for mask_idx, indices in mask_to_indices.items():
        # prior_log = torch.log(mask_material_prior[mask_idx] + 1e-6)
        # biased_logits[indices] += mask_lambda * prior_log
        biased_logits[indices] += mask_lambda * mask_material_prior[mask_idx]
    return torch.softmax(biased_logits, dim=1)

def material_seg_render(
    result, query_pts, orig_img, out_dir, w2c, K, args,
):
    seg_pcd = o3d.geometry.PointCloud()
    seg_pcd.points = o3d.utility.Vector3dVector(query_pts.cpu().numpy())
    #colors_seg = similarities_to_colors(result['query_pred_probs'])
    colors_seg = similarities_to_colors(result['dense_pred_probs'])
    seg_pcd.colors = o3d.utility.Vector3dVector(colors_seg)
    h, w = orig_img.shape[:2]

    render = render_pcd(seg_pcd, w2c, K, hw=(h,w), show=args.show)
    if args.show != 0:
        combined = composite_and_save(orig_img, render, args.compositing_alpha,
            savefile=os.path.join(out_dir, '_seg.png' ))
        print(f'saved to {os.path.join(out_dir, "_seg.png")}')
    return render

@torch.no_grad()
def predict_physical_property_integral(args, scene_dir, clip_model, clip_tokenizer, preprocess=None, heavy_close_split=None, debugging=False):
    """Predict the volume integral of a physical property (e.g. for mass). Returns a [low, high] range."""
    names_dict = name_creation(args)
    
    out_dir = os.path.join(scene_dir, 'features')
    
    scene_name = os.path.basename(scene_dir)

    pcd_file, dt_file, info_file = get_file_paths(args, scene_dir, scene_name, heavy_close_split)
    if debugging:
        print('info file path', info_file)
        print('info dim path', os.path.join(scene_dir, 'info_dim.json'))


    info = read_json(info_file)

    feature_dir = os.path.join(scene_dir, 'features')

    start_idx, end_idx = get_start_end_idx(args, scene_dir)
    run_args = copy.deepcopy(args)
    run_args.start_idx = start_idx
    run_args.end_idx = end_idx

    #print('check required feature files:', check_required_feature_files(scene_dir, run_args, names_dict)) 
    if not check_required_feature_files(scene_dir, run_args, names_dict) or (args.fuse_overwrite and args.feature_load_name == '2d_patch'):
        if args.check_required:
            print(f'scene {scene_name}, missing required feature files')
            return None
        run_feature_fusion(scene_dir, run_args)

    #print('heheheheheh')
    # loading source point info
    with open(os.path.join(scene_dir, 'features', 'voxel_size_%s.json' % args.feature_load_name), 'r') as f:
        feature_voxel_size = json.load(f)['voxel_size']
        

    source_pts = load_source_pts(
        args=args,
        pcd_file=pcd_file,
        dt_file=dt_file,
        feature_dir=feature_dir,
        feature_voxel_size=feature_voxel_size,
        names_dict=names_dict
    )
        #print('source pts shape:', source_pts.shape)
    good_volume_scene = False

    pts_3d = None
    if not args.viz_only and 'abo' in args.data_dir:
        pts_3d = load_ns_point_cloud(pcd_file, dt_file, ds_size=None)
    if (args.use_3d_volume) and args.feature_load_name == '2d_patch':
        if args.use_3d_volume:
            if args.log_volume:
                if args.viz_save_name == 'Hoppe':
                    obj_volume, geo, mode = compute_volume_from_point_cloud(pts_3d)
                elif args.viz_save_name == 'hull':
                    obj_volume, geo, mode = compute_volume_from_point_cloud_hull(pts_3d)
                    if mode == 'mesh':
                        good_volume_scene = True
                        print('scene', info["caption"], 'has good volume')

                np.save(os.path.join(scene_dir, f'obj_volume_{args.viz_save_name}.npy'), obj_volume)
                print(f'saved volume to {os.path.join(scene_dir, f"obj_volume_{args.viz_save_name}.npy")}')
            else:
                obj_volume = np.load(os.path.join(scene_dir, f'obj_volume_{args.viz_save_name}.npy'))

    patch_features = torch.load(os.path.join(out_dir, f'patch_features_{args.feature_load_name}{names_dict["patch_feature_name"]}.pt')).to(args.device)  # (N, 512)
    is_visible = torch.load(os.path.join(scene_dir, 'features', f'is_visible_{args.feature_load_name}{names_dict["is_visible_name"]}.pt')).to(args.device)

    if args.verbose:
        print('patch feature path:', os.path.join(out_dir, f'patch_features_{args.feature_load_name}{names_dict["patch_feature_name"]}.pt'))
        print('is visible path:', os.path.join(scene_dir, 'features', f'is_visible_{args.feature_load_name}{names_dict["is_visible_name"]}.pt'))
        print('patch features shape:', patch_features.shape)
        print('is visible shape:', is_visible.shape)
        
    # preparing material info
    mat_val_list = info['candidate_materials_%s' % args.property_name]
    #print('material val list:', mat_val_list)
    mat_names, mat_vals = parse_material_list(mat_val_list)
    mat_vals = torch.tensor(mat_vals, dtype=torch.float32, device=args.device)

    if 'abo' not in args.data_dir:
        mat_names_gt = mat_vals.clone()

    if args.property_name == 'density':
        mat_tn_list = info['thickness']
        mat_names, mat_tns = parse_material_list(mat_tn_list)
        mat_tns = torch.Tensor(mat_tns).to(args.device) / 100  # cm to m
    
    # print(f"[Debug] Raw source_pts shape: {source_pts.shape}")
    # print(f"[Debug] Raw patch_features shape: {patch_features.shape}")

    if args.random_var == 'density_hard':
        #randomize mat_vals
        # Randomize material density values within a specified range
        low_density = 100   # Lower bound (100 kg/m³)
        high_density = 8000.0  # Upper bound (2000 kg/m³)
        rand_vals = torch.rand_like(mat_vals)  # Values between 0 and 1
        mat_vals = low_density + (high_density - low_density) * rand_vals  # Scale to desired range

    elif args.random_var == 'thickness':
        # Randomize material thickness values within a specified range
        low_thickness = 0.001
        high_thickness = 0.5
        rand_vals = torch.rand_like(mat_tns)  # Values between 0 and 1
        mat_tns = low_thickness + (high_thickness - low_thickness) * rand_vals  # Scale to desired range

    # predictions on source points
    text_features = get_text_features(mat_names, clip_model, clip_tokenizer, device=args.device)
    #if not '2d' in args.feature_load_name:
    agg_patch_features, is_valid = get_agg_patch_features(patch_features, is_visible, args)
    agg_patch_features = agg_patch_features.to(text_features.device)

    is_valid = is_valid.to(text_features.device)
    source_pts = source_pts.to(text_features.device)
    source_pts = source_pts[is_valid]

    mask_paths_dict = masks_related_names(scene_dir, args)

    masks_dir = mask_paths_dict["masks_dir"]
    jsons_dir = mask_paths_dict["jsons_dir"]
    mask_name = mask_paths_dict["mask_name"]

    scene_mask_dir = masks_dir

    if args.volume_method == 'thickness_physx_volume':
        physx_vol_dict = read_json(args.volume_path)
    
    if args.nerf_vs_physx_scale != 'None':
        nerf_vs_physx_scale_dict = read_json(args.nerf_vs_physx_scale)

    #if args.feature_load_name == 'default':
    if (args.feature_load_name == 'default') or (not args.viz_only) and ('abo' in args.data_dir.lower() or 'physx' in args.data_dir.lower() or ('ho3d' in args.data_dir.lower() and args.dense_point_stride == 0)):
        w2cs, K = parse_transforms_json(os.path.join(scene_dir, 'transforms.json'), return_w2c=True)
        c2ws, K = parse_transforms_json(os.path.join(scene_dir, 'transforms.json'), return_w2c=False)
        c2w = c2ws[0]  # Use the first view's camera
    if args.plot_cuboids_mode != 'None':
        c2ws, K = parse_transforms_json(os.path.join(scene_dir, 'transforms.json'), return_w2c=False)
        c2w = c2ws[0]  # Use the first view's camera
        if args.feature_load_name == '2d_patch':
            # depth_map_path = os.path.join(scene_dir, 'ns', 'renders', 'depth', f'{scene_name}_00.npy.gz')
            # with gzip.open(depth_map_path, 'rb') as f:
            #     depth_map = np.load(f, allow_pickle=True)
            depth_dir = os.path.join(scene_dir, 'ns', 'renders', 'depth')
            depth_map = load_depths(depth_dir, Ks)
            #data/abo_500/scenes/B00DUGZFLK_ATVPDKIKX0DER/ns/renders/depth
    source_pts_to_use = source_pts.clone()

    if args.feature_load_name == '2d_patch':
        mask_ids = build_mask_ids(source_pts.cpu().numpy().astype(np.int32), scene_mask_dir)
        source_pts_2d = source_pts.cpu().numpy().astype(np.int32)
    elif args.feature_load_name == 'default':
        #if args.log_source_pts:
        source_pts_2d = project_3d_to_2d(
            source_pts.cpu().numpy(),
            w2c=w2cs[0],  # No camera transform needed for default features
            K=K    # No intrinsic matrix needed for default features
        )
        source_pts_2d_path = os.path.join(feature_dir, f'source_pts_{args.feature_load_name}{names_dict["source_point_name"]}_prj2d.pt' )
        torch.save(torch.tensor(source_pts_2d), source_pts_2d_path)

        if args.evaluate_segmentation == 'intensive':
            source_pts_to_use = torch.Tensor(source_pts_2d).to(args.device).float()

    if args.evaluate_segmentation == 'intensive' and 'abo' in args.data_dir:
        # create GT sim map
        sim_coco = load_sim_blip2(scene_dir, args, mode= 'coco')

        scene_name = os.path.basename(scene_dir)
        source_pts_np = source_pts_to_use.cpu().numpy().astype(np.int32)  # (N, 3)
        mask_ids_gt_check = build_mask_ids(source_pts_np, mask_dir=args.gt_json, one_mask=args.one_mask, mode='coco', scene_name=scene_name)
        
        manu_gt_sim = expand_sim_map(sim_coco, mask_ids_gt_check, args.one_mask)
        manu_gt_sim = manu_gt_sim.to(args.device)

    if args.mlp_checkpoint != "None" and args.testing:
        # from models.mlp_contrastive import *
        print(f"🤖 Using MLP model with checkpoint: {args.mlp_checkpoint}")
        clip_feat_dim = agg_patch_features.shape[1]  # D
        if args.model == 'mlp':
            model = MLPContrastive(clip_feat_dim).to(args.device)
        elif args.model == 'mlp_attention':
            model = MaskAwareMLPContrastive(clip_feat_dim).to(args.device)
        
        model.load_state_dict(torch.load(args.mlp_checkpoint, map_location=args.device))
        model.eval()

        with torch.no_grad():
            if args.model == 'mlp':
                similarities = model(agg_patch_features, text_features)  # (N, K)
            elif args.model == 'mlp_attention':
                #print('hehehehehhehehe')
                similarities = model(agg_patch_features, text_features, mask_ids)
                attention_output = model.get_mask_aggregator_output(agg_patch_features, mask_ids)  # Get attention output if needed

        model_info = get_model_related_info(args)

        if 'abo' in args.data_dir:
            # this part is for soft segmentaiton evaluation only
            if model_info['alternate_gt'] == 'dot_product':
                gt_sim = agg_patch_features @ text_features.T
            elif model_info['alternate_gt'] == 'our22':
                gt_sim = our22_prep(
                    scene_dir=scene_dir,
                    text_features=text_features,
                    clip_model=clip_model,
                    preprocess=preprocess,
                    masks_dir=masks_dir,
                    jsons_dir=jsons_dir,
                    mask_ids=mask_ids,
                    mat_names=mat_names,
                    args=args
                )
            elif model_info['alternate_gt'] == 'blip2':
                #print('ehhehe')
                sim_blip2 = load_sim_blip2(scene_dir, args)

                gt_sim = expand_sim_map(
                    sim_blip2, mask_ids, args.one_mask
                )
    else:
        similarities = agg_patch_features @ text_features.T
            
    logits = similarities / args.temperature # (N, K)

    if args.model == 'None' and args.feature_load_name == '2d_patch':
        if args.use_sam and args.mask_prior_lambda is not None:
            mask_paths = sorted(glob.glob(os.path.join(scene_mask_dir, f"mask_*.npy")))
            if not mask_paths:
                print(f"[⚠️] No SAM masks found in {scene_mask_dir}, skipping refinement.")
                source_pred_probs = torch.softmax(logits, dim=1)
            else:
                # Load all masks into a list
                masks = [np.load(p).astype(bool) for p in mask_paths]
                source_coords = source_pts.cpu().numpy().astype(np.int32)  # (N, 2), in (x, y)

                # Apply refinement
                source_pred_probs = apply_sam_prior_refinement(
                    logits=logits,
                    source_coords=source_coords,
                    masks=masks,
                    mask_lambda=args.mask_prior_lambda,
                )

        # mass evaluation with manual material segmentation
        elif args.materials_existed_name != 'None':
            #gt_sim_vec_path = os.path.join(scene_dir, f'{args.materials_existed_name}_gt_sim_vecs.json')
            sim_coco = load_sim_blip2(scene_dir, args, mode= 'coco')

            scene_name = os.path.basename(scene_dir)
            source_pts_np = source_pts.cpu().numpy().astype(np.int32)  # (N, 2)
            mask_ids = build_mask_ids(source_pts_np, mask_dir=args.gt_json, one_mask=args.one_mask, mode='coco', scene_name=scene_name)
            
            source_pred_probs = expand_sim_map(sim_coco, mask_ids, args.one_mask)
            source_pred_probs = source_pred_probs.to(args.device)

        elif args.sim_map_source == 'our22':
            # load (M, K) logits from CLIP mask
            sim_dot = our22_prep(
                scene_dir = scene_dir,
                text_features = text_features,
                clip_model = clip_model,
                preprocess = preprocess,
                masks_dir = masks_dir,
                jsons_dir = jsons_dir,
                mask_ids = mask_ids,
                mat_names = mat_names,
                args=args
            )

            source_pred_probs = torch.softmax(sim_dot, dim=1)
            #print('eheheheheheh')
        elif args.sim_map_source == 'blip2':
            # load (M, K) logits from CLIP mask
            sim_blip2 = load_sim_blip2(scene_dir, args)
            source_pred_probs = expand_sim_map(sim_blip2, mask_ids, args.one_mask)
            source_pred_probs = source_pred_probs.to(args.device)
            
        else:
            source_pred_probs = torch.softmax(logits, dim=1)

    else:
        source_pred_probs = torch.softmax(logits, dim=1)

    if args.randomize_prob:
        # Create a random tensor with the same shape as source_pred_probs
        random_tensor = torch.rand(source_pred_probs.shape, device=source_pred_probs.device)
        # Normalize each row to sum to 1 (to maintain probability distribution)
        random_tensor = random_tensor / random_tensor.sum(dim=1, keepdim=True)
        source_pred_probs = random_tensor
        #print('eheh')

    source_pred_mat_idxs = similarities.argmax(1)

    if getattr(args, "top_material", False):
        # Use only the top-1 material's value
        top_idxs = source_pred_probs.argmax(dim=1)  # (N,)
        source_pred_vals = mat_vals[top_idxs]       # (N, 2) if mat_vals is Nx2
    else:
        # Use expected value over all materials
        source_pred_vals = source_pred_probs @ mat_vals

    # volume integration
    if args.feature_load_name == 'default' or 'abo' in args.data_dir:
        #print('heheheheh') 
        ns_transform, scale = parse_dataparser_transforms_json(dt_file)
    # else:
    #     scale = 1 # not an idealy accurate choice, still under development
        #print('heheheheheheh')
    
    #if not args.viz_only:
    if args.evaluate_segmentation == 'None' and not args.viz_only:
        if args.surface_cell_size == 0:
            surface_cell_size = args.sample_voxel_size / scale #unit: m
        else:
            surface_cell_size = args.surface_cell_size
        
        mat_cell_volumes = surface_cell_size**2 * mat_tns #shape (K, 2)
        mat_cell_products = mat_vals * mat_cell_volumes # shape (K, 2)

    if 'thickness' in args.volume_method:
        if not '2d' in args.feature_load_name:
            if not args.viz_only:
                if not 'physx' in args.feature_load_name:
                    if args.evaluate_segmentation == 'None':
                        dense_pts = load_ns_point_cloud(pcd_file, dt_file, ds_size=args.sample_voxel_size)
                    else:
                        dense_pts = load_ns_point_cloud(pcd_file, dt_file, ds_size=None)
                else:
                    pcd_physx_file = os.path.join(scene_dir, 'physx', 'point_cloud.ply')
                    dense_pts = load_ply_point_cloud(pcd_physx_file, ds_size=surface_cell_size)
            else:
                if not 'physx' in args.feature_load_name:
                    dense_pts = load_ns_point_cloud(pcd_file, dt_file, ds_size=None)
                else:
                    pcd_physx_file = os.path.join(scene_dir, 'physx', 'point_cloud.ply')
                    dense_pts = load_ply_point_cloud(pcd_physx_file, ds_size=surface_cell_size)

            dense_pts = torch.Tensor(dense_pts).to(args.device)
        else:
            mask, _ = get_mask_abo(
                scene_dir=scene_dir,
                scene_name=scene_name,
            )

            imgs, alphas = load_images(os.path.join(scene_dir, 'images'), return_masks=True)
            alpha = alphas[0]  # Use the first view's alpha mask

            if args.evaluate_segmentation == 'None':
                if args.dense_point_stride == 0 and not args.viz_only:
                    #print('hehehehehheheh')
                    depth_map = None

                    z_mean_real = None
                    if args.depth_method == 'depth_map':
                        if not 'abo' in args.data_dir:
                            depth_map_path = os.path.join(scene_dir, "depth_pred", f'{scene_name}_00.pt')
                            #print('depth map path', depth_map_path)
                            depth_map = torch.load(depth_map_path)
                            depth_map = depth_map.cpu().numpy()
                            print(f"[Debug] Loaded depth map from: {depth_map_path}")
                    elif args.depth_method == 'z_mean':
                        if 'sam3d' in args.volume_method:
                            gs_model = 'sam3d_obj'
                        else:
                            gs_model = 'imsplat'
                        z_mean_real = compute_z_from_gs(scene_dir, gs_model)
                        #print('ehehehehhehe')
                    pixel_stride = compute_dense_stride(
                        pts = pts_3d,
                        scene_dir = scene_dir,
                        K = K,
                        c2w = c2w,
                        surface_cell_size = surface_cell_size,
                        depth_map = depth_map,
                        z_mean_real = z_mean_real,
                        depth_mask = alpha
                    )
                    pixel_stride /= 2
                    
                elif not args.viz_only:
                    pixel_stride = args.dense_point_stride
                
                else:
                    pixel_stride = 1
                print(f"[Debug] Using pixel stride: {pixel_stride}")
                # Downsample the mask
                if args.feature_load_name == '2d_patch':
                    dense_pts = downsample_mask(mask, pixel_stride)
                    dense_pts = torch.Tensor(dense_pts).to(args.device).float()
                elif args.feature_load_name == '2d_patch_3d':
                    #dense_pts_path = os.path.join(out_dir, f'dense_pts_{args.feature_save_name}{names_dict["dense_point_name"]}.pt')
                    dense_pts_path = get_dense_pts_path(
                        scene_dir = scene_dir,
                        args = args,
                        #names_dict = names_dict
                    )
                    dense_pts = torch.load(dense_pts_path).to(args.device).float()
                    #print('dense_pts shape:', dense_pts.shape)
            else:
                if args.feature_load_name == '2d_patch':
                    dense_pts = downsample_mask(mask, 0)
                    dense_pts = torch.Tensor(dense_pts).to(args.device).float()
                elif args.feature_load_name == '2d_patch_3d':
                    #dense_pts_path = os.path.join(out_dir, f'dense_pts_{args.feature_save_name}{names_dict["dense_point_name"]}.pt')
                    dense_pts_path = get_dense_pts_path(
                        scene_dir = scene_dir,
                        args = args,
                        #names_dict = names_dict
                    )
                    dense_pts = torch.load(dense_pts_path).to(args.device).float()

            if args.plot_cuboids_mode != "None":
                dense_pts_3d = lift_2d_to_3d(dense_pts, depth_map, K, c2w)
                dense_pts_3d = torch.Tensor(dense_pts_3d).to(args.device).float()

        if args.use_3d_volume:
            if args.feature_load_name == '2d_patch' and not args.viz_only:
                estimated_voxel_volume = obj_volume/len(dense_pts)  # Placeholder for actual volume estimation logic
                mat_cell_volumes = torch.full_like(mat_vals, estimated_voxel_volume)
                mat_cell_products = mat_vals * mat_cell_volumes
    
        print("source_pts:", source_pts.shape)
        print("dense_pts:", dense_pts.shape)

        dense_pred_probs = get_interpolated_values(source_pts, source_pred_probs, dense_pts, batch_size=2048, k=args.k)
        
        
        if args.evaluate_segmentation == 'None' and not args.viz_only:
            mat_vol_all = dense_pred_probs @ mat_cell_volumes #shape (N, 2)
        
        if args.evaluate_segmentation == 'None' and not args.viz_only:
            if getattr(args, "top_material", False):
                dense_top_idxs = dense_pred_probs.argmax(dim=1)
                dense_pred_products = mat_cell_products[dense_top_idxs]
            else:
                dense_pred_products = dense_pred_probs @ mat_cell_products

            if args.mats_load_name == 'info_gp':
                dense_pred_products = info_gp_mass(
                    scene,
                    dense_pts,
                    mat_cell_products,
                    mat_names,
                    args
                )

            total_pred_val = (dense_pred_products).sum(0)

        if not '2d' in args.feature_load_name and args.evaluate_segmentation == 'None' and not args.viz_only:
            if 'ho3d' not in args.data_dir.lower():
                carved, grid_cell_size = get_carved_pts(scene_dir, dist_thr_ns=0.05)
                
                bound_volume = grid_cell_size ** 3 * len(carved)
            total_volume = (dense_pred_probs @ mat_cell_volumes).max(1)[0].sum(0)
            #print('total volume before bound:', total_volume.item())
            #print('bound volume:', bound_volume)
            if 'ho3d' not in args.data_dir.lower() and total_volume > bound_volume:
                total_pred_val *= bound_volume / total_volume

            if args.volume_method == 'thickness_sum':
                total_volume_thickness_sum = np.sum(mat_vol_all.cpu().numpy(), axis=0)  # (2,)
                if args.log_volume:
                    log_volume_path = os.path.join(scene_dir, f'{names_dict["predict_name"]}_volume_thickness_sum.npy')

                    np.save(log_volume_path, total_volume_thickness_sum)
                    np.save(log_volume_path2, total_volume_thickness_sum)

                mean_sim_vector = dense_pred_probs.mean(0)  # (K,)
                mean_density = mean_sim_vector @ mat_vals  # (2,)

                # convert to np
                mean_density = mean_density.cpu().numpy()  # (2,)                                # (N, 2) where N is the number of dense points

                total_pred_val = mean_density * total_volume_thickness_sum * (args.correction_factor / 100) #shape (2,)

                total_pred_val = torch.tensor(total_pred_val, dtype=torch.float32, device=args.device)  # (2,)

        #print('volume method', args.volume_method)
        if args.volume_method == 'thickness_physx_volume':
            volume = physx_vol_dict[scene_name]
            volume *= args.physx_scale ** 3
            
            if args.nerf_vs_physx_scale != 'None':
                nerf_vs_physx_scale = nerf_vs_physx_scale_dict[scene_name]
                volume *= nerf_vs_physx_scale ** 3
                print('applying nerf2physic scale', nerf_vs_physx_scale)

            mean_density = (dense_pred_probs.mean(0) @ mat_vals).cpu().numpy()  # (2,)
            total_pred_val = mean_density * volume
        elif args.volume_method == 'thickness_sam3d_volume':
            sam3d_dir = os.path.join(scene_dir, 'sam3d_obj')
            volume = volume_reconstructed(sam3d_dir, scene_dir)

            mean_density = (dense_pred_probs.mean(0) @ mat_vals).cpu().numpy()  # (2,)
            total_pred_val = mean_density * volume
            #print('hehehehhehe')
        if args.volume_method != 'thickness_physx_volume' and args.evaluate_segmentation == 'None' and not args.viz_only:
            total_pred_val *= (args.correction_factor / 100)

    elif args.volume_method == 'carving':
        carved, grid_cell_size = get_carved_pts(scene_dir)
        carved_pred_probs = get_interpolated_values(source_pts, source_pred_probs, carved, batch_size=2048, k=args.k)
        carved_pred_vals = carved_pred_probs @ mat_vals
        grid_cell_volume = grid_cell_size ** 3
        total_pred_val = carved_pred_vals.sum(0) * grid_cell_volume * (args.correction_factor / 100)

        dense_pts = carved
        dense_pred_probs = carved_pred_probs

    else:
        raise NotImplementedError
    
    print('-' * 50)
    print('scene:', scene_name)
    print('-' * 50)
    print('num. dense points:', len(dense_pts))
    print('caption:', info['caption'])
    print('candidate materials:')
    if args.property_name == 'density':
        for mat_i, mat_name in enumerate(mat_names):
            print('%16s: %8.1f -%8.1f kg/m^3, %5.1f -%5.1f cm' % (mat_name, mat_vals[mat_i][0], mat_vals[mat_i][1],
                mat_tns[mat_i][0] * 100, mat_tns[mat_i][1] * 100))

    if args.evaluate_segmentation == 'None' and not args.viz_only:
        if not args.use_3d_volume:
            print('surface cell size: %.4f cm' % (surface_cell_size * 100))
        if args.property_name == 'density':
            print('predicted total mass: [%.4f - %.4f kg]' % (total_pred_val[0], total_pred_val[1]))

    #predict_save(args, scene_dir, source_pred_probs, query_pred_probs)

    if args.show_mat_seg:
        # Visualize material segmentation in open3d
        cmap = mpl.colormaps['tab10']
        mat_colors = [cmap(i)[:3] for i in range(len(mat_names))]
        dense_pred_colors = np.array([mat_colors[i] for i in dense_pred_probs.argmax(1)])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(dense_pts.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(dense_pred_colors)
        o3d.visualization.draw_geometries([pcd])
    
    dense_features = get_interpolated_values(
        source_pts, agg_patch_features, dense_pts, batch_size=2048, k=args.k
    )

    dense_pred_vals = dense_pred_probs @ mat_vals

    if args.property_name == 'density':
        mat_tns_all = dense_pred_probs @ mat_tns #shape (N, 2)
    #mat_vol_all = dense_pred_probs @ mat_cell_volumes #shape (N, 2)

    pred_dict = {
        'dense_pred_probs': dense_pred_probs.cpu().numpy(),
        'source_pts': source_pts.cpu().numpy(),
        'patch_features': agg_patch_features.cpu().numpy(),
        'source_similarities': similarities.cpu().numpy(),
        'source_pred_probs': source_pred_probs.cpu().numpy(),
        'mat_names': mat_names,
        'mat_vals': mat_vals.cpu().numpy(),
        'image_features': agg_patch_features.cpu().numpy(),
        'text_features': text_features.cpu().numpy(),
        # 'segmentation_evaluation': segmentation_evaluation if args.testing and args.model != 'None' and args.evaluate_segmentation != 'None' and 'abo' in args.data_dir else None,
        # 'wrong_gt_evaluation': wrong_gt_check if (args.testing and args.model != 'None' and args.evaluate_segmentation != 'None' and 'abo' in args.data_dir) else None,
        #'nerf_wrong_gt_evaluation': wrong_gt_check_bool if args.feature_load_name == 'default' and args.evaluate_segmentation == 'intensive' else None,
        'gt_sim': gt_sim.cpu().numpy() if args.testing and args.model != 'None' and args.evaluate_segmentation != 'None' and 'abo' in args.data_dir else None,
        'manu_gt_sim': manu_gt_sim.cpu().numpy() if args.testing and args.model != 'None' and args.evaluate_segmentation != 'None' and 'abo' in args.data_dir else None,
        'mask_paths': mask_paths_dict,
        # 'iou_per_class_dict': iou_per_class_dict if args.evaluate_segmentation == 'intensive' else None,
        # 'iou_per_class': iou_per_class if args.evaluate_segmentation == 'intensive' else None,
        # 'dense_pts_2d': dense_pts_2d if args.feature_load_name == 'default' and args.evaluate_segmentation == 'intensive' else dense_pts.cpu().numpy(),
        # 'dense_pred_probs_gt': dense_pred_probs_gt.cpu().numpy() if args.evaluate_segmentation == 'intensive' and 'abo' in args.data_dir else None,
        # 'dense_pred_probs': dense_pred_probs.cpu().numpy(),
        # 'segmap_pred': segmap_pred if args.evaluate_segmentation == 'intensive' else None,
        # 'segmap_gt': segmap_gt if args.evaluate_segmentation == 'intensive' else None,
        'good_volume_scene': good_volume_scene,
        'dense_features': dense_features.cpu().numpy(),
        "dense_pred_vals": dense_pred_vals.cpu().numpy(),
        'dense_pts': dense_pts.cpu().numpy(),
        "surface_cell_size": surface_cell_size if args.evaluate_segmentation == 'None' and not args.viz_only else None,
        "mat_tns_all": mat_tns_all.cpu().numpy() if args.evaluate_segmentation == 'None' and not args.viz_only else None,  # shape (N, 2)
        "mat_vol_all": mat_vol_all.cpu().numpy() if args.evaluate_segmentation == 'None' and not args.viz_only else None,  # shape (N,
        "volume_nerf_sum": total_volume_thickness_sum if args.volume_method == 'thickness_sum' and not args.viz_only else None,
        'attention_output': attention_output.cpu().numpy() if args.model == 'mlp_attention' and args.testing else None,
        }

    if args.evaluate_segmentation == 'None' and not args.viz_only:
        pred_dict.update({
            'total_pred_val': total_pred_val.cpu().numpy() if isinstance(total_pred_val, torch.Tensor) else np.array(total_pred_val),  # shape (2,) or () depending on parsing
        })
    if args.plot_cuboids_mode != 'None':
        pred_dict.update({
            "dense_pts": dense_pts.cpu().numpy() if args.feature_load_name != '2d_patch' else dense_pts_3d.cpu().numpy(),
            "voxel_size": surface_cell_size,
            "mat_thickness": mat_tns.cpu().numpy(),  # shape (K, 2) or (K,) depending on parsing
            "c2w": c2w,
            "is_visible": is_visible.cpu().numpy(),
        })

    if getattr(args, 'plot_dims', False):
        pred_dict.update({
            'surface_cell_size': surface_cell_size,
            'avg_thickness_low': mat_tns[:, 0].mean().item(),
            'avg_thickness_high': mat_tns[:, 1].mean().item(),
        })
    segmentation_input_dict = {
        'scene_name': scene_name,
        'scene_dir': scene_dir,
        'mat_names': mat_names,
        'masks_dir': masks_dir,
        # 'out_dir': out_dir,
        'source_pts': source_pts.cpu().numpy() if isinstance(source_pts, torch.Tensor) else source_pts,
        'dense_pred_probs': dense_pred_probs.cpu().numpy() if isinstance(dense_pred_probs, torch.Tensor) else dense_pred_probs,
        'dense_pts': dense_pts.cpu().numpy() if isinstance(dense_pts, torch.Tensor) else dense_pts,
        'mat_names_gt': mat_names_gt if 'abo' not in args.data_dir else None,
    }


    if ('abo' in args.data_dir.lower() or 'physx' in args.data_dir.lower() or 'ho3d' in args.data_dir.lower()) and not args.viz_only:
        segmentation_input_dict.update({
            'w2cs': w2cs,
            'K': K,
        })

        pred_dict.update({
            'w2cs': w2cs,
            'K': K,
        })
    plotting_related_dict = {
        'source_similarities': similarities.detach().cpu().numpy(),
        'source_pts_2d': source_pts_2d,
    }

    if args.evaluate_segmentation != 'None' or args.viz_only:
        total_pred_val = None
    return total_pred_val.tolist() if total_pred_val is not None else None, pred_dict, segmentation_input_dict, plotting_related_dict


def get_gp_iou_per_class(args, scene_dir, dataset = 'abo'):
    segmap_pred_path = os.path.join(args.segmap_pred_dir, f'{scene}_00', '001_global_map.npy')
    segmap_pred = np.load(segmap_pred_path)
    scene_name = os.path.basename(scene_dir)
    
    if dataset == 'abo':
        segmap_gt_path = os.path.join(scene_dir, 'global_material_map.npy')
        segmap_gt = np.load(segmap_gt_path)
        
    elif dataset == 'mvimg' or dataset == 'physx':
        segmap_gt = load_mvimg_segmap_gt(args, scene_name)
        if segmap_gt.shape != segmap_pred.shape:
            segmap_gt = align_label_map(
                src_mask = segmap_gt,
                ref_mask = segmap_pred,
            )
    mask_paths = masks_related_names(scene_dir, args)
    masks_dir = mask_paths["masks_dir"]

    miou, iou_per_class, wrong_gt_check_bool = scene_iou(
        pred_map=segmap_pred,
        gt_map=segmap_gt,
        supercat_to_idx=read_json(args.supercat_to_idx_path),
        masks_dir = masks_dir
    )

    iou_per_class_dict = {}
    for class_name, class_idx in read_json(args.supercat_to_idx_path).items():
        if dataset == 'mvimg':
            class_idx -= 1
        iou_per_class_dict[class_name] = iou_per_class[class_idx] 

    # check if the most common material of pred is same as most common material of gt
    flat_pred = segmap_pred.flatten()
    flat_gt   = segmap_gt.flatten()

    # Filter out invalid values (typically -1 or 255 used as padding/mask)
    flat_pred = flat_pred[flat_pred >= 0]
    flat_gt   = flat_gt[flat_gt >= 0]

    # If either is empty, assume mismatch (or handle however you prefer)
    if flat_pred.size == 0 or flat_gt.size == 0:
        wrong_gt_check = True
        pred_most_common = -1
        gt_most_common = -1
    else:
        pred_most_common = np.bincount(flat_pred).argmax()
        gt_most_common = np.bincount(flat_gt).argmax()
        wrong_gt_check = pred_most_common != gt_most_common
    
    out = {
        'iou_per_class_dict': iou_per_class_dict,
        'iou_per_class': iou_per_class,
        'segmap_pred': segmap_pred,
        'segmap_gt': segmap_gt,
        'wrong_gt_check': wrong_gt_check,
    }
    return out

def evaluate_other_prop(
    pred_dict,
    scene_dir,
    gt_dir,
    args=None,
    viz_dir=None
):
    """
    Build other property maps (e.g., normals, depth) from prediction dictionary.
    
    Args:
        pred_dict: Dictionary containing predicted properties
        scene_dir: Directory of the scene
        gt_dir: Directory of ground truth data
        args: Additional arguments (optional)
    
    Returns: a scalar representing the evaluation metric (e.g., mean error)
    """
    scene_name = os.path.basename(scene_dir)
    gt_path = os.path.join(gt_dir, f'{scene_name}_{args.property_name}.npy')
    
    property_map_gt = np.load(gt_path)

    dense_pred_vals = pred_dict['dense_pred_vals']  # (N,2)
    dense_pts = pred_dict['dense_pts']          # (N,2) or (N,3)
    if args.verbose:
        print(f'dense pred val shape: {dense_pred_vals.shape}, dense pts shape: {dense_pts.shape}')
    if args.feature_load_name == 'default':
        w2cs = pred_dict['w2cs']              # (M,4,4)
        K = pred_dict['K']                    # (M,3,3)
        # convert to np if not already
        if isinstance(dense_pts, torch.Tensor):
            dense_pts = dense_pts.cpu().numpy()
        dense_pts_2d = project_3d_to_2d(
            dense_pts,
            w2cs[0],  # Use the first view's camera
            K
        )

        # keep points within image bounds
        H, W = property_map_gt.shape
        valid_mask = (dense_pts_2d[:, 0] >= 0) & (dense_pts_2d[:, 0] < W) & (dense_pts_2d[:, 1] >= 0) & (dense_pts_2d[:, 1] < H)
        dense_pts_2d = dense_pts_2d[valid_mask]
        dense_pred_vals = dense_pred_vals[valid_mask]

        if args.verbose:
            print(f'Projected dense pts to 2D shape: {dense_pts_2d.shape}')
    elif args.feature_load_name == '2d_patch':
        dense_pts_2d = dense_pts # (N,2)
    
    # conveert to int
    dense_pts_2d = dense_pts_2d.astype(np.int32)
    # convert the dense_pred_val to a full image
    H, W = property_map_gt.shape
    property_map_pred = np.zeros((H, W), dtype=np.float32)
    property_map_pred[dense_pts_2d[:, 1], dense_pts_2d[:, 0]] = dense_pred_vals.mean(1)  # use the mean value if it's a range


    # # compute the error
    # valid_mask = property_map_pred > 0 
    # if valid_mask.sum() == 0:
    #     if args.verbose:
    #         print(f"No valid pixels found in prediction for scene {scene_name}. Skipping evaluation.")
    #     return None  # no valid pixels to evaluate
    

    # error_map = np.abs(property_map_pred - property_map_gt) * valid_mask
    # mean_error = error_map.sum() / valid_mask.sum()

    # if args.verbose:
    #     print(f"Mean error for other property: {mean_error:.4f}")
    
    # if args.debug_plot and viz_dir is not None:
    #     pred_vs_gt_path = os.path.join(viz_dir, f'pred_vs_gt_{args.property_name}.png')
    #     plot_np_array(property_map_pred, property_map_gt, save_path=pred_vs_gt_path)
    mean_error = other_prop_error(
        property_map_pred,
        property_map_gt,
        args=args,
        viz_dir=viz_dir
    )

    return mean_error

if __name__ == '__main__':

    args = get_args()

    # if args.mask_prior_lambda != 1:
    #     assert args.use_sam == True 
    if args.nerf_vs_physx_scale != 'None':
        assert args.physx_scale == 1

    scenes_dir = os.path.join(args.data_dir, 'scenes')
    scenes = get_scenes_list(args)

    clip_model, _, preprocess = open_clip.create_model_and_transforms(CLIP_BACKBONE, pretrained=CLIP_CHECKPOINT)
    clip_model.to(args.device)
    clip_tokenizer = open_clip.get_tokenizer(CLIP_BACKBONE)

    preds = {}
    pred_dicts = {}
    pred_dicts_vol = {}
    pred_dicts_other = {}
    names_dict = name_creation(args)

    not_match_gt_scene = 0
    wrong_gt_scene = 0

    iou_per_class_all = []
    good_volume_scenes = []

    if args.evaluate_segmentation == 'intensive':
        if 'mvimg' in args.data_dir.lower():
            assert 'mvimg' in args.supercat_to_idx_path, "Supercategory to index mapping should be for mvimg dataset."
            assert 'mvimg' in args.material_to_supercat_path, "Material to supercategory mapping should be for mvimg dataset."

    all_inter = 0
    all_total = 0
    num_pts = []

    splits = read_json(os.path.join(args.data_dir, 'splits.json'))
    heavy_close_split = None
    if args.mats_load_name == 'combine':
        heavy_close_split = splits['heavy_close_test'] if 'heavy_close_test' in splits else None
    elif 'combine' in args.mats_load_name:
        #heavy_close_split = splits['heavy_close_test_llm'] if 'heavy_close_test_llm' in splits else None
        heavy_close_split = splits[f'heavy_close_{args.split}_llm'] if f'heavy_close_{args.split}_llm' in splits else None
 



    if args.verbose:
        print('evaluate_segmentation:', args.evaluate_segmentation)
    
    # build basic metadata 
    # if 'abo' in args.data_dir.lower():
    #     dataset = 'abo500'
    # elif 'mvimg' in args.data_dir.lower():
    #     dataset = 'mvimg'
    # elif 'physx' in args.data_dir.lower():
    #     dataset = 'physx'
    
    names_dict = name_creation(args)
    dataset = names_dict['dataset']
    material_model_name = names_dict['material_model_name']
    if args.evaluate_segmentation == 'intensive':
        out_root_dir = f'viz/{dataset}/materal_eval_{args.viz_save_name}/{material_model_name}'
    elif args.evaluate_segmentation == 'other_property':
        out_root_dir = f'viz/{dataset}/{args.property_name}_eval/{material_model_name}'
        all_errors = []

    if args.debug_plot:
        pdf_images = []
        #pdf_path = os.path.join('viz', f'{args.viz_save_name}', get_out_dir_all(args), f'{args.viz_save_name}_segmentation_evaluation.pdf')
        pdf_path = os.path.join(out_root_dir, 'all_scenes', f'{args.viz_save_name}_segmentation_evaluation.pdf')
        os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    for j, scene in tqdm(enumerate(scenes)):
        debugging = False
        if args.debug_scene_name != 'None' or args.debug_scene_idx != 'None':
            if scene != args.debug_scene_name and j != int(args.debug_scene_idx):
                continue
            debugging = True
        scene_dir = os.path.join(scenes_dir, scene)

        scene = os.path.basename(scene_dir)
        #out_dir = get_out_dir(args, scene_name=scene)

        if args.prediction_mode == 'integral':
            if args.overwrite:
                out = predict_physical_property_integral(args, scene_dir, clip_model, clip_tokenizer, preprocess, heavy_close_split=heavy_close_split, debugging=debugging)
                
                if out is None:
                    print(f"[⚠️] Skipping scene {scene} idx {j} due to missing features.")
                    break
                pred, pred_dict, segmentation_input_dict, plotting_related_dict = out

                if args.evaluate_segmentation == 'intensive':
                    if args.verbose:
                        print('Evaluating segmentation for scene:', scene)
                    final_seg_results = evaluate_material_segmentation(
                        segmentation_input_dict = segmentation_input_dict,
                        clip_model = clip_model,
                        clip_tokenizer = clip_tokenizer,
                        args=args,
                        scene_name=scene,
                        scene_dir=scene_dir,
                        plotting_related_dict=plotting_related_dict,
                        iou_per_class_all=iou_per_class_all,
                        out_dir=os.path.join(out_root_dir, scene)
                    )

                    if args.verbose:
                        print('out_path:', final_seg_results['out_path'])
                    out_path = final_seg_results['out_path']
                    # read image and add to pdf_images
                    image = Image.open(out_path).convert("RGB")
                    pdf_images.append(image)
                
                elif args.evaluate_segmentation == 'other_property':
                    #print("hahahahahahah")
                    #print('pred dict', pred_dict.keys())
                    evaluate_metric = evaluate_other_prop(
                        pred_dict = pred_dict,
                        scene_dir = scene_dir,
                        gt_dir = args.segmap_gt_dir,
                        args=args,
                        viz_dir = os.path.join(out_root_dir, scene)
                    )
                    all_errors.append(evaluate_metric)
                num_pts.append(len(pred_dict['dense_pts']))
                
                if args.feature_load_name == 'default':
                    if args.evaluate_segmentation == 'None':
                        iou_volume, scene_inter, scene_total = overlap_volume_ratio(
                            centers = pred_dict['dense_pts'],
                            mat_tns = pred_dict['mat_tns_all'],
                            mat_vol = pred_dict['mat_vol_all'],
                            voxel_size = pred_dict['surface_cell_size'],
                        )
                        #print('iou_volume:', iou_volume)
                        all_inter += scene_inter
                        all_total += scene_total

                if pred_dict["good_volume_scene"]:
                    good_volume_scenes.append(scene)

                if args.feature_load_name == '2d_patch':
                    not_match_gt_scene, wrong_gt_scene = count_bad_cases(
                        pred_dict = pred_dict,
                        scene_dir = scene_dir,
                        not_match_gt_scene = not_match_gt_scene,
                        wrong_gt_scene = wrong_gt_scene,
                        args = args
                    )
                elif args.feature_load_name == 'default' and args.segmap_pred_path == 'None':
                    if args.evaluate_segmentation == 'intensive':
                        if pred_dict['nerf_wrong_gt_evaluation']:
                            not_match_gt_scene += 1

                
                if args.evaluate_segmentation == 'intensive':
                    iou_per_class_all.append(final_seg_results['iou_per_class'])

                    # Plot origimage, pred, gt segmentation, iou per class, conversion from categories to supercategories
            else:
                pred_dict = {}
                # Choose folder and file name
                name = 'preds' if not '2d' in args.feature_load_name else 'preds_2d'
                pred_file = os.path.join(name, f'{name}_{args.preds_save_name}.json')

                # Load previous predictions if available
                if os.path.exists(pred_file):
                    with open(pred_file, 'r') as f:
                        pred_orig = json.load(f)
                else:
                    pred_orig = {}

                # Check if scene has valid prediction
                if scene in pred_orig:
                    pred_value = pred_orig[scene]
                    if (
                        isinstance(pred_value, list)
                        and len(pred_value) == 2
                        and all(isinstance(x, float) and not np.isnan(x) for x in pred_value)
                    ):
                        pred = pred_value
                    else:
                        print(f"[Recompute] NaN or invalid prediction for scene: {scene}")
                        pred, pred_dict, _,_ = predict_physical_property_integral(args, scene_dir, clip_model, clip_tokenizer, preprocess, heavy_close_split=heavy_close_split, debugging=debugging)
                else:
                    print(f"[Recompute] No prediction found for scene: {scene}")
                    pred, pred_dict,_,_ = predict_physical_property_integral(args, scene_dir, clip_model, clip_tokenizer, preprocess, heavy_close_split=heavy_close_split, debugging=debugging)

            preds[scene] = pred
            pred_dicts_other[scene] = {}
            pred_dicts_other[scene]['volume_nerf_sum'] = pred_dict['volume_nerf_sum'].tolist() if args.volume_method == 'thickness_sum' else None
            pred_dicts[scene] = pred_dict
        
        elif args.prediction_mode == 'grid':
            pred = predict_physical_property_query(args, 'grid', scene_dir, clip_model, clip_tokenizer)

        elif args.prediction_mode == 'gp_eval' or args.prediction_mode == 'octopi_eval':
            #print('hahahahahhaah')
            if args.evaluate_segmentation == 'intensive':
                if 'abo' in args.data_dir:
                    dataset = 'abo'
                elif 'mvimg' in args.data_dir.lower():
                    dataset = 'mvimg'
                elif 'physx' in args.data_dir.lower():
                    dataset = 'physx'

                if args.prediction_mode == 'gp_eval':
                    out = get_gp_iou_per_class(
                        args = args,
                        scene_dir = scene_dir,
                        dataset = dataset
                    )
                
                elif args.prediction_mode == 'octopi_eval':
                    out = get_octopi_iou_per_class(
                        args = args,
                        scene_dir = scene_dir,
                        dataset = dataset
                    )

                iou_per_class_dict = out['iou_per_class_dict']
                iou_per_class = out['iou_per_class']
                segmap_pred = out['segmap_pred']
                segmap_gt = out['segmap_gt']
                wrong_gt_check_bool = out['wrong_gt_check']

                iou_per_class_all.append(iou_per_class)
                segmap_pred_path = os.path.join(args.segmap_pred_dir, f'{scene}_00', '001_global_map.npy')
                #print('segmap_pred_path:', segmap_pred_path)
                if args.debug_plot:
                    out_path = plot_segmaps(
                        segmap_pred = segmap_pred,
                        segmap_gt = segmap_gt,
                        orig_image_path = os.path.join(scene_dir, 'images', f'{scene}_00.png'),
                        supercat_to_idx = read_json(args.supercat_to_idx_path),
                        out_dir = os.path.join(out_root_dir, scene),
                        args=args,
                        iou_per_class_dict = iou_per_class_dict
                    )
                    log_paths(args.log_file, [out_path])

                    pdf_images.append(Image.open(out_path).convert("RGB"))

                if wrong_gt_check_bool:
                    wrong_gt_scene += 1
                    #print(f"[Warning] Wrong GT check failed for scene: {scene}")
            elif args.evaluate_segmentation == 'other_property':
                #print('hohohooho')
                out = evaluate_other_prop_octopi(
                    scene_dir = scene_dir,
                    gt_dir = args.segmap_gt_dir,
                    dataset= dataset,
                    args = args,
                    viz_dir = os.path.join(out_root_dir, scene)
                )
                all_errors.append(out)
        else:
            raise NotImplementedError("Unsupported prediction mode.")

        if args.debug_scene_name != 'None' or  args.debug_scene_idx != 'None':
            print(f"Debugging scene: {scene}")
            break
    
    # convert all inter and all total from np to float
    all_inter = float(all_inter)
    all_total = float(all_total)

    pred_dicts_vol['all_vol_iou'] = all_inter / all_total if all_total > 0 else 0
    
    if args.log_good_volume:
        split_file = read_json(os.path.join(args.data_dir, 'splits.json'))
        split_file[f'good_volume_{args.split}'] = good_volume_scenes
        write_json(split_file, os.path.join(args.data_dir, 'splits.json'))

    if args.plot_dims:
        plot_dims_from_pred_dicts(pred_dicts)

    if args.save_pred:
        # Save predictions dicts
        save_prediction_dicts(args, pred_dicts)
    
    if args.debug_plot and len(pdf_images) > 0:
        #os.makedirs('viz', exist_ok=True)
        pdf_images[0].save(
            pdf_path, save_all=True, append_images=pdf_images[1:]
        )
        print(f"[📄] Saved debug plots to {pdf_path}")
        log_paths(args.log_file, [pdf_path])
    # Save predictions
    if args.prediction_mode == 'integral' and args.save_preds:
        name = 'preds' if not '2d' in args.feature_load_name else 'preds_2d'
        os.makedirs(name, exist_ok=True)
        pred_json_path = os.path.join(name, f'{name}_{args.preds_save_name}_{names_dict["predict_name"]}.json')
        #print('pred json path'. pred_json_path)
        pred_dict_path = os.path.join(name, f'{name}_{args.preds_save_name}_{names_dict["predict_name"]}_dict.json')
        pred_dicts_other_path = os.path.join(name, f'{name}_{args.preds_save_name}_{names_dict["predict_name"]}_dict_other.json')
        num_points_path = os.path.join(name, f'{name}_{args.preds_save_name}_{names_dict["predict_name"]}_num_points.npy')

        np.save(num_points_path, np.array(num_pts))
        #print('pred json path:', pred_json_path)
        
        with open(pred_json_path, 'w') as f:
            json.dump(preds, f, indent=4)
        
        write_json(pred_dicts_vol, pred_dict_path)
        write_json(pred_dicts_other, pred_dicts_other_path)

        if args.evaluate:
            print("🔍 Evaluating predictions...")
            command = build_common_args(args)
            #print('command:', command)
            command = f"python evaluation.py {command} --preds_json_path {pred_json_path} --pred_dicts_path {pred_dict_path} --pred_other_path {pred_dicts_other_path} --num_points_path {num_points_path} --general_config {args.general_config}"
            #print('pred json path:', pred_json_path)
            subprocess.run(command, shell=True)
        
        # Save segmentation evaluation results
        #segmentation_evaluation(args, not_match_gt_scene, wrong_gt_scene)
    summarize_segmentation_results(
        args=args,
        not_match_gt_scene=not_match_gt_scene,
        wrong_gt_scene=wrong_gt_scene, 
        iou_per_class_all=iou_per_class_all,
        mean_error=np.mean(all_errors) if args.evaluate_segmentation == 'other_property' else None
    )
