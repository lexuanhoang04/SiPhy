import sys, os 
sys.path.insert(0, os.getcwd())

from utils import * 
from arguments import get_args 
from plot.plot_utils import plot_np_array

from utils_pixel_level import material_to_other, other_prop_error



def get_octopi_iou_per_class(
    args,
    scene_dir,
    dataset
):
    # scene = os.path.basename(scene_dir)
    # if dataset == 'mvimg' or dataset == 'physx':
    #     segmap_gt = load_mvimg_segmap_gt(args, scene)
    # segmap_pred = octopi_segmap_pred(
    #     octopi_output_json_path = args.octopi_output_json_path,
    #     scene = scene,
    #     gp_combined_mask_path = os.path.join(args.segmap_pred_dir, f'{scene}_00', 'seg', '001_s.npy'),
    #     supercat_to_idx_path = args.supercat_to_idx_path,
    # )
    # if segmap_gt.shape != segmap_pred.shape:
    #     segmap_gt = align_label_map(
    #             src_mask = segmap_gt,
    #             ref_mask = segmap_pred,
    #     )
    # miou, iou_per_class, wrong_gt_check_bool = scene_iou(
    #     pred_map=segmap_pred,
    #     gt_map=segmap_gt,
    #     supercat_to_idx = read_json(args.supercat_to_idx_path)
    
    out = octopi_segmaps(
        args,
        scene_dir,
        dataset
    )
    segmap_pred = out['segmap_pred']
    segmap_gt = out['segmap_gt']
    iou_per_class = out['iou_per_class']
    wrong_gt_check_bool = out['wrong_gt_check']

    #print('iou per class', iou_per_class)
    #print('supercat to idx:', read_json(args.supercat_to_idx_path))
    iou_per_class_dict = {}
    for class_name, class_idx in read_json(args.supercat_to_idx_path).items():
        if dataset == 'mvimg':
            class_idx -= 1 # mvimgnet has background as 0
        iou_per_class_dict[class_name] = iou_per_class[class_idx]

    out = {
        'iou_per_class_dict': iou_per_class_dict,
        'iou_per_class': iou_per_class,
        'segmap_pred': segmap_pred,
        'segmap_gt': segmap_gt,
        'wrong_gt_check': 0
    }
    return out

def octopi_segmaps(
    args,
    scene_dir,
    dataset
):
    scene = os.path.basename(scene_dir)
    if dataset == 'mvimg' or dataset == 'physx' or dataset == 'ho3d' or dataset == 'arctic':
        segmap_gt = load_mvimg_segmap_gt(args, scene)
    segmap_pred = octopi_segmap_pred(
        octopi_output_json_path = args.octopi_output_json_path,
        scene = scene,
        gp_combined_mask_path = os.path.join(args.segmap_pred_dir, f'{scene}_00', 'seg', '001_s.npy'),
        supercat_to_idx_path = args.supercat_to_idx_path,
    )
    if segmap_gt.shape != segmap_pred.shape:
        segmap_gt = align_label_map(
                src_mask = segmap_gt,
                ref_mask = segmap_pred,
        )
    miou, iou_per_class, wrong_gt_check_bool = scene_iou(
        pred_map=segmap_pred,
        gt_map=segmap_gt,
        supercat_to_idx = read_json(args.supercat_to_idx_path)
    )
    out = {
        'segmap_pred': segmap_pred,
        'segmap_gt': segmap_gt,
        'iou_per_class': iou_per_class,
        'wrong_gt_check': wrong_gt_check_bool

    }
    return out

def evaluate_other_prop_octopi(scene_dir, gt_dir, dataset, args, viz_dir=None):
    out = octopi_segmaps(args, scene_dir, dataset)
    material_segmap_pred = out['segmap_pred']   # shape: (H, W)

    # super_idx_to_otherprop = read_json(args.superidx_to_otherprop)

    # max_idx = 100
    # other_lookup = np.zeros(max_idx + 1, dtype=np.float32)
    # for k, v in super_idx_to_otherprop.items():
    #     other_lookup[int(k)] = float(v)

    # other_pred = other_lookup[material_segmap_pred]
    
    other_pred = material_to_other(material_segmap_pred, args)

    scene_name = os.path.basename(scene_dir)
    gt_path = os.path.join(gt_dir, f'{scene_name}_{args.property_name}.npy')
    other_gt = np.load(gt_path)
    
    #print('other prop shapes:', other_pred.shape, other_gt.shape)
    scene_name = os.path.basename(scene_dir)
    orig_image_path = os.path.join(scene_dir, 'images', f'{scene_name}_00.png')
    mean_error = other_prop_error(
        property_map_pred=other_pred,
        property_map_gt=other_gt,
        args=args,
        viz_dir=viz_dir,
        orig_image_path=orig_image_path
    )
    #print('mean error', mean_error)
    return mean_error

def main():
    args = get_args()

    scenes = get_scenes_list(args)

    for scene in scenes:
        segmap_gt = load_mvimg_segmap_gt(scene, args)
        segmap_pred = octopi_segmap_pred(
            octopi_output_json_path = read_json(args.octopi_output_json_path),
            scene = scene,
            gp_combined_mask_path = os.path.join(args.segmap_pred_dir, f'{scene}_00', 'seg', '001_s.npy'),
            supercat_to_idx_path = args.supercat_to_idx_path,
        )

        miou, iou_per_class = scene_iou(
            pred_map=segmap_pred,
            gt_map=segmap_gt,
            supercat_to_idx = read_json(args.supercat_to_idx_path)
        )
    
        #print('iou per class:', iou_per_class)