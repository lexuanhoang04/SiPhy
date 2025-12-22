import os
import time
import json
import openai
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import shutil

from gpt_inference import gpt_candidate_materials, gpt_thickness, parse_material_list, \
    parse_material_hardness, gpt4v_candidate_materials, gpt4v_thickness, parse_material_json, _parse_heavy_output, gpt_mass_classify
from utils import load_images, get_scenes_list, read_json, write_json
from arguments import get_args
from my_api_key import OPENAI_API_KEY
from utils import read_json, write_json 

BASE_SEED = 100


def gpt_wrapper(gpt_fn, parse_fn, max_tries=10, sleep_time=3):
    """Wrap gpt_fn with error handling and retrying."""
    #print('hehe')
    tries = 0
    # sleep to avoid overloading openai api
    time.sleep(sleep_time)
    try:
        gpt_response = gpt_fn(BASE_SEED + tries)
        gpt_response = gpt_response.strip(";")
        #print('gpt_response:', gpt_response)
        result = parse_fn(gpt_response)
    except Exception as error:
        print('error:', error)
        result = None
    while result is None and tries < max_tries:
        tries += 1
        time.sleep(sleep_time)
        print('retrying...')
        try:
            gpt_response = gpt_fn(BASE_SEED + tries)
            result = parse_fn(gpt_response)
        except:
            result = None
    return gpt_response


def show_img_to_caption(scene_dir, idx_to_caption):
    img_dir = os.path.join(scene_dir, 'images')
    imgs = load_images(img_dir, bg_change=None, return_masks=False)
    img_to_caption = imgs[idx_to_caption]
    plt.imshow(img_to_caption)
    plt.show()
    plt.close()
    return


def predict_candidate_materials(args, scene_dir, show=False):
    if args.materials_existed_name == 'None':
        # load caption info
        with open(os.path.join(scene_dir, '%s.json' % args.caption_load_name), 'r') as f:
            info = json.load(f)
    else:
        # load existing materials info
        with open(os.path.join(scene_dir, '%s.json' % args.materials_existed_name), 'r') as f:
            info = json.load(f)
    
    parse_fn = parse_material_hardness if args.property_name == 'hardness' else parse_material_list

    if args.mats_save_name == 'info_new' or 'candidate' in args.mats_save_name or not 'abo' in args.data_dir.lower():
        caption = info['caption']

        #print('caption:', caption)
        #print('args', args)
        gpt_fn = lambda seed: gpt_candidate_materials(caption, property_name=args.property_name, 
                                                    model_name=args.gpt_model_name, seed=seed, args=args)

        candidate_materials = gpt_wrapper(gpt_fn, parse_fn)
    else:
        # load existing candidate materials
        info_orig = read_json(os.path.join(scene_dir, 'info.json'))
        candidate_materials = info_orig['candidate_materials_%s' % args.property_name]
        
    #print('candidate_materials:', candidate_materials)
    info['candidate_materials_%s' % args.property_name] = candidate_materials
    
    print('-' * 50)
    print('scene: %s, info:' % os.path.basename(scene_dir), info)
    print('candidate materials (%s):' % args.property_name)
    mat_names, mat_vals = parse_fn(candidate_materials)
    for mat_i, mat_name in enumerate(mat_names):
        print('%16s: %8.1f -%8.1f' % (mat_name, mat_vals[mat_i][0], mat_vals[mat_i][1]))
    if show:
        show_img_to_caption(scene_dir, int(info['idx_to_caption']))
    
    if args.materials_existed_name == 'None':
        # save info to json
        with open(os.path.join(scene_dir, '%s.json' % args.mats_save_name), 'w') as f:
            json.dump(info, f, indent=4)
            #print('Caption saved to:', os.path.join(scene_dir, '%s.json' % args.mats_save_name))
    else:
        # update info in existing json
        with open(os.path.join(scene_dir, '%s.json' % args.materials_existed_name), 'w') as f:
            json.dump(info, f, indent=4)
            
    return info


def predict_object_info_gpt4v(args, scene_dir, show=False):
    """(EXPERIMENTAL) Predict materials directly from image with GPT-4V."""
    img_dir = os.path.join(scene_dir, 'images')
    imgs, masks = load_images(img_dir, return_masks=True)
    mask_areas = [np.mean(mask) for mask in masks]

    idx_to_caption = np.argsort(mask_areas)[int(len(mask_areas) * args.mask_area_percentile)]
    img_to_caption = imgs[idx_to_caption]

    # save img_to_caption in img_dir
    img_to_caption = Image.fromarray(img_to_caption)
    img_path = os.path.join(scene_dir, 'img_to_caption.png')
    img_to_caption.save(img_path)

    gpt_fn = lambda seed: gpt4v_candidate_materials(img_path, property_name=args.property_name, seed=seed)
    candidate_materials = gpt_wrapper(gpt_fn, parse_material_json)

    info = {'idx_to_caption': str(idx_to_caption), 
            'candidate_materials_%s' % args.property_name: candidate_materials}
    
    print('-' * 50)
    print('scene: %s, info:' % os.path.basename(scene_dir), info)
    print('candidate materials (%s):' % args.property_name)
    mat_names, mat_vals = parse_material_list(candidate_materials)
    for mat_i, mat_name in enumerate(mat_names):
        print('%16s: %8.1f -%8.1f' % (mat_name, mat_vals[mat_i][0], mat_vals[mat_i][1]))
    if show:
        show_img_to_caption(scene_dir, int(info['idx_to_caption']))
    
    # save info to json
    with open(os.path.join(scene_dir, '%s.json' % args.mats_save_name), 'w') as f:
        json.dump(info, f, indent=4)

    return info


def predict_thickness(args, scene_dir, mode='list', show=False, idx_to_supercat=None, property_name2='thickness', heavy=False):
    #print('scene_dir:', scene_dir)
    if args.materials_existed_name == 'None':
        # load info
        if os.path.exists(os.path.join(scene_dir, '%s.json' % args.mats_save_name)):
            with open(os.path.join(scene_dir, '%s.json' % args.mats_save_name), 'r') as f:
                info = json.load(f)
        else:
            info = read_json(os.path.join(scene_dir, 'info_new.json'))
    else:
        # load existing materials info
        with open(os.path.join(scene_dir, '%s.json' % args.materials_existed_name), 'r') as f:
            info = json.load(f)

    if mode == 'list': 
        caption = info['caption']
    elif mode == 'json':  # json contains caption inside
        caption = None
    else:
        raise NotImplementedError

    #if args.caption_load_name != 'info_gp':
    candidate_materials = info['candidate_materials_density']

    print('candiate_materials', candidate_materials)
    if not 'new' in args.mats_save_name and args.mats_save_name != 'info_dim' or args.mats_save_name == 'info_new':
        #print('hehehehe')
        if args.verbose:
            print('predicting thickness for scene:', os.path.basename(scene_dir))

        gpt_fn = lambda seed: gpt_thickness(caption, candidate_materials, 
                                            model_name=args.gpt_model_name,  mode=mode, seed=seed, args=args)
    else:
        #print('heheheheheh')
        if heavy or args.mats_save_name == 'info_dim':
            scene_name = os.path.basename(scene_dir)
            image_path = os.path.join(scene_dir, 'images', '%s_00.png' % scene_name)
            gpt_fn = lambda seed: gpt4v_thickness(image_path, caption, candidate_materials, seed=seed, property_name2=property_name2)
        else:
            # load from info
            info_path = os.path.join(scene_dir, 'info.json')
            with open(info_path, 'r') as f:
                info_orig = json.load(f)
            info[property_name2] = info_orig.get(property_name2, None)
            write_json(info, os.path.join(scene_dir, '%s.json' % args.mats_save_name))

            return info

    #print('property', property_name2)
    if property_name2 == 'thickness':
        property2 = gpt_wrapper(gpt_fn, parse_material_list)

        print(f'{property_name2}:', property2)
        #print('thickness (cm):')
        mat_names, mat_vals = parse_material_list(property2)
        for mat_i, mat_name in enumerate(mat_names):
            print('%16s: %8.1f -%8.1f' % (mat_name, mat_vals[mat_i][0], mat_vals[mat_i][1]))

        if show:
            show_img_to_caption(scene_dir, int(info['idx_to_caption']))
    elif property_name2 == 'dimension':
        property2 = gpt4v_thickness(image_path, caption, candidate_materials, property_name2=property_name2)
    
    info[property_name2] = property2
    # save info to json
    if args.materials_existed_name == 'None':
        #print('info:', info)
        # with open(os.path.join(scene_dir, '%s.json' % args.mats_save_name), 'w') as f:
        #     json.dump(info, f, indent=4)
        write_json(info, os.path.join(scene_dir, '%s.json' % args.mats_save_name))
        print(f'saved to {os.path.join(scene_dir, "%s.json" % args.mats_save_name)}')
    else:
        with open(os.path.join(scene_dir, '%s.json' % args.materials_existed_name), 'w') as f:
            json.dump(info, f, indent=4)
            
    return info

# (4) ---------------   MAIN DRIVER   -----------------------------------------
def classify_heavy_light(
    args,
    scene_dir: str,
    mode: str = "list",
    show: bool = False,
):
    """
    Similar signature to predict_thickness().
    Adds two keys to the scene’s JSON:
        'heavy_classification': raw GPT string
        'heavy_state_conf': {'state': 'HEAVY'|'NOT_HEAVY', 'confidence': float}
    """
    # ----- 1. load caption & choose representative image ----------------------
    with open(
        os.path.join(
            scene_dir,
            "%s.json" % (args.mats_save_name if args.materials_existed_name == "None"
                         else args.materials_existed_name)
        ),
        "r",
    ) as f:
        info = json.load(f)

    caption = info["caption"]
    idx_to_caption = int(info["idx_to_caption"])
    img_dir = os.path.join(scene_dir, "images")
    img_path = os.path.join(scene_dir, "img_for_heavy.png")

    # save the chosen view to disk (only once)
    if not os.path.exists(img_path):
        imgs = load_images(img_dir, return_masks=False)
        Image.fromarray(imgs[idx_to_caption]).save(img_path)

    # ----- 2. call GPT via our retry wrapper ---------------------------------
    gpt_fn = lambda seed: gpt_mass_classify(
        caption, img_path, seed=seed
    )
    heavy_raw = gpt_wrapper(gpt_fn, _parse_heavy_output)      # retries & validation

    # ----- 3. parse again for logging / downstream use ------------------------
    heavy_state, confidence = _parse_heavy_output(heavy_raw)

    info["heavy_classification"] = heavy_raw
    info["heavy_state_conf"] = {"state": heavy_state, "confidence": confidence}

    print(f"heavy_state: {heavy_state}, confidence: {confidence:.2f}")

    if show:
        show_img_to_caption(scene_dir, idx_to_caption)

    # ----- 4. save back to the same JSON -------------------------------------
    json_name = (
        args.mats_save_name
        if args.materials_existed_name == "None"
        else args.materials_existed_name
    )
    write_json(info, os.path.join(scene_dir, f"{json_name}.json"))

    return info


if __name__ == '__main__':

    args = get_args()

    scenes_dir = os.path.join(args.data_dir, 'scenes')
    scenes = get_scenes_list(args)

    openai.api_key = OPENAI_API_KEY
    supercat_to_idx = read_json(args.supercat_to_idx_path)
    idx_to_supercat = {v: k for k, v in supercat_to_idx.items()}

    heavy_objs = []
    splits = read_json(os.path.join(args.data_dir, 'splits.json'))
    ran = False
    heavy_close_test_scenes = splits['heavy_close_test_llm'] if 'heavy_close_test_llm' in splits else []

    for j, scene in enumerate(scenes):
        if args.debug_scene_name != 'None' and scene != args.debug_scene_name:
            continue

        if args.caption_load_name == 'info_gp':
            segmap_pred_path = os.path.join('external/Gaussian-Property/Results_abo500_test_dirs', f'{scene}_00', '001_global_map.npy')
            segmap_pred = np.load(segmap_pred_path)
            unique_mat_idx = np.unique(segmap_pred)
            
            unique_mat_idx = unique_mat_idx[unique_mat_idx != -1]  # remove background
            unique_mat_idx = unique_mat_idx.tolist()

            candidate_materials_list = [idx_to_supercat[mat_idx] for mat_idx in unique_mat_idx]
            caption = ', '.join(candidate_materials_list)
            info_file = {'caption': caption}
            write_json(info_file, os.path.join(scenes_dir, scene, 'info_gp.json'))
            mode = 'json'
        
        #print('include_thickness:', args.include_thickness)
        # if args.mats_save_name != 'info_new_llm_tn' or args.include_thickness == 1:
        # if not args.classifying:
        #     #print('hehe')
        #     if args.mats_save_name != 'info_new_llm_tn' and args.mats_save_name != 'info_new_tn' and args.mats_save_name != 'info_dim':
        #         mats_info = predict_candidate_materials(args, os.path.join(scenes_dir, scene))
        #         ran = True
        #     if args.include_thickness:
        #         property_name2 = 'thickness'
        #         if 'dim' in args.mats_save_name:
        #             property_name2 = 'dimension'

        #         json_file_path = os.path.join(scenes_dir, scene, '%s.json' % args.mats_save_name)
        #         # if args.overwrite or scene not in heavy_close_test_scenes and args.mats_save_name == 'info_new_llm_tn':
        #         #     # copy info.json to mats_save_name json
        #         #     shutil.copy(os.path.join(scenes_dir, scene, 'info.json'), json_file_path)
        #         mats_info = predict_thickness(args, os.path.join(scenes_dir, scene), idx_to_supercat=idx_to_supercat, property_name2=property_name2)
        #         ran = True
        # else:
        #     if args.overwrite:
        #         info = classify_heavy_light(args, os.path.join(scenes_dir, scene))
        #     else:
        #         info = read_json(os.path.join(scenes_dir, scene, f'{args.mats_save_name}.json'))
        #     if info['heavy_state_conf']['state'] == 'HEAVY':
        #         heavy_objs.append(scene)

        old_overwrite = args.overwrite
        args.overwrite = False
        if not args.overwrite:
            # if scene in splits['heavy_close_test_llm']:
            #     heavy_objs.append(scene)
            #     continue
            is_heavy = False
            if args.mats_save_name != 'info_new_tn' and args.mats_save_name != 'info_dim' and args.mats_save_name != 'info_new_llm_tn':
                if args.classifying:
                    info = classify_heavy_light(args, os.path.join(scenes_dir, scene))
                else:
                    info = read_json(os.path.join(scenes_dir, scene, f'{args.mats_save_name}.json'))
                if 'heavy_state_conf' in info and info['heavy_state_conf']['state'] == 'HEAVY':
                    heavy_objs.append(scene)
                    is_heavy = True
                mats_info = predict_candidate_materials(args, os.path.join(scenes_dir, scene))
            
            if args.include_thickness:
                property_name2 = 'thickness'
                if 'dim' in args.mats_save_name:
                    property_name2 = 'dimension'
                    if os.path.exists(os.path.join(scenes_dir, scene, 'info_dim.json')):
                        info_path = os.path.join(scenes_dir, scene, 'info_dim.json')
                        info_dim = read_json(info_path)
                        if 'dimension' in info_dim:
                            continue
                mats_info = predict_thickness(args, os.path.join(scenes_dir, scene), idx_to_supercat=idx_to_supercat, property_name2=property_name2, heavy=is_heavy)
                ran = True

        if args.delay_time > 0 and ran:
            time.sleep(args.delay_time)
        if args.debug_scene_name != 'None':
            break
    #if args.mats_save_name == 'info_new_llm_tn' and args.include_thickness == 0:
    if args.classifying:
        if args.split == 'test' and args.overwrite:
            splits['heavy_close_test_llm'] = heavy_objs
        elif args.split == 'train+val':
            splits['heavy_close_train+val_llm'] = heavy_objs
            # combine heavy_close_test_llm and heavy_close_train+val_llm to get heavy_close_all_llm
        splits['heavy_close_all_llm'] = splits['heavy_close_test_llm'] + splits['heavy_close_train+val_llm']
        write_json(splits, os.path.join(args.data_dir, 'splits.json'))
    
    args.overwrite = old_overwrite