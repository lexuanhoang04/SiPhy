import os
import numpy as np 
import json
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration

from utils import load_images, get_scenes_list, read_json
from arguments import get_args
from my_api_key import OPENAI_API_KEY  # Ensure you have this file with your OpenAI API key
import openai
import shutil 

#from tqdm import tqdm
CAPTIONING_PROMPT = "Question: Give a detailed description of the object. Answer:"

CAPTIONING_PROMPT_DETAIL = """You are a product-design expert describing objects for a material-thickness estimator.

Write a detailed 80–120 word caption that:
• Names every distinct part and its likely material.
• States qualitative thickness cues (e.g. thin veneer, solid slab, padded).
• Notes visible edges, layers, joinery or hollowness.
• Gives at least one real-world size reference if scale is obvious.
Use concise, factual sentences. Return ONLY the caption."""

from PIL import Image  # if not already imported
from gpt_inference import gpt_captioning
from tqdm import tqdm
from material_proposal import gpt_wrapper

def load_blip2(model_name, device='cuda'):
    if "instruct" in model_name:
        model = InstructBlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
        processor = InstructBlipProcessor.from_pretrained(model_name)
    else:
        processor = AutoProcessor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()
    return model, processor


def display_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))


def generate_text(img, model, processor, prompt=CAPTIONING_PROMPT, device='cuda', max_new_tokens=30):
    if prompt is not None:
        inputs = processor(img, text=prompt, return_tensors="pt").to(device, torch.float16)
    else:
        inputs = processor(img, return_tensors="pt").to(device, torch.float16)
    
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    #print('generated text:', generated_text)
    return generated_text


# def predict_caption(args, scene_dir, vqa_model, vqa_processor, show=False):
#     img_dir = os.path.join(scene_dir, 'images')
#     imgs, masks = load_images(img_dir, return_masks=True)
#     mask_areas = [np.mean(mask) for mask in masks]

#     if not args.custom_data:
#         idx_to_caption = np.argsort(mask_areas)[int(len(mask_areas) * args.mask_area_percentile)]
#         img_to_caption = imgs[idx_to_caption]
#     else:
#         idx_to_caption = imgs[0]
    
#     prompt = CAPTIONING_PROMPT_DETAIL if args.caption_save_name == 'info_caption_detail' else CAPTIONING_PROMPT
#     with torch.no_grad():
#         caption = generate_text(img_to_caption, vqa_model, vqa_processor, device=args.device)

#     info = {'idx_to_caption': str(idx_to_caption), 'caption': caption} 

#     print('scene: %s, info:' % os.path.basename(scene_dir), info)
#     if show:
#         plt.imshow(img_to_caption)
#         plt.show()

#     save_path = os.path.join(scene_dir, '%s.json' % args.caption_save_name)
    
#     # save info to json
#     with open(save_path, 'w') as f:
#         json.dump(info, f, indent=4)
#         print('Caption saved to:', save_path)

#     return info

def predict_caption(args, scene_dir, vqa_model, vqa_processor, show=False):
    img_dir = os.path.join(scene_dir, 'images')
    imgs, masks = load_images(img_dir, return_masks=True)
    mask_areas = [np.mean(mask) for mask in masks]


    if not args.custom_data:
        idx_to_caption = np.argsort(mask_areas)[int(len(mask_areas) * args.mask_area_percentile)]
        img_to_caption = imgs[idx_to_caption]
        
    else:
        idx_to_caption = imgs[0]

    scene_name = os.path.basename(scene_dir)
    tmp_path = os.path.join(scene_dir, 'images', f'{scene_name}_00.png')

    if 'detail' in args.caption_save_name:
        prompt = CAPTIONING_PROMPT_DETAIL
    else:
        prompt = CAPTIONING_PROMPT

    if 'gpt' in args.caption_save_name:
        # save a temporary image file
        #tmp_path = os.path.join(scene_dir, 'img_to_caption.png')
        #Image.fromarray(img_to_caption).save(tmp_path)
        # wrap GPT-based captioning
        gpt_fn = lambda s: gpt_captioning(tmp_path, prompt=prompt, seed=s)
        # parse_fn: identity, since caption is raw text
        caption = gpt_wrapper(gpt_fn, lambda x: x)
    else:
        with torch.no_grad():
            caption = generate_text(img_to_caption, vqa_model, vqa_processor, prompt=prompt, device=args.device)

    info = {'idx_to_caption': str(idx_to_caption), 'caption': caption}
    print('scene: %s, info:' % os.path.basename(scene_dir), info)
    if show:
        plt.imshow(img_to_caption)
        plt.show()

    save_path = os.path.join(scene_dir, '%s.json' % args.caption_save_name)
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=4)
        print('Caption saved to:', save_path)

    return info


if __name__ == '__main__':

    args = get_args()


    def run_caption():
        caption_info = predict_caption(args, os.path.join(scenes_dir, scene), model, processor)
        if args.delay_time > 0:
            time.sleep(args.delay_time)
        return caption_info

    openai.api_key = OPENAI_API_KEY

    scenes_dir = os.path.join(args.data_dir, 'scenes')
    scenes = get_scenes_list(args)

    model = processor = None
    if not 'gpt' in args.caption_save_name:
        try:
            model, processor = load_blip2(args.blip2_model_dir, device=args.device)
        except:
            model_name = "Salesforce/blip2-flan-t5-xl"
            model, processor = load_blip2(model_name, device=args.device)
    for j, scene in enumerate(tqdm(scenes, desc="Processing scenes")):
        if args.caption_save_name != 'info_new_llm_tn' and args.caption_save_name != 'info_dim':
            current_info_path = os.path.join(scenes_dir, scene, f'{args.caption_save_name}.json')
            if not os.path.exists(current_info_path):
                # #print('scene', scene)
                # caption_info = predict_caption(args, os.path.join(scenes_dir, scene), model, processor)
                # if args.delay_time > 0:
                #     time.sleep(args.delay_time)
                run_caption()
            # else:
            #     current_info = read_json(current_info_path)
            #     print('current info', current_info)
            #     if caption_save_name == 'info_dim' and 'dimension' not in current_info:
            #         run_caption()
            #         print('hehe')
        else:
            info_path = os.path.join(scenes_dir, scene, 'info.json')
            current_info_path = os.path.join(scenes_dir, scene, f'{args.caption_save_name}.json')
            if not os.path.exists(current_info_path):
                shutil.copy(info_path, current_info_path)
                print(f'Copied {info_path} to {current_info_path}')