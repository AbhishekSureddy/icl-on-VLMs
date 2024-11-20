import os
import pickle
import random
import json
from tqdm import tqdm
import os.path as osp
import re
from io import BytesIO
import argparse
import numpy as np

import requests
import torch
from PIL import Image

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from eval_datasets import KeyPPDataset


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--model_name_or_path",
    type=str,
    default= "Efficient-Large-Model/VILA1.5-3b",
    help="vila model name or path"
)
arg_parser.add_argument(
    "--n_shots",
    type=int,
    help="number of incontext examples",
)

# arg_parser.add_argument(
#     "--save_path",
#     type=str,
#     help="where to save model outputs",
# )


args = arg_parser.parse_args()

# input args
model_name_or_path = args.model_name_or_path

conv_mode = "llava_v1"

train_image_dir_path = "/scratch/workspace/asureddy_umass_edu-llm_alignment/dataset/train2017"
train_annotations_path = "/home/asureddy_umass_edu/cs682/dataset/annotations/person_keypoints_train2017.json"

val_image_dir_path = "/scratch/workspace/asureddy_umass_edu-llm_alignment/dataset/val2017"
val_annotations_path = "/home/asureddy_umass_edu/cs682/dataset/annotations/person_keypoints_val2017.json"

train_dataset = KeyPPDataset(train_image_dir_path, train_annotations_path, is_train=True)
val_dataset = KeyPPDataset(val_image_dir_path, val_annotations_path)

model_name = get_model_name_from_path(model_name_or_path)

tokenizer, model, image_processor, context_len = load_pretrained_model(model_name_or_path, model_name, None)
image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN



def get_output_for_query(query, imgs, conv_mode=conv_mode,max_new_tokens=5):
    query = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, query)
    # conv_mode = "hermes-2"
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    images_tensor = process_images(imgs, image_processor, model.config).to(model.device, dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    # print(images_tensor.shape)
    temperature = 0.2
    num_beams = 3
    top_p = 0.95
    max_new_tokens = max_new_tokens
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[
                images_tensor,
            ],
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    # print(outputs)
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    # print(outputs)
    return outputs

more_descr = """The nose is a prominent feature located in the center of the face, serving as the primary organ for breathing and smell.
It typically sits between the eyes, above the mouth, and forms a central part of facial structure and symmetry. 
I will give a series of example images and corresponding output. The output would be visible if the image containsa person's Nose, 
else the output would be not visible. Use the examples and answer for the last image"""
# more_descr = ""
def get_images_query(kp_yes_samples, kp_no_samples, item, kp_name='nose', verbose=0, more_descr= more_descr):
    images = []
    query = ""
    query += more_descr
    all_samples = kp_yes_samples+kp_no_samples
    all_samples = random.sample(all_samples, len(all_samples))
    # print(all_samples)
    for i in range(len(all_samples)):
        example = train_dataset[all_samples[i]]
        img = example['image']
        visibility = example['kp_visibility']
        images.append(img)
        query += f"<image> Output: {visibility}"
        
        
    query += f"<image> Output:"
    example = item
    img = example['image']
    images.append(img)
    
    if verbose:
        print(f"generated query: {query}")
    return query, images


def get_output(item, n= 2):

    kp_yes_samples = train_dataset.sample_idxs(n//2)
    kp_no_samples = train_dataset.sample_idxs(n//2, False)
    query, im_list = get_images_query(kp_yes_samples, kp_no_samples, item)
    
    return get_output_for_query(query, im_list, max_new_tokens=5)

if __name__=="__main__":
    out_data = {
        "yes": [],
        "no": []
    }
    n = int(args.n_shots)
    model_name = model_name_or_path.split("/")[-1]
    save_path = f"/home/asureddy_umass_edu/cs682/VILA/results/keypoint/{model_name}_{n}-shot"
    save_path += ".json"

    for i in tqdm(range(len(val_dataset.yes_samples))):
        idx = val_dataset.yes_samples[i]
        out = get_output(val_dataset[idx],n)
        out_data["yes"].append({
            "idx": idx,
            "output": out
        })
        # print(out)
    for i in tqdm(range(len(val_dataset.no_samples))):
        idx = val_dataset.no_samples[i]
        out = get_output(val_dataset[idx],n)
        out_data["no"].append({
            "idx": idx,
            "output": out
        })

    with open(save_path,'w') as f:
        json.dump(out_data, f)