import os
import pickle
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
from eval_datasets import VQADataset
from rice import RICES


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
arg_parser.add_argument(
    "--use_random",
    action="store_true",
    help="Pass in a list of MMC4 shards in the format path_to_shard/shard_{0..23098}.zip",
)
arg_parser.add_argument(
    "--n_random",
    type=int,
    default=0,
    help="number of random incontext examples",
)
# arg_parser.add_argument(
#     "--save_path",
#     type=str,
#     help="where to save model outputs",
# )


args = arg_parser.parse_args()

# input args
model_name_or_path = args.model_name_or_path

train_image_dir_path = "/scratch/workspace/asureddy_umass_edu-llm_alignment/dataset/vqa/train2014"
train_questions_path = "/scratch/workspace/asureddy_umass_edu-llm_alignment/dataset/vqa/v2_OpenEnded_mscoco_train2014_questions.json"
train_annotations_path = "/scratch/workspace/asureddy_umass_edu-llm_alignment/dataset/vqa/v2_mscoco_train2014_annotations.json"

val_image_dir_path = "/scratch/workspace/asureddy_umass_edu-llm_alignment/dataset/vqa/val2014"
val_questions_path = "/scratch/workspace/asureddy_umass_edu-llm_alignment/dataset/vqa/v2_OpenEnded_mscoco_val2014_questions.json"
val_annotations_path = "/scratch/workspace/asureddy_umass_edu-llm_alignment/dataset/vqa/v2_mscoco_val2014_annotations.json"

# dataset = VQADataset(image_dir_path, questions_path, annotations_path,True, "vqav2")

rice_cached_features_path = "/scratch/workspace/asureddy_umass_edu-llm_alignment/features-cache/coco_train_2014.pkl" 
train_dataset = VQADataset(train_image_dir_path, train_questions_path, train_annotations_path,True, "vqav2")
val_dataset = VQADataset(val_image_dir_path, val_questions_path, val_annotations_path,False, "vqav2")
if rice_cached_features_path:
    with open(rice_cached_features_path, 'rb') as f:
        rice_cached_features = pickle.load(f)

retriever = RICES(train_dataset, 'cpu',1, cached_features=rice_cached_features)

model_name = get_model_name_from_path(model_name_or_path)

tokenizer, model, image_processor, context_len = load_pretrained_model(model_name_or_path, model_name, None)
image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

# conv_mode = "hermes-2" # for vila-40b
# llava_v1 for vila-13b
# vicuna_v1 for vila-3b
def get_output_for_query(query, imgs, conv_mode="vicuna_v1",max_new_tokens=5):
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

def get_n_shot_demonstrations(item, n=2, use_random=True, n_random=0):
    if n==0: return [[]]
    if use_random:
        train_idxs = list(np.random.choice(len(train_dataset),n))
        icl_demonstrs = [[train_dataset[idx] for idx in train_idxs]]
    else:
        icl_demonstrs = []
        if n_random:
            train_idxs = list(np.random.choice(len(train_dataset),n_random))
            icl_demonstrs = [train_dataset[idx] for idx in train_idxs]
        icl_demonstrs = [icl_demonstrs + retriever.find([item['image']],n-n_random)[0]]
        
    return icl_demonstrs

def construct_vqa_query(query_items, icl_demonstrs_list):
    querys, im_lists = [], []
    for query_item,icl_demonstrs in zip(query_items, icl_demonstrs_list):
        # query = "Answer the questions in one or two words: "
        query = ""
        images = []
        for item in icl_demonstrs:
            query += f" <image> Question: {item['question']} Short Answer: {item['answers'][0]}"
            images.append(item['image'])
        images.append(query_item['image'])
        query += f"<image> Question: {query_item['question']} Short Answer: "
        querys.append(query)
        im_lists.append(images)
    return querys, im_lists

def get_output(item, n= 2, use_random= False, n_random=0):

    icl_demonstrs = get_n_shot_demonstrations(item, n, use_random, n_random)
    querys, im_lists = construct_vqa_query([item], icl_demonstrs)
    
    return get_output_for_query(querys[0], im_lists[0],"vicuna_v1",5)

if __name__=="__main__":
    out_data = {
        "outputs": []
    }
    n = int(args.n_shots)
    use_random = args.use_random
    n_random = int(args.n_random)
    model_name = model_name_or_path.split("/")[-1]
    save_path = f"/home/asureddy_umass_edu/cs682/VILA/results/vqa/{model_name}_{n}-shot"
    if use_random:
        save_path += "_random-examples"
    if n_random:
        save_path += f"{n_random}_random-examples"
    save_path += ".json"
    print("new vqa_coco vila 3b")
    print(args)
    # doing for a max of 10k examples
    for i in tqdm(range(min(5000, len(val_dataset)))):
        out = get_output(val_dataset[i],n,use_random, n_random)
        out_data["outputs"].append(out)
    
    with open(save_path,'w') as f:
        json.dump(out_data, f)