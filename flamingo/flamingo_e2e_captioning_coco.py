import argparse
import json
import pickle
import numpy as np
from tqdm import tqdm
import torch
from eval_datasets import CaptionDataset
from flamingo_utils import get_flamingo_model_processor_tok
from rice import RICES

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--model_name_or_path",
    type=str,
    default= "openflamingo/OpenFlamingo-3B-vitl-mpt1b",
    help="open flamingo model name or path"
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
train_annotations_path = "../dataset/annotations/captions_train2017.json"
train_image_dir_path = "/scratch/workspace/asureddy_umass_edu-llm_alignment/dataset/train2017"
val_annotations_path = "../dataset/annotations/captions_val2017.json"
val_image_dir_path = "/scratch/workspace/asureddy_umass_edu-llm_alignment/dataset/val2017"
rice_cached_features_path = "/scratch/workspace/asureddy_umass_edu-llm_alignment/features-cache/coco_train.pkl" 

train_dataset = CaptionDataset(train_image_dir_path, train_annotations_path)
val_dataset = CaptionDataset(val_image_dir_path, val_annotations_path)
if rice_cached_features_path:
    with open(rice_cached_features_path, 'rb') as f:
        rice_cached_features = pickle.load(f)

retriever = RICES(train_dataset, 'cpu',1, cached_features=rice_cached_features)
model, image_processor, tokenizer = get_flamingo_model_processor_tok()
tokenizer.padding_side = "left" # For generation padding tokens should be on the left
model = model.to('cuda')

# helper functions
def preprocess_images(ims, image_processor):
    vision_x = [image_processor(im_).unsqueeze(0) for im_ in ims]
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    return vision_x

def tok_query(query):
    """
    Example query:
    "<image>An image of two cats.<|endofchunk|><image>An image of a bathroom sink.<|endofchunk|><image>An image of"
    """
    lang_x = tokenizer(
        [query],
        return_tensors="pt",
    )
    return lang_x

def get_n_shot_demonstrations(item, n=2, use_random=True, n_random=0):
    if n==0: return [[]]
    if use_random:
        train_idxs = list(np.random.choice(len(train_dataset),n))
        icl_demonstrs = [[train_dataset[idx] for idx in train_idxs]]
    else:
        if n_random:
            train_idxs = list(np.random.choice(len(train_dataset),n_random))
            icl_demonstrs = [train_dataset[idx] for idx in train_idxs]
        icl_demonstrs = [icl_demonstrs + retriever.find([item['image']],n-n_random)[0]]
        
    return icl_demonstrs

def construct_captioning_query(query_items, icl_demonstrs_list):
    querys, im_lists = [], []
    for query_item,icl_demonstrs in zip(query_items, icl_demonstrs_list):
        query = "<image> Output: "
        images = []
        for item in icl_demonstrs:
            query += f"{item['caption']} |<endofchunk>| <image> Output: "
            images.append(item['image'])
        images.append(query_item['image'])
        querys.append(query)
        im_lists.append(images)
    return querys, im_lists

def get_output(item, n= 2, use_random= False, n_random=0):

    icl_demonstrs = get_n_shot_demonstrations(item, n, use_random, n_random)
    querys, im_lists = construct_captioning_query([item], icl_demonstrs)
    vision_x = preprocess_images(im_lists[0], image_processor)
    lang_x = tok_query(querys[0])
    # print(querys[0])
    generated_text = model.generate(
        vision_x=vision_x.cuda(),
        lang_x=lang_x["input_ids"].cuda(),
        attention_mask=lang_x["attention_mask"].cuda(),
        max_new_tokens=20,
        num_beams=1,
    )
    # print("Generated text: ", tokenizer.decode(generated_text[0]))
    return tokenizer.decode(generated_text[0])[len(querys[0]):]

if __name__=="__main__":
    out_data = {
        "outputs": []
    }
    n = int(args.n_shots)
    use_random = args.use_random
    n_random = int(args.n_random)
    model_name = model_name_or_path.split("/")[-1]
    save_path = f"/home/asureddy_umass_edu/cs682/flamingo/results/captioning/{model_name}_{n}-shot"
    if use_random:
        save_path += "_random-examples"
    if n_random:
        save_path += f"{n_random}_random-examples"
    save_path += ".json"
    print(args)
    for i in tqdm(range(len(val_dataset))):
        out = get_output(val_dataset[i],n,n_random)
        out_data["outputs"].append(out)
    
    with open(save_path,'w') as f:
        json.dump(out_data, f)