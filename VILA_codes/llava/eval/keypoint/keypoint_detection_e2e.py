import sys
sys.path.append('/home/asureddy_umass_edu/cs682/VILA_codes/llava/eval/keypoint')
sys.path.append('/home/asureddy_umass_edu/cs682/VILA_codes/llava/eval')
from my_model_utils import VisionLanguageModel
from eval_datasets import KeyPointFaceDataset
from keypoint_utils import get_ic_pp_func
import numpy as np
from tqdm import tqdm
import json
import argparse
import pickle
from rice import RICES


np.random.seed(42)

def parse_args():
    """
    Parse command-line arguments.
    
    :return: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Script for configuring model settings.")

    # Add arguments with default values
    parser.add_argument("--model_name_or_path", type=str, default="Efficient-Large-Model/VILA1.5-13b",
                        help="Path to the model or model name.")
    parser.add_argument("--conv_mode", type=str, default="llava_v1",
                        help="conv mode.")
    parser.add_argument("--style", type=str, default="vqa_style",
                        help="Style configuration.")
    parser.add_argument("--strategy", type=str, default="rice",
                        help="Strategy to use.")
    parser.add_argument("--n_shots", type=int, default=2,
                        help="Number of shots.")
    parser.add_argument("--n_runs", type=int, default=1,
                        help="Number of runs.")

    return parser.parse_args()

# seeding the random number
# np.random.seed()
args = parse_args()
model_name_or_path = args.model_name_or_path 
conv_mode = args.conv_mode 

vlm = VisionLanguageModel(model_name_or_path, conv_mode)

img_dir = "/scratch/workspace/asureddy_umass_edu-llm_alignment/dataset/train2017"
annotations_path = "/home/asureddy_umass_edu/cs682/dataset/annotations/person_keypoints_train2017.json"
train_dataset = KeyPointFaceDataset(img_dir, annotations_path, is_train=True)

img_dir = "/scratch/workspace/asureddy_umass_edu-llm_alignment/dataset/val2017"
annotations_path = "/home/asureddy_umass_edu/cs682/dataset/annotations/person_keypoints_val2017.json"
val_dataset = KeyPointFaceDataset(img_dir, annotations_path, is_train=False)

# RICE cached features
rice_cached_features_path = "/scratch/workspace/asureddy_umass_edu-llm_alignment/features-cache/keypoint_face_coco.pkl"
with open(rice_cached_features_path, 'rb') as f:
    rice_cached_features = pickle.load(f)
# retriever = RICES(train_dataset, 'cpu', 1, cached_features=rice_cached_features)

class Controller:
    def __init__(self, vlm, train_dataset, val_dataset, strategy='random', rice_cached_features=None, query_imgs_constructor_func=None, gt_post_processor_func=None):
        self.vlm = vlm
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.strategy = strategy
        self.retriever = None
        if strategy not in ('rice', 'random'):
            raise ValueError("Strategy must be rice or random!!")
        if strategy=='rice':
            self.retriever = RICES(train_dataset, 'cpu',1, cached_features=rice_cached_features)
        self.query_imgs_constructor_func = query_imgs_constructor_func
        self.gt_post_processor_func = gt_post_processor_func if gt_post_processor_func is not None else lambda x: x
        
    def construct_query_imgs(self, query_item, icl_items_list, **kwargs):
        return self.query_imgs_constructor_func(query_item, icl_items_list)
        
    def get_output_for_n_shots(self, item=None, n=2, **kwargs):
        # print(kwargs)
        key_points_ = list(np.random.choice(train_dataset.must_keypoints, size=n, replace=False)) # n-shot + query
        query_kp = [np.random.choice(train_dataset.must_keypoints)]
        key_points_.append(query_kp[0])
        print(key_points_)
        if self.strategy=='random':
            train_idxs = list(np.random.choice(len(self.train_dataset),n))
            if item is None:
                item = self.val_dataset.get_item_with_support(kwargs['index'],key_points_[-1])
            icl_demonstrs = [self.train_dataset.get_item_with_support(idx,key_points_[i]) for i,idx in enumerate(train_idxs)]
        else:
            if item is None:
                item = self.val_dataset.get_item_with_support(kwargs['index'],key_points_[-1])
            train_idxs = self.retriever.find([item['image']], n, return_indices=True)[0]
            print(train_idxs)
            icl_demonstrs = [self.train_dataset.get_item_with_support(idx,key_points_[i]) for i,idx in enumerate(train_idxs)]
        query, imgs = self.construct_query_imgs(item, icl_demonstrs)
        max_new_tokens = kwargs.get('max_new_tokens',5)
        output = self.vlm.get_output_for_query(query, imgs, max_new_tokens=5)
        res = {
            "val_idx": kwargs['index'],
            "train_idxs": train_idxs,
            "key_points": key_points_[:-1],
            "ground_truth": self.gt_post_processor_func(key_points_[-1]),
            "output": output
        }
        # print(max_new_tokens)
        return res
            

            
if __name__=='__main__':
    print(args)
    style = args.style
    strategy = args.strategy
    model_name = vlm.model_name
    n_shots = args.n_shots
    n_runs = args.n_runs
    save_path = f"/home/asureddy_umass_edu/cs682/VILA_codes/results/keypoint_detection/{style}/{model_name}_{strategy}_{n_shots}_shots.json"
    res = []
    query_imgs_constructor_func, gt_post_processor_func = get_ic_pp_func(style)
    controller = Controller(vlm, 
                            train_dataset, 
                            val_dataset, 
                            strategy = strategy,
                            rice_cached_features = rice_cached_features,
                            query_imgs_constructor_func=query_imgs_constructor_func, 
                            gt_post_processor_func=gt_post_processor_func)
    for idx in tqdm(range(len(val_dataset))):
        for run in range(n_runs):
            out = controller.get_output_for_n_shots(n=n_shots, index=idx)
            res.append(out)
    with open(save_path, 'w') as f:
        json.dump(res,f)