import json
import os

from PIL import Image
from torch.utils.data import Dataset
import random

class CaptionDataset(Dataset):
    def __init__(
        self,
        image_dir_path,
        annotations_path
    ):
        self.image_dir_path = image_dir_path
        self.data = None
        self.dataset_name = "coco"

        full_data = json.load(open(annotations_path))

        data = {}
        for item in full_data["images"]:
            data[item["id"]] = {"id": item["id"] ,"file_name": item["file_name"], "captions": []}
        
        for item in full_data["annotations"]:
            data[item['image_id']]["captions"].append(item["caption"])
        
        self.data = list(data.values())
        self.data_dict = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(
            os.path.join(
                self.image_dir_path, self.data[idx]["file_name"]
            )
        )
        
        caption = self.data[idx]["captions"][0]
        return {
            "image": image,
            "caption": caption,
            "image_id": self.data[idx]["id"]
        }
    
    def get_item_with_idx(self, idx):
        image = Image.open(
            os.path.join(
                self.image_dir_path, self.data[idx]["file_name"]
            )
        )
        
        caption = self.data[idx]["captions"]
        return {
            "image": image,
            "captions": caption,
            "image_id": self.data[idx]["id"]
        }

    def get_item(self, image_id, return_all_captions=True):
        image = Image.open(
            os.path.join(
                self.image_dir_path, self.data_dict[image_id]["file_name"]
            )
        )
        if not return_all_captions:
            caption = self.data_dict[image_id]["captions"][0]
        else:
            caption = self.data_dict[image_id]["captions"]
        return {
            "image": image,
            "caption": caption,
            "image_id": self.data_dict[image_id]["id"]
        }
    
class VQADataset(Dataset):
    def __init__(
        self, image_dir_path, question_path, annotations_path, is_train, dataset_name, take_unique_image=True
    ):
        self.questions = json.load(open(question_path, "r"))["questions"]
        if annotations_path is not None:
            self.answers = json.load(open(annotations_path, "r"))["annotations"]
        else:
            self.answers = None
        self.image_dir_path = image_dir_path
        self.is_train = is_train
        self.dataset_name = dataset_name
        if self.dataset_name in {"vqav2", "ok_vqa"}:
            self.img_coco_split = self.image_dir_path.strip("/").split("/")[-1]
            assert self.img_coco_split in {"train2014", "val2014", "test2015"}
        if take_unique_image:
            seen_image_ids = set()
            questions = []
            answers = []
            for question, answer in zip(self.questions,self.answers):
                if question["image_id"] not in seen_image_ids:
                    seen_image_ids.add(question["image_id"])
                    questions.append(question)
                    answers.append(answer)
            self.questions = questions
            self.answers = answers
    def __len__(self):
        return len(self.questions)

    def get_img_path(self, question):
        if self.dataset_name in {"vqav2", "ok_vqa"}:
            return os.path.join(
                self.image_dir_path,
                f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg"
                if self.is_train
                else f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg",
            )
        elif self.dataset_name == "vizwiz":
            return os.path.join(self.image_dir_path, question["image_id"])
        elif self.dataset_name == "textvqa":
            return os.path.join(self.image_dir_path, f"{question['image_id']}.jpg")
        else:
            raise Exception(f"Unknown VQA dataset {self.dataset_name}")

    def __getitem__(self, idx):
        question = self.questions[idx]
        img_path = self.get_img_path(question)
        image = Image.open(img_path)
        image.load()
        results = {
            "image": image,
            "question": question["question"],
            "question_id": question["question_id"],
        }
        if self.answers is not None:
            answers = self.answers[idx]
            results["answers"] = [a["answer"] for a in answers["answers"]]
        return results
    
class KeyPPDataset(Dataset):
    def __init__(self, img_dir, annotations_path, kp_chosen="nose", is_train=False):
        self.full_data = json.load(open(annotations_path))
        
        image_ids_counter = {}
        for idx,item in enumerate(self.full_data['annotations']):
            image_ids_counter[item['image_id']] = image_ids_counter.get(item['image_id'], [])+[idx]
        good_keypoint_idxs = [x[0] for k,x in image_ids_counter.items() if len(x)==1]
        data = [self.full_data['annotations'][x] for x in good_keypoint_idxs]
        self.full_data['annotations'] = data
        
        self.img_dir = img_dir
        self.keypoints = ['nose',
                   'left_eye',
                   'right_eye',
                   'left_ear',
                   'right_ear',
                   'left_shoulder',
                   'right_shoulder',
                   'left_elbow',
                   'right_elbow',
                   'left_wrist',
                   'right_wrist',
                   'left_hip',
                   'right_hip',
                   'left_knee',
                   'right_knee',
                   'left_ankle',
                   'right_ankle']
        self.kp_chosen = kp_chosen
        self.is_train = is_train
        if kp_chosen not in self.keypoints:
            raise ValueError(f"Invalid keypoint: {kp_chosen}; available options: {self.keypoints}")
        # if self.is_train:
        self.yes_samples = [idx for idx,ann in enumerate(self.full_data['annotations']) if self._preprocess_kp(ann['keypoints'])=="visible"]
        self.no_samples = [idx for idx,ann in enumerate(self.full_data['annotations']) if self._preprocess_kp(ann['keypoints'])!="visible"]

    def sample_idxs(self,n,is_yes_samples=True):
        if is_yes_samples:
            return random.sample(self.yes_samples, n)
        else:
            return random.sample(self.no_samples, n)
            
    def _preprocess_kp(self, annotation):
        kp_annotation = {}
        for idx, kp in enumerate(self.keypoints):
            x,y,v = 3*idx, 3*idx+1, 3*idx+2
            kp_annotation[kp] = (annotation[x],annotation[y],annotation[v])
        kp_req = kp_annotation[self.kp_chosen]
        return "visible" if kp_req[-1]==2 else "not visible"

    def __len__(self):
        return len(self.full_data['annotations'])

    def get_img_path(self, image_id):
        return os.path.join(self.img_dir, f"{image_id:012d}.jpg")

    def __getitem__(self, idx):
        img_path = self.get_img_path(self.full_data['annotations'][idx]['image_id'])
        image = Image.open(img_path)
        image.load()
        results = {
            "image_id": self.full_data['annotations'][idx]['image_id'],
            "kp_visibility": self._preprocess_kp(self.full_data['annotations'][idx]['keypoints']),
            "image": image,
            'iscrowd': self.full_data['annotations'][idx]['iscrowd'],
            'area': self.full_data['annotations'][idx]['area']
        }
        return results
        