import json
import os

from PIL import Image
from torch.utils.data import Dataset

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
    
class VqaDataset(Dataset):