n_few_shot = 4
import argparse
from open_flamingo import create_model_and_transforms
# grab model checkpoint from huggingface hub
from huggingface_hub import hf_hub_download
import os
import pandas as pd
import torch
import random
from PIL import Image
import time

def inference(demo_images, query, max_new_tokens=20, verbose=True):
    """
    Step 2: Preprocessing images
    Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
    batch_size x num_media x num_frames x channels x height x width. 
    In this case batch_size = 1, num_media = 3, num_frames = 1,
    channels = 3, height = 224, width = 224.
    """
    vision_x = []

    for img in demo_images:
        vision_x.append(image_processor(img).unsqueeze(0))

    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    
    """
    Step 3: Preprocessing text
    Details: In the text we expect an <image> special token to indicate where an image is.
    We also expect an <|endofchunk|> special token to indicate the end of the text 
    portion associated with an image.
    """
    tokenizer.padding_side = "left" # For generation padding tokens should be on the left
    lang_x = tokenizer(
        [query],
        return_tensors="pt",
    )

    """
    Step 4: Generate text
    """
    generated_text = model.generate(
        vision_x=vision_x.cuda(),
        lang_x=lang_x["input_ids"].cuda(),
        attention_mask=lang_x["attention_mask"].cuda(),
        max_new_tokens=max_new_tokens,
        num_beams=1,
    )
    torch.cuda.empty_cache()

    generated_text = tokenizer.decode(generated_text[0][len(lang_x["input_ids"][0]):])
    if verbose:
        print("### Generated text: ", generated_text)
    return generated_text


model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1,
    # cache_dir="/scratch/workspace/asureddy_umass_edu-llm_alignment/hf-cache"  # Defaults to ~/.cache
    )
    
checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path), strict=False)
model.to("cuda")

print("Model Loaded")


val_dir = "/scratch/workspace/dsaluru_umass_edu-email/imagenet/imagenet/val"
random.seed(42)

with open("/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/durga_sandeep/icl-on-VLMs/flamingo/classification/imagenet/LOC_synset_mapping.txt", 'r') as f:
    mapping = f.readlines()
file_to_label_dict = {x.split(" ")[0] : (" ".join(x.replace("\n", "").lower().split(" ")[1:])).split(", ") for x in mapping}
all_folder_names = list(file_to_label_dict.keys())

final_folder_names = random.sample(all_folder_names, 200)
final_folder_names = final_folder_names[:100]
final_folder_names[:10]


folder_to_files = {}
folder_to_other_folders = {}

for folder in final_folder_names:
    folder_dir = f"{val_dir}/{folder}/"
    other_folders = [x for x in final_folder_names if x != folder]
    folder_to_other_folders[folder] = other_folders
    all_folder_files = os.listdir(folder_dir)
    folder_to_files[folder] = random.sample(all_folder_files, 10)

count = 0


print("######### Getting few-shot Prompt")
test_set = []

for folder in final_folder_names:
    other_folders = folder_to_other_folders[folder]
    target_label = file_to_label_dict[folder]
    for idx in range(10):
        if count%100==0:
            print(count)
        random.seed(count+1)
        pick_n_folders = random.sample(other_folders, n_few_shot)
        pick_n_files = []
        pick_n_files_names = []
        for ele in pick_n_folders:
            ex_file_name = random.sample(os.listdir(f"{val_dir}/{ele}/"), 1)[0]
            ex_label = file_to_label_dict[ele]

            ex_image_path = f"{val_dir}/{ele}/{ex_file_name}"
            ex_demo_image = Image.open(ex_image_path)
            pick_n_files.append((ex_demo_image, ex_label[0]))
            pick_n_files_names.append(ex_file_name)
        
        curr_demo_images = []
        few_shot_query =f""""""
        for k in range(n_few_shot):
            curr_demo_images.append(pick_n_files[k][0])
            few_shot_query += f"<image>\nQuestion: Classify the image into one of the imagenet1k label.\nAnswer: {pick_n_files[k][1]} |<endofchunk>|\n\n"
        
        
        # add test sample
        curr_image_path = f"{val_dir}/{folder}/{folder_to_files[folder][idx]}"
        curr_demo_image = Image.open(curr_image_path)
        curr_demo_images.append(curr_demo_image)
        few_shot_query += "<image>\nQuestion: Classify the image into one of the imagenet1k label.\nAnswer:"
        
        test_set.append([folder, target_label, folder_to_files[folder][idx], pick_n_folders, pick_n_files_names, few_shot_query, curr_demo_images])
        count += 1
print("Length of test data : ", count)

test_df = pd.DataFrame(test_set, columns = ['folder', 'target_label', "filename", 'pick_n_folders', 'pick_n_files_names', 'few_shot_query', 'demo_images'])


print("######### Running inference")

start = time.time()

all_responses = []
for i in range(len(test_df)):
    if i%200 == 0:
        print(f"############### {i} values are processed")

    sample = test_df.iloc[i]
    curr_generated_response = inference(demo_images=sample['demo_images'], query=sample['few_shot_query'], max_new_tokens=10, verbose=False)
    all_responses.append(curr_generated_response)
    torch.cuda.empty_cache()
end = time.time()

print("############# Total time taken for inference : ", (end-start))

test_df['raw_responses'] = all_responses
test_df['predicted_label'] = test_df['raw_responses'].apply(lambda x: x.split(" |")[0][1:])
acc = test_df[test_df.apply(lambda x: 1 if (x['predicted_label'] in x['target_label']) else 0, 1) == 1].shape[0]/len(test_df)
print(f"Random - Imagenet Accuracy at {n_few_shot} shot: ", acc)

