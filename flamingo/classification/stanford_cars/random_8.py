n_few_shot = 8

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


test_data = pd.read_pickle("stanford_cars_test_data_1k_samples.pickle")
print(test_data.shape)
test_data.head(3)

random_data = pd.read_pickle("stanford_cars_rice_data_1k_samples.pickle")
print(random_data.shape)
random_data.head(4)

import random

count = 0

test_set = []
for test_idx in range(len(test_data)):

    if test_idx%100 == 0:
        print(f"######################## {test_idx} processed")
    sample = test_data.iloc[test_idx]
    
    target_label = sample['true_class_name']

    # random demonstrations
    random.seed(count)
    indexes = random.sample(list(range(len(random_data))), n_few_shot)
    pick_n_files = []
    pick_n_files_names = []
    for rd_idx in indexes:
        ex_image_path = random_data.iloc[rd_idx]['image_path']
        pick_n_files_names.append(ex_image_path)
        ex_label = random_data.iloc[rd_idx]['true_class_name']
        ex_demo_image = Image.open(ex_image_path)
        pick_n_files.append((ex_demo_image, ex_label))
    
    curr_demo_images = []
    few_shot_query =f""""""
    for k in range(n_few_shot):
        curr_demo_images.append(pick_n_files[k][0])
        few_shot_query += f"<image>\nQuestion: Identify and classify the car in the provided image. Provide the label in the exact format: [Make] [Model] [Year].\nAnswer: {pick_n_files[k][1]} |<endofchunk>|\n\n"
    
    
    # add test sample
    curr_image_path = sample['image_path']
    curr_demo_image = Image.open(curr_image_path)
    curr_demo_images.append(curr_demo_image)
    few_shot_query += "<image>\nQuestion: Identify and classify the car in the provided image. Provide the label in the exact format: [Make] [Model] [Year].\nAnswer:"
    
    test_set.append([test_idx, target_label, pick_n_files_names, few_shot_query, curr_demo_images])
    count += 1


test_df = pd.DataFrame(test_set, columns = ['index', 'target_label', 'pick_n_files_names', 'few_shot_query', 'demo_images'])
print(test_df.shape)
test_df.head(2)

print("######### Running inference")

start = time.time()

all_responses = []
for i in range(len(test_df)):
    if i%200 == 0:
        print(f"############### {i} values are processed")

    sample = test_df.iloc[i]
    curr_generated_response = inference(demo_images=sample['demo_images'], query=sample['few_shot_query'], max_new_tokens=15, verbose=False)
    all_responses.append(curr_generated_response)
    torch.cuda.empty_cache()
end = time.time()

print("############# Total time taken for inference : ", (end-start))

test_df['raw_responses'] = all_responses
test_df.drop(['demo_images'], axis=1).to_pickle(f"random_{n_few_shot}_shot.pickle")
test_df['predicted_label'] = test_df['raw_responses'].apply(lambda x: x.split(" |")[0][1:])
acc = test_df[test_df.apply(lambda x: 1 if (x['predicted_label'] in x['target_label']) else 0, 1) == 1].shape[0]/len(test_df)
print(f"random Accuracy at {n_few_shot} shot: ", acc)
test_df.drop(['demo_images'], axis=1).to_pickle(f"random_{n_few_shot}_shot.pickle")