from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
import torch
from PIL import Image
import requests

def get_flamingo_model_processor_tok(fl_model = "openflamingo/OpenFlamingo-3B-vitl-mpt1b"):
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
        tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
        cross_attn_every_n_layers=1,
        # cache_dir="/scratch/workspace/asureddy_umass_edu-llm_alignment/hf-cache"  # Defaults to ~/.cache
    )
    # grab model checkpoint from huggingface hub
    checkpoint_path = hf_hub_download(fl_model, "checkpoint.pt",)
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    return model, image_processor, tokenizer

# model, image_processor, tokenizer = get_flamingo_model_processor_tok()
# tokenizer.padding_side = "left" # For generation padding tokens should be on the left

def preprocess_images(ims, image_processor):
    
    vision_x = [image_processor(im_).unsqueeze(0) for im_ in ims]
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)
    return vision_x
