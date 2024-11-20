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

class VisionLanguageModel:
    def __init__(self, model_name_or_path, conv_mode="llava_v1"):
        self.model_name_or_path = model_name_or_path
        self.conv_mode = conv_mode
        
        # Load the model components
        self.model_name = get_model_name_from_path(model_name_or_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_name_or_path, self.model_name, None
        )


    def get_output_for_query(self, query, imgs, max_new_tokens=5):
        # Replace placeholders in the query
        query = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, query)
        
        # Prepare the conversation template
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Process images and prepare inputs
        images_tensor = process_images(imgs, self.image_processor, self.model.config).to(
            self.model.device, dtype=torch.float16
        )
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        
        # Stopping criteria
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        # Model generation parameters
        temperature = 0.2
        num_beams = 3
        top_p = 0.95
        
        # Generate output
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=[images_tensor],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        # Decode and process the output
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        return outputs.strip()