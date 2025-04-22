import os

import torch
from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline, DPMSolverMultistepScheduler
# from rtpt import RTPT
from torch import autocast
from transformers import CLIPTextModel, set_seed
import pandas as pd
from argparse import ArgumentParser
from utils.hf_captions import create_hf_coco_dataset


NUM_SAMPLES = 9
set_seed(42)

# BASE_SWAP_DIR = "/data2/haikang/projects/cloned/Dialet-Biases-Unlearning/images/mscoco_swap"
# BASE_SWAP_DIR = "/data2/haikang/projects/cloned/Dialet-Biases-Unlearning/images/mscoco_swap_kl_image_as_class"
BASE_SWAP_DIR = "/data2/haikang/projects/cloned/Dialet-Biases-Unlearning/images/mscoco_swap_kl_iac_20ep"
BASE_ORIG_DIR = "/data2/haikang/projects/cloned/Dialet-Biases-Unlearning/images/mscoco_orig"


caption_file_path = "/data2/haikang/projects/cloned/Dialet-Biases-Unlearning/data/mscoco/annotations/captions_val2017.json"
image_folder_path = "/data2/haikang/projects/cloned/Dialet-Biases-Unlearning/data/mscoco/val2017"


def main(args):
    mscoco = create_hf_coco_dataset(caption_file_path, image_folder_path).select(range(4950, 5000))
    
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
    ).to("cuda")
    # pipe.safety_checker = None  # disable safety checker if desired
    
    if args.swap:
        base_dir = BASE_SWAP_DIR
        if "stable-diffusion-2-1" in args.model:
            # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            text_encoder = CLIPTextModel.from_pretrained("models/singlish_sd21", use_safetensors=True, device_map="auto")
            # text_encoder = CLIPTextModel.from_pretrained("models/sge/singlish_kl_sd21", use_safetensors=True, device_map="auto")
            pipe.text_encoder = text_encoder
        if "stable-diffusion-v1-5" in args.model:
            # pipe.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
            # text_encoder = CLIPTextModel.from_pretrained("models/singlish", use_safetensors=True, device_map="auto")
            # text_encoder = CLIPTextModel.from_pretrained("models/sge/singlish_kl", use_safetensors=True, device_map="auto")
            # text_encoder = CLIPTextModel.from_pretrained("models/sge/singlish_kl_image_as_class", use_safetensors=True, device_map="auto")
            text_encoder = CLIPTextModel.from_pretrained("models/sge/singlish_kl_iac_20ep", use_safetensors=True, device_map="auto")
            pipe.text_encoder = text_encoder
    else:
        base_dir = BASE_ORIG_DIR
        

    prompts = [ct[0] for ct in mscoco["captions"]]

    for prompt in prompts:
        model_base_name = args.model.split("/")[-1]
        prompt_dir = os.path.join(base_dir, model_base_name, prompt)
        os.makedirs(prompt_dir, exist_ok=False)
        
        for k in range(NUM_SAMPLES):
            with autocast("cuda"):
                image = pipe(prompt).images[0]
            image_path = os.path.join(base_dir, model_base_name, prompt, f"{k}.jpg")
            image.save(image_path)

def parse_arguments():
    parser = ArgumentParser(description="Generate images using a stable diffusion model.")
    parser.add_argument("--model", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5", 
                        choices=["stabilityai/stable-diffusion-2-1", "stable-diffusion-v1-5/stable-diffusion-v1-5"])
    parser.add_argument("--swap", action="store_true", help="Swap in the trained text encoder.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
