import os

import torch
from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline, DPMSolverMultistepScheduler
# from rtpt import RTPT
from torch import autocast
from transformers import CLIPTextModel, set_seed
import pandas as pd
from argparse import ArgumentParser


# HF_TOKEN = 'INSERT_HF_TOKEN'
# OUTPUT_FOLDER = 'images'
NUM_SAMPLES = 9
set_seed(42)

BASE_SWAP_DIR = "/data2/haikang/projects/cloned/Dialet-Biases-Unlearning/images"
BASE_ORIG_DIR = "/data2/haikang/projects/cloned/Dialet-Biases-Unlearning/images_orig"
DATA_FILE = "/data2/haikang/projects/cloned/Dialet-Biases-Unlearning/data/train_val_test/4-1-1/basic/sge/test.csv"
# dialect type is the second last path of the prompt directory
DIALECT_TYPE = DATA_FILE.split("/")[-2]
# model = "stabilityai/stable-diffusion-2-1"


def main(args):
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
    ).to("cuda")
    
    if args.swap:
        base_dir = BASE_SWAP_DIR
        if "stable-diffusion-2-1" in args.model:
            # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            text_encoder = CLIPTextModel.from_pretrained("models/singlish_sd21", use_safetensors=True, device_map="auto")
            pipe.text_encoder = text_encoder
        if "stable-diffusion-v1-5" in args.model:
            # pipe.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
            text_encoder = CLIPTextModel.from_pretrained("models/singlish", use_safetensors=True, device_map="auto")
            pipe.text_encoder = text_encoder
    else:
        base_dir = BASE_ORIG_DIR
        

    df = pd.read_csv(DATA_FILE, encoding="unicode_escape")
    dialect_prompts = df["Dialect_Prompt"].tolist()
    sae_prompts = df["SAE_Prompt"].tolist()

    for i in range(len(dialect_prompts)):
        dialect_prompt = dialect_prompts[i]
        sae_prompt = sae_prompts[i]
        
        model_base_name = args.model.split("/")[-1]
        prompt_dir = os.path.join(base_dir, model_base_name, DIALECT_TYPE, dialect_prompt)
        os.makedirs(prompt_dir, exist_ok=False)
        prompt_dir = os.path.join(base_dir, model_base_name, "sae", sae_prompt)
        os.makedirs(prompt_dir, exist_ok=False)
        
        for k in range(NUM_SAMPLES):
            ## DIALECT
            with autocast("cuda"):
                image = pipe(dialect_prompt).images[0]
            image_path = os.path.join(base_dir, model_base_name, DIALECT_TYPE, dialect_prompt, f"{k}.jpg")
            image.save(image_path)
            
            ## SAE
            with autocast("cuda"):
                image = pipe(sae_prompt).images[0]
            image_path = os.path.join(base_dir, model_base_name, "sae", sae_prompt, f"{k}.jpg")
            image.save(image_path)


def parse_arguments():
    parser = ArgumentParser(description="Generate images using a stable diffusion model.")
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-2-1", 
                        choices=["stabilityai/stable-diffusion-2-1", "stable-diffusion-v1-5/stable-diffusion-v1-5"])
    parser.add_argument("--swap", action="store_true", help="Swap in the trained text encoder.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
