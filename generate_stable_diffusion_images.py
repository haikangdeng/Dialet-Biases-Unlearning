import os

import torch
from diffusers import LMSDiscreteScheduler, StableDiffusionPipeline
from rtpt import RTPT
from torch import autocast
from transformers import CLIPTextModel, set_seed
import pandas as pd

# HF_TOKEN = 'INSERT_HF_TOKEN'
# OUTPUT_FOLDER = 'images'
NUM_SAMPLES = 9
set_seed(1)

BASE_DIR = "/data2/haikang/projects/cloned/Dialet-Biases-Unlearning/images"
DATA_FILE = "/data2/haikang/projects/cloned/Dialet-Biases-Unlearning/data/train_val_test/4-1-1/basic/sge/test.csv"
# dialect type is the second last path of the prompt directory
DIALECT_TYPE = DATA_FILE.split("/")[-2]
model_name = "runwayml/stable-diffusion-v1-5"
model_base_name = model_name.split("/")[-1]

def main():
    lms = LMSDiscreteScheduler(beta_start=0.00085,
                               beta_end=0.012,
                               beta_schedule="scaled_linear")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        scheduler=lms,
        # use_auth_token=HF_TOKEN
    ).to("cuda")
    
    # text_encoder = CLIPTextModel.from_pretrained("models/indian", use_safetensors=True, device_map="auto")
    text_encoder = CLIPTextModel.from_pretrained("models/singlish", use_safetensors=True, device_map="auto")
    pipe.text_encoder = text_encoder


    df = pd.read_csv(DATA_FILE, encoding="unicode_escape")
    dialect_prompts = df["Dialect_Prompt"].tolist()
    sae_prompts = df["SAE_Prompt"].tolist()

    for i in range(len(dialect_prompts)):
        dialect_prompt = dialect_prompts[i]
        sae_prompt = sae_prompts[i]
        
        prompt_dir = os.path.join(BASE_DIR, model_base_name, DIALECT_TYPE, dialect_prompt)
        os.makedirs(prompt_dir, exist_ok=True)
        prompt_dir = os.path.join(BASE_DIR, model_base_name, "sae", sae_prompt)
        os.makedirs(prompt_dir, exist_ok=True)
        
        for k in range(NUM_SAMPLES):
            ## DIALECT
            with autocast("cuda"):
                image = pipe(dialect_prompt, num_inference_steps=100).images[0]
            image_path = os.path.join(BASE_DIR, model_base_name, DIALECT_TYPE, dialect_prompt, f"{k}.jpg")
            image.save(image_path)
            
            ## SAE
            with autocast("cuda"):
                image = pipe(sae_prompt, num_inference_steps=100).images[0]
            image_path = os.path.join(BASE_DIR, model_base_name, "sae", sae_prompt, f"{k}.jpg")
            image.save(image_path)
            
        
    # prompt_sae = 'an abandoned temple'
    # prompt_dialect = 'an abandoned mandapa'
    # prompt_sae = 'a child holding an ang pow'
    # prompt_dialect = 'a child holding a red packet'
    
    # general_prompts = [
    #     "A giant dinosaur frozen into a glacier and recently discovered by scientists, cinematic still",
    #     "A cute puppy leading a session of the United Nations, newspaper photography",
    #     "A towering hurricane of rainbow colors towering over a city, cinematic digital art",
    #     "A redwood tree rising up out of the ocean"
    # ]


if __name__ == "__main__":
    main()
