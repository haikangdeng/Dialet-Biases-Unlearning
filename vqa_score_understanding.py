import os
import pandas as pd
from tqdm import tqdm
import t2v_metrics
import json
from argparse import ArgumentParser

# ------------------------- Configuration -------------------------
# IMG_DIR = "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/image/basic/bre"
# DATA_FILE = "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/text/basic/bre.csv"
# MODELS_TO_EVALUATE = ["stable-diffusion-3.5-large-turbo"]
IMG_DIR_SWAP = "/data2/haikang/projects/cloned/Dialet-Biases-Unlearning/images"
IMG_DIR_ORIG = "/data2/haikang/projects/cloned/Dialet-Biases-Unlearning/images_orig"
DATA_FILE = "/data2/haikang/projects/cloned/Dialet-Biases-Unlearning/data/train_val_test/4-1-1/basic/sge/test.csv"
MODELS_TO_EVALUATE = ["stable-diffusion-v1-5", "stable-diffusion-2-1"]
# ------------------------------------------------------------------

# Initialize the new scoring metric.
scorer = t2v_metrics.VQAScore(model='clip-flant5-xxl')

def get_average_score(img_dir, model_name, folder, gen_prompt, ref_prompt, num_images=9):
    """
    Compute the average similarity score for a set of generated images using the new metric.
    """
    prompt_dir = os.path.join(img_dir, model_name, folder, gen_prompt)
    scores = []

    for i in range(num_images):
        image_path = os.path.join(prompt_dir, f"{i}.jpg")
        if not os.path.exists(image_path):
            # Handle filename inconsistencies.
            processed_prompt = gen_prompt.replace("'", "_")
            prompt_dir = os.path.join(img_dir, model_name, folder, processed_prompt)
            image_path = os.path.join(prompt_dir, f"{i}.jpg")

        # Compute the score for the (image, text) pair.
        score_output = scorer(images=[image_path], texts=[ref_prompt])
        try:
            score = score_output[0][0]
        except TypeError:
            score = score_output
        scores.append(score)

    return float(sum(scores)/len(scores))

def main(args):
    df = pd.read_csv(DATA_FILE, encoding="unicode_escape")
    dialect_prompts = df["Dialect_Prompt"].tolist()
    sae_prompts = df["SAE_Prompt"].tolist()
    
    if args.swap:
        img_dir = IMG_DIR_SWAP
    else:
        img_dir = IMG_DIR_ORIG

    results = {
        "dialect": {model: [] for model in MODELS_TO_EVALUATE},
        "sae": {model: [] for model in MODELS_TO_EVALUATE},
    }

    for i in tqdm(range(len(dialect_prompts)), desc="Processing prompts"):
        dialect_prompt = dialect_prompts[i]
        sae_prompt = sae_prompts[i]

        # Evaluate dialect images (using SAE prompt as reference).
        for model in MODELS_TO_EVALUATE:
            score = get_average_score(img_dir, model, DATA_FILE.split("/")[-2], dialect_prompt, sae_prompt)
            results["dialect"][model].append(score)
            print(f"Prompt {i} - {model} (dialect): {score:.4f}")

        # Evaluate SAE images (using SAE prompt for both generated and reference).
        for model in MODELS_TO_EVALUATE:
            score = get_average_score(img_dir, model, "sae", sae_prompt, sae_prompt)
            results["sae"][model].append(score)
            print(f"Prompt {i} - {model} (sae): {score:.4f}")

    print("\n------------------- Final Results -------------------")
    for set_type, model_scores in results.items():
        for model, scores in model_scores.items():
            avg_score = sum(scores) / len(scores)
            print(f"{set_type.capitalize()} total score for {model}: {avg_score:.4f}")
    
    output_file = os.path.join(img_dir, "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--swap", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
