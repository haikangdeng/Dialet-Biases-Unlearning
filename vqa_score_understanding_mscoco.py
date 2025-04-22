import os
import pandas as pd
from tqdm import tqdm
import t2v_metrics
import json
from argparse import ArgumentParser
from utils.hf_captions import create_hf_coco_dataset

# ------------------------- Configuration -------------------------
# IMG_DIR = "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/image/basic/bre"
# DATA_FILE = "/local1/bryanzhou008/Dialect/multimodal-dialectal-bias/data/text/basic/bre.csv"
# MODELS_TO_EVALUATE = ["stable-diffusion-3.5-large-turbo"]
# IMG_DIR_SWAP = "/data2/haikang/projects/cloned/Dialet-Biases-Unlearning/images/mscoco_swap"
# IMG_DIR_SWAP = "/data2/haikang/projects/cloned/Dialet-Biases-Unlearning/images/mscoco_swap_kl"
# IMG_DIR_SWAP = "/data2/haikang/projects/cloned/Dialet-Biases-Unlearning/images/mscoco_swap_kl_image_as_class"
IMG_DIR_SWAP = "/data2/haikang/projects/cloned/Dialet-Biases-Unlearning/images/mscoco_swap_kl_iac_20ep"
IMG_DIR_ORIG = "/data2/haikang/projects/cloned/Dialet-Biases-Unlearning/images/mscoco_orig"

# DATA_FILE = "/data2/haikang/projects/cloned/Dialet-Biases-Unlearning/data/train_val_test/4-1-1/basic/sge/test.csv"
# MODELS_TO_EVALUATE = ["stable-diffusion-v1-5", "stable-diffusion-2-1"]
MODELS_TO_EVALUATE = ["stable-diffusion-v1-5"]
# ------------------------------------------------------------------

# Initialize the new scoring metric.
scorer = t2v_metrics.VQAScore(model='clip-flant5-xxl')

caption_file_path = "/data2/haikang/projects/cloned/Dialet-Biases-Unlearning/data/mscoco/annotations/captions_val2017.json"
image_folder_path = "/data2/haikang/projects/cloned/Dialet-Biases-Unlearning/data/mscoco/val2017"

def get_average_score(img_dir, model_name, folder, gen_prompt, num_images=9):
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
        score_output = scorer(images=[image_path], texts=[gen_prompt])
        try:
            score = score_output[0][0]
        except TypeError:
            score = score_output
        scores.append(score)
    return float(sum(scores)/len(scores))


def main(args):
    mscoco = create_hf_coco_dataset(caption_file_path, image_folder_path).select(range(4950, 5000))
    prompts = [ct[0] for ct in mscoco["captions"]]
    
    if args.swap:
        img_dir = IMG_DIR_SWAP
    else:
        img_dir = IMG_DIR_ORIG

    results = {model: [] for model in MODELS_TO_EVALUATE}

    for i in tqdm(range(len(prompts)), desc="Processing prompts"):
        prompt = prompts[i]

        # Evaluate dialect images (using SAE prompt as reference).
        for model in MODELS_TO_EVALUATE:
            score = get_average_score(img_dir, model, '', prompt)
            results[model].append(score)
            print(f"Prompt {i} - {model} (mscoco): {score:.4f}")

    print("\n------------------- Final Results -------------------")
    avgs = {}
    for model, scores in results.items():
        avg_score = sum(scores) / len(scores)
        avgs[f'avg_{model}'] = avg_score
        print(f"mscoco total score for {model}: {avg_score:.4f}")
    
    for key, value in avgs.items():
        results[key] = value
    
    output_file = os.path.join(img_dir, "results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, sort_keys=True)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--swap", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
