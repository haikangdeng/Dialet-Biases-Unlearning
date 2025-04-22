import argparse
import os
from datetime import datetime
import torch
import torch.nn.functional as F
import wandb
from utils.config_parser import ConfigParser
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd
from transformers import set_seed, CLIPProcessor, CLIPModel
from torch.nn import KLDivLoss
from utils.hf_captions import create_hf_coco_dataset

# os.environ["TOKENIZERS_PARALLELISM"] = "true"


def clip_inference(clip_model, clip_processor, caption_dataset_batch):
    text = [ct[0] for ct in caption_dataset_batch["captions"]]
    images = caption_dataset_batch["image"]
    inputs = clip_processor(text=text, images=images, return_tensors="pt", padding=True).to(clip_model.device)
    with torch.no_grad():
        outputs = clip_model(**inputs, return_dict=True)
    return outputs.logits_per_image, outputs.image_embeds


def main(args):
    config = ConfigParser(args.config)
    config_path = args.config
    set_seed(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(config.training.get('num_threads', 4))

    # --- Load Dialect Data ---
    dataset = process_dialect_data(config.dialect_file_folder, args.dialect)
    dataset = dataset.shuffle(seed=config.seed)

    # --- Load Caption Dataset for Regularization ---
    caption_file_path = config.caption_file_path
    image_folder_path = config.image_folder_path
    kl_batch_size = config.kl_batch_size
    kl_control_size = config.kl_control_size
    caption_dataset = create_hf_coco_dataset(caption_file_path, image_folder_path).select(range(kl_control_size))
    
    clip_processor = CLIPProcessor.from_pretrained(config.clip_model)
    # clip_model = CLIPModel.from_pretrained(config.clip_model, torch_dtype=torch.float16).to(device)
    clip_model = CLIPModel.from_pretrained(config.clip_model).to(device)
    for param in clip_model.parameters():
        param.requires_grad = False
    
    reference_logits, image_embeds = clip_inference(clip_model, clip_processor, caption_dataset)
    print(f"image_embeds shape: {image_embeds.shape}")

    # --- Load Models ---
    encoder_reference, tokenizer = config.load_encoder_and_tokenizer()
    encoder_reference = encoder_reference.to(device)
    encoder_policy, _ = config.load_encoder_and_tokenizer()
    encoder_policy = encoder_policy.to(device)

    # Freeze reference model
    for param in encoder_reference.parameters():
        param.requires_grad = False

    # --- Optimizer and Scheduler ---
    optimizer = config.create_optimizer(encoder_policy)
    lr_scheduler = config.create_lr_scheduler(optimizer)

    # --- Loss Functions ---
    loss_fkt = config.loss_fkt
    kl_loss_fn = KLDivLoss(reduction='batchmean', log_target=True)
    
    alpha = config.training.get('alpha_sae_reg', 1.0)
    beta = config.training.get('beta_dialect_reg', 0.1)
    gamma = config.training.get('kl_weight', 1)

    # --- WandB Logging ---
    if config.wandb['enable_logging']:
        wandb_run = wandb.init(**config.wandb['args'])
        wandb.save(config_path, policy='now')
        wandb.watch(encoder_policy)
        # Log more config details
        wandb.config.update({
            'optimizer_type': type(optimizer).__name__,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'weight_decay': optimizer.param_groups[0]['weight_decay'],
            'training_epochs': config.training['epochs'],
            'seed': config.seed,
            'dialect': args.dialect,
            'alpha_sae_reg': alpha,
            'beta_dialect_reg': beta,
            'gamma_kl_weight': gamma,
            'kl_batch_size': kl_batch_size,
            'kl_control_size': kl_control_size
        })

    # --- Training Preparation ---
    step = 0
    encoder_policy.train()
    encoder_reference.eval()

    # --- Training Loop ---
    print(f'Starting Training for {config.training["epochs"]} epochs...')
    for ep in range(config.training['epochs']):
        print(f"\n--- Epoch {ep+1}/{config.training['epochs']} ---")
        encoder_policy.train() # Set model to train mode each epoch
        total_epoch_loss = 0.0
        total_epoch_unlearn_loss = 0.0
        total_epoch_kl_loss = 0.0
        total_epoch_sae_reg_loss = 0.0
        total_epoch_dialect_reg_loss = 0.0

        num_batches = int(np.ceil(len(dataset["train"])/config.clean_batch_size))

        for i in range(num_batches):
            batch = dataset["train"][i*config.clean_batch_size:(i+1)*config.clean_batch_size]
            
            batch_sae_prompt = batch['sae_prompts']
            batch_dialect_prompt = batch['dialect_prompts']

            sae_input = tokenizer(
                batch_sae_prompt, padding="max_length", max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt"
            ).to(device)
            dialect_input = tokenizer(
                batch_dialect_prompt, padding="max_length", max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt"
            ).to(device)

            # --- Forward Pass ---
            embed_policy_sae = encoder_policy(sae_input.input_ids)[0] # Assuming [0] gives final embeddings/logits
            embed_policy_dialect = encoder_policy(dialect_input.input_ids)[0]
            
            with torch.no_grad():
                embed_reference_sae = encoder_reference(sae_input.input_ids)[0]
                embed_reference_dialect = encoder_reference(dialect_input.input_ids)[0]

            loss_unlearn = loss_fkt(embed_policy_sae, embed_policy_dialect)
            loss_reg_sae = loss_fkt(embed_policy_sae, embed_reference_sae)
            loss_reg_dialect = loss_fkt(embed_policy_dialect, embed_reference_dialect)

            # --- KL Divergence Loss Calculation ---
            # clip_model.text_model = encoder_policy
            # clip_model = clip_model.to(device)
            # policy_logits = clip_inference(clip_model, clip_processor, caption_dataset, caption_dataloader)
 
            text_embeds = []
            for i in range(int(np.ceil(kl_control_size/kl_batch_size))):
                batch = caption_dataset[i*kl_batch_size:(i+1)*kl_batch_size]
                ts = [ct[0] for ct in batch["captions"]]
                inputs = tokenizer(
                    ts, return_tensors="pt", padding=True, truncation=True
                ).to(device)
                text_output = encoder_policy(**inputs)[1]
                text_embeds.append(text_output)
            
            text_embeds = torch.cat(text_embeds, dim=0)
            text_embeds = clip_model.text_projection(text_embeds)
            # print(text_embeds)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            
            print(f'text_embeds shape: {text_embeds.shape}')

            # cosine similarity as logits
            logit_scale = clip_model.logit_scale.exp()
            logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
            logits_per_image = logits_per_text.T
            
            # kl_dim = 1    # for row-wise kl (per image)   # default
            kl_dim = 0     # for column-wise kl (per caption)
            
            loss_kl_reg = kl_loss_fn(
                F.log_softmax(logits_per_image, dim=kl_dim),
                F.log_softmax(reference_logits, dim=kl_dim)
            )
            
            if logits_per_image.isnan().any():
                print("NaN detected in logits_per_image")
                exit()
                

            loss = loss_unlearn + alpha * loss_reg_sae + beta * loss_reg_dialect + gamma * loss_kl_reg

            # --- Backpropagation and Optimization ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- Learning Rate Scheduling ---
            if lr_scheduler:
                lr_scheduler.step() # Check if scheduler steps per batch or per epoch

            # --- Logging ---
            loss_val = loss.item()
            loss_unlearn_val = loss_unlearn.item()
            loss_kl_reg_val = loss_kl_reg.item()
            loss_reg_sae_val = loss_reg_sae.item()
            loss_reg_dialect_val = loss_reg_dialect.item()

            total_epoch_loss += loss_val
            total_epoch_unlearn_loss += loss_unlearn_val
            total_epoch_kl_loss += loss_kl_reg_val
            total_epoch_sae_reg_loss += loss_reg_sae_val
            total_epoch_dialect_reg_loss += loss_reg_dialect_val

            print(
                f'Epoch {ep+1}/{config.training["epochs"]} | Step {step} | Batch {i+1}/{num_batches} | '
                f'LR: {optimizer.param_groups[0]["lr"]:.2e} | '
                f'Loss: {loss_val:.4f} | Unlearn: {loss_unlearn_val:.4f} | '
                f'KL Reg: {loss_kl_reg_val:.4f} | SAE Reg: {loss_reg_sae_val:.4f} | Dialect Reg: {loss_reg_dialect_val:.4f}'
            )
            if config.wandb['enable_logging']:
                wandb.log({
                    'step': step,
                    'epoch': ep + (i / num_batches), # Fractional epoch
                    'train_loss': loss_val,
                    'train_loss_unlearn': loss_unlearn_val,
                    'train_loss_kl_reg': loss_kl_reg_val,
                    'train_loss_reg_sae': loss_reg_sae_val,
                    'train_loss_reg_dialect': loss_reg_dialect_val,
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'alpha_sae_reg': alpha,
                    'beta_dialect_reg': beta,
                    'gamma_kl_weight': gamma,
                })
            step += 1 # Increment global step counter

        # --- End of Epoch ---
        avg_epoch_loss = total_epoch_loss / num_batches
        print(f"--- Epoch {ep+1} Summary ---")
        print(f"Average Train Loss: {avg_epoch_loss:.4f}")

        # --- VALIDATION ---
        print("Running Validation...")
        encoder_policy.eval() # Set model to evaluation mode
        total_eval_loss = 0.0
        total_eval_unlearn_loss = 0.0
        total_eval_kl_loss = 0.0
        total_eval_sae_reg_loss = 0.0
        total_eval_dialect_reg_loss = 0.0
        # Use the validation split of the dialect dataset
        eval_batch_data = dataset["validation"][:] # Get all validation data (adjust if too large)

        sae_input_eval = tokenizer(
            eval_batch_data['sae_prompts'], padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).to(device)
        dialect_input_eval = tokenizer(
            eval_batch_data['dialect_prompts'], padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).to(device)

        with torch.inference_mode(): # Use inference mode for validation
            # Policy model outputs
            embed_policy_sae_eval = encoder_policy(sae_input_eval.input_ids)[0]
            embed_policy_dialect_eval = encoder_policy(dialect_input_eval.input_ids)[0]

            # Reference model outputs
            embed_reference_sae_eval = encoder_reference(sae_input_eval.input_ids)[0]
            embed_reference_dialect_eval = encoder_reference(dialect_input_eval.input_ids)[0]

        # Calculate validation losses
        loss_unlearn_eval = loss_fkt(embed_policy_sae_eval, embed_policy_dialect_eval)
        loss_reg_sae_eval = loss_fkt(embed_policy_sae_eval, embed_reference_sae_eval)
        loss_reg_dialect_eval = loss_fkt(embed_policy_dialect_eval, embed_reference_dialect_eval)

        # KL Divergence Loss (using train kl loss for simplicity)
        loss_kl_reg_eval = loss_kl_reg

        # Total validation loss
        eval_loss = (loss_unlearn_eval + alpha * loss_reg_sae_eval +
                        beta * loss_reg_dialect_eval + gamma * loss_kl_reg_eval)

        total_eval_loss = eval_loss.item()
        total_eval_unlearn_loss = loss_unlearn_eval.item()
        total_eval_kl_loss = loss_kl_reg_eval.item()
        total_eval_sae_reg_loss = loss_reg_sae_eval.item()
        total_eval_dialect_reg_loss = loss_reg_dialect_eval.item()


        print(
            f'Validation Loss: {total_eval_loss:.4f} | Unlearn: {total_eval_unlearn_loss:.4f} | '
            f'KL Reg: {total_eval_kl_loss:.4f} | SAE Reg: {total_eval_sae_reg_loss:.4f} | Dialect Reg: {total_eval_dialect_reg_loss:.4f}'
        )
        if config.wandb['enable_logging']:
            wandb.log({
                'step': step,
                'epoch': ep + 1,
                'eval_loss': total_eval_loss,
                'eval_loss_unlearn': total_eval_unlearn_loss,
                'eval_loss_kl_reg': total_eval_kl_loss,
                'eval_loss_reg_sae': total_eval_sae_reg_loss,
                'eval_loss_reg_dialect': total_eval_dialect_reg_loss,
            })

    # --- Save Trained Model ---
    print("Training finished. Saving model...")
    save_path_base = config.training.get('save_path', 'models')
    run_id = wandb_run.id if config.wandb['enable_logging'] and 'wandb_run' in locals() else f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_path = os.path.join(save_path_base, args.dialect, run_id)

    os.makedirs(save_path, exist_ok=True)
    encoder_policy.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to: {save_path}")

    if config.wandb['enable_logging'] and 'wandb_run' in locals():
        model_artifact = wandb.Artifact(f'policy_encoder_{args.dialect}', type='model')
        model_artifact.add_dir(save_path)
        wandb_run.log_artifact(model_artifact)

        wandb.summary['model_save_path'] = save_path
        wandb.summary['final_eval_loss'] = total_eval_loss
        wandb.finish()


# --- Data Processing Functions (Keep as they are) ---
def process_dialect_data(folder, dialect):
    split_dict = {}
    for split in ['train', 'val', 'test']:
        file_path = os.path.join(folder, dialect, f'{split}.csv')
        try:
            df = pd.read_csv(file_path, encoding="unicode_escape")
            data_dict = {
                "dialect_words": df["Dialect_Word"].astype(str).tolist(),
                "sae_words": df["SAE_Word"].astype(str).tolist(),
                "dialect_prompts": df["Dialect_Prompt"].astype(str).tolist(),
                "sae_prompts": df["SAE_Prompt"].astype(str).tolist()
            }
            split_dict[split] = Dataset.from_dict(data_dict)
        except FileNotFoundError:
             print(f"Warning: File not found {file_path}. Skipping split '{split}'.")
             split_dict[split] = Dataset.from_dict({
                "dialect_words": [], "sae_words": [], "dialect_prompts": [], "sae_prompts": []
             })
        except KeyError as e:
             print(f"Error: Column {e} not found in {file_path}. Please check CSV headers.")
             raise
    return DatasetDict({
        "train": split_dict["train"],
        "validation": split_dict["val"],
        "test": split_dict["test"]
    })


def parse_arguments():
    parser = argparse.ArgumentParser(description='Dialect Unlearning with KL Regularization')
    parser.add_argument('-c',
                        '--config',
                        default="configs/dialect_unlearning_kl.yaml",
                        type=str,
                        dest="config",
                        help='Config .yaml file path (default: configs/dialect_unlearning.yaml)')
    parser.add_argument('--dialect', type=str, default='sge', choices=['aae','bre','che','ine','sge'])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)


# def clip_inference(clip_model, clip_processor, caption_dataset, caption_dataloader):
#     text = [caption_dataset.captions_by_imageid[imageid][0] for imageid in caption_dataset.image_ids]
#     # text = text[:1000]
#     logits_per_image = []
#     image_embeds = []
#     dataiter = iter(caption_dataloader)
#     # Iterate over the DataLoader to get batches of images and captions
#     for i in range(len(caption_dataloader)):
#         cap_batch = next(dataiter)
#         images = cap_batch['images']
#         inputs = clip_processor(text=text, images=images, return_tensors="pt", padding=True).to(clip_model.device)
#         with torch.no_grad():
#             outputs = clip_model(**inputs, return_dict=True)
#             # print(f'*** CLIP LOSS: {outputs.loss} ***')
#         logits_per_image.append(outputs.logits_per_image) # shape: (batch_size, num_captions)
#         # for j in range(outputs.logits_per_image.shape[0]):
#         #     row = F.softmax(outputs.logits_per_image[j])
#         #     print(torch.topk(row, k=5))
#         # exit()
#         image_embeds.append(outputs.image_embeds)
        
#     logits_per_image = torch.cat(logits_per_image, dim=0) # shape: (total_images, num_captions)
#     image_embeds = torch.cat(image_embeds, dim=0) # shape: (total_images, num_captions)
#     print(logits_per_image.shape)
    
#     return logits_per_image, image_embeds