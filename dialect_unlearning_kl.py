import argparse
import os
from datetime import datetime
import torch
import wandb
from torch.utils.data import DataLoader
from utils.config_parser import ConfigParser
import csv
import numpy as np
from datasets import Dataset, DatasetDict
import pandas as pd
import random


def main(args):
    # define and parse arguments
    # config, config_path = create_parser()
    config = ConfigParser(args.config)
    config_path = args.config
    torch.manual_seed(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(config.training['num_threads'])

    # rtpt = config.create_rtpt()
    # rtpt.start()

    # load dataset
    # dataset = config.load_datasets()
    # dataloader = DataLoader(dataset,
    #                         batch_size=config.clean_batch_size,
    #                         shuffle=True)
    # train_set = process_dialect_data(config.dialect_file)
    
    dataset = process_dialect_data(config.dialect_file_folder, args.dialect)
    dataset = dataset.shuffle(seed=config.seed)
    
    control_dataset = process_control_data(config.dialect_file_folder, args.dialect)

    # load models
    # tokenizer = config.load_tokenizer()
    # encoder_reference = config.load_text_encoder().to(device)
    # encoder_policy = config.load_text_encoder().to(device)
    
    
    encoder_reference, tokenizer = config.load_encoder_and_tokenizer()
    encoder_reference = encoder_reference.to(device)
    encoder_policy, _ = config.load_encoder_and_tokenizer()
    encoder_policy = encoder_policy.to(device)

    # freeze reference model
    for param in encoder_reference.parameters():
        param.requires_grad = False

    # define optimizer
    optimizer = config.create_optimizer(encoder_policy)
    lr_scheduler = config.create_lr_scheduler(optimizer)

    # define loss function
    loss_fkt = config.loss_fkt

    # init WandB logging
    if config.wandb['enable_logging']:
        wandb_run = wandb.init(**config.wandb['args'])
        wandb.save(config_path, policy='now')
        wandb.watch(encoder_policy)
        wandb.config.optimizer = {
            'type': type(optimizer).__name__,
            'betas': optimizer.param_groups[0]['betas'],
            'lr': optimizer.param_groups[0]['lr'],
            'eps': optimizer.param_groups[0]['eps'],
            'weight_decay': optimizer.param_groups[0]['weight_decay']
        }
        # wandb.config.injection = config.injection
        wandb.config.training = config.training
        wandb.config.seed = config.seed

    # prepare training
    step = -1
    encoder_policy.train()
    encoder_reference.eval()
    # dataloader_iter = iter(dataloader)

    # training loop
    print(f'EPOCHS: {config.training["epochs"]}')
    for ep in range(config.training['epochs']):
        
        ## TRAIN ##
        for i in range(int(np.ceil(len(dataset["train"])/config.clean_batch_size))):
            step += 1
            batch = dataset["train"][i*config.clean_batch_size:(i+1)*config.clean_batch_size]
            
            batch_sae_prompt = batch['sae_prompts']
            batch_dialect_prompt = batch['dialect_prompts']
            batch_sae_word = batch['sae_words']
            batch_dialect_word = batch['dialect_words']
            
            # random.seed(config.seed)
            # batch_control_prompt = random.sample(control_dataset["train"]['sae_prompts'], k=5*config.clean_batch_size)
            batch_control_prompt = control_dataset["train"]['sae_prompts']
            
            # # OR
            # batch_sae_prompt = [example['sae_prompts'] for example in batch]
            # batch_dialect_prompt = [example['dialect_prompts'] for example in batch]
            # batch_sae_word = [example['sae_words'] for example in batch]
            # batch_dialect_word = [example['dialect_words'] for example in batch]
            
            sae_input = tokenizer(
                batch_sae_prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            dialect_input = tokenizer(
                batch_dialect_prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            control_input = tokenizer(
                batch_control_prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            print(batch_sae_prompt)
            print(batch_dialect_prompt)
            
            embed_policy_sae = encoder_policy(sae_input.input_ids.to(device))[0]
            embed_policy_dialect = encoder_policy(dialect_input.input_ids.to(device))[0]
            embed_policy_control = encoder_policy(control_input.input_ids.to(device))[0]
            
            with torch.no_grad():
                embed_reference_sae = encoder_reference(sae_input.input_ids.to(device))[0]
                embed_reference_dialect = encoder_reference(dialect_input.input_ids.to(device))[0]
                embed_reference_control = encoder_reference(control_input.input_ids.to(device))[0]
            
            loss_unlearn = loss_fkt(embed_policy_sae, embed_policy_dialect)
            loss_reg_sae = loss_fkt(embed_policy_sae, embed_reference_sae)
            loss_reg_dialect = loss_fkt(embed_policy_dialect, embed_reference_dialect)
            loss_reg_control = loss_fkt(embed_policy_control, embed_reference_control)

            beta = 0.1
            loss = loss_unlearn + loss_reg_sae + beta * loss_reg_dialect + loss_reg_control

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # update rtpt and lr scheduler
            # rtpt.step()

            if lr_scheduler:
                lr_scheduler.step()

            # log results
            loss_unlearn = loss_unlearn.detach().cpu().item()
            loss_reg_sae = loss_reg_sae.detach().cpu().item()
            loss_reg_dialect = loss_reg_dialect.detach().cpu().item()
            loss = loss.detach().cpu().item()
            print(
                f'Step {step}: Unlearning Loss: {loss_unlearn:.4f} \t'
                f'Control Loss: {loss_reg_control:.4f} \t'
                f'SAE Reg Loss: {loss_reg_sae:.4f} \t'
                f'Dialect Reg Loss: {loss_reg_dialect:.4f} \t'
                f'Total Loss: {loss:.4f}'
            )
            if config.wandb['enable_logging']:
                wandb.log({
                    'loss': loss,
                    'loss_unlearn': loss_unlearn,
                    'loss_reg_control': loss_reg_control,
                    'loss_reg_sae': loss_reg_sae,
                    'loss_reg_dialect': loss_reg_dialect,
                    'Loss Weight': config.loss_weight,
                    'Learning Rate': optimizer.param_groups[0]['lr'],
                    'beta': beta
                })
        
        
        ## VALIDATION ##
        eval_batch = dataset["validation"]
        
        batch_sae_prompt = eval_batch['sae_prompts']
        batch_dialect_prompt = eval_batch['dialect_prompts']
        batch_sae_word = eval_batch['sae_words']
        batch_dialect_word = eval_batch['dialect_words']
        # batch_control_prompt = random.sample(control_dataset["validation"]['sae_prompts'], k=5*config.clean_batch_size)
        batch_control_prompt = control_dataset["validation"]['sae_prompts']
        
        sae_input = tokenizer(
            batch_sae_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        dialect_input = tokenizer(
            batch_dialect_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        control_input = tokenizer(
            batch_control_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.inference_mode():
            embed_policy_sae = encoder_policy(sae_input.input_ids.to(device))[0]
            embed_policy_dialect = encoder_policy(dialect_input.input_ids.to(device))[0]
            embed_policy_control = encoder_policy(control_input.input_ids.to(device))[0]
            embed_reference_sae = encoder_reference(sae_input.input_ids.to(device))[0]
            embed_reference_dialect = encoder_reference(dialect_input.input_ids.to(device))[0]
            embed_reference_control = encoder_reference(control_input.input_ids.to(device))[0]
            
        
        loss_unlearn = loss_fkt(embed_policy_sae, embed_policy_dialect)
        loss_reg_sae = loss_fkt(embed_policy_sae, embed_reference_sae)
        loss_reg_dialect = loss_fkt(embed_policy_dialect, embed_reference_dialect)
        loss_reg_control = loss_fkt(embed_policy_control, embed_reference_control)

        beta = 0.1
        loss = loss_unlearn + loss_reg_sae + beta * loss_reg_dialect + loss_reg_control

        # log results
        loss_unlearn = loss_unlearn.detach().cpu().item()
        loss_reg_sae = loss_reg_sae.detach().cpu().item()
        loss_reg_dialect = loss_reg_dialect.detach().cpu().item()
        loss_reg_control = loss_reg_control.detach().cpu().item()
        loss = loss.detach().cpu().item()
        
        print(
            f'Step {step}: Unlearning Loss: {loss_unlearn:.4f} \t'
            f'Control Loss: {loss_reg_control:.4f} \t'
            f'SAE Reg Loss: {loss_reg_sae:.4f} \t'
            f'Dialect Reg Loss: {loss_reg_dialect:.4f} \t'
            f'Total Loss: {loss:.4f}'
        )
        if config.wandb['enable_logging']:
            wandb.log({
                'eval_loss': loss,
                'eval_loss_unlearn': loss_unlearn,
                'eval_loss_reg_control': loss_reg_control,
                'eval_loss_reg_sae': loss_reg_sae,
                'eval_loss_reg_dialect': loss_reg_dialect,
            })
        

    # save trained policy model
    if config.wandb['enable_logging']:
        save_path = os.path.join(config.training['save_path'], wandb_run.id)
        
    os.makedirs(save_path, exist_ok=True)
    encoder_policy.save_pretrained(f'{save_path}')

    if config.wandb['enable_logging']:
        wandb.save(os.path.join(save_path, '*'), policy='now')
        wandb.summary['model_save_path'] = save_path
        wandb.summary['config_save_path'] = config_path
        # finish logging
        wandb.finish()
    
    
def process_dialect_data(folder, dialect):
    split_dict = {}
    for split in ['train', 'val', 'test']:
        # dialect_words = []
        # sae_words = []
        # dialect_prompts = []
        # sae_prompts = []
        file_path = os.path.join(folder, dialect, f'{split}.csv')
        # with open(file_path, 'r') as f:
        #     reader = csv.reader(f)
        #     next(reader)
        #     for row in reader:
        #         dialect_words.append(row[0])
        #         sae_words.append(row[1])
        #         dialect_prompts.append(row[2])
        #         sae_prompts.append(row[3])
        df = pd.read_csv(file_path, encoding="unicode_escape")
        data_dict = {
            "dialect_words": df["Dialect_Word"].tolist(),
            "sae_words": df["SAE_Word"].tolist(),
            "dialect_prompts": df["Dialect_Prompt"].tolist(),
            "sae_prompts": df["SAE_Prompt"].tolist()
        }
        split_dict[split] = Dataset.from_dict(data_dict)
    return DatasetDict({
        "train": split_dict["train"],
        "validation": split_dict["val"],
        "test": split_dict["test"]
    })


def process_control_data(folder, dialect):
    # avoid using the same dialect as the dialect file
    dialect_list = ['aae', 'bre', 'che', 'ine', 'sge']
    dialect_list.remove(dialect)
    
    prompts = {
        "train": [],
        "val": [],
        "test": []
    }
    
    for d in dialect_list:
        for split in ['train', 'val', 'test']:
            data_file = os.path.join(folder, d, f'{split}.csv')
            df = pd.read_csv(data_file, encoding="unicode_escape")
            prompts[split].extend(df["SAE_Prompt"].tolist())
    split_dict = {}
    for split in ['train', 'val', 'test']:
        split_dict[split] = Dataset.from_dict({"sae_prompts": prompts[split]})
    return DatasetDict({
        "train": split_dict["train"],
        "validation": split_dict["val"],
        "test": split_dict["test"]
    })


def parse_arguments():
    parser = argparse.ArgumentParser(description='Integrating homoglyph')
    parser.add_argument('-c',
                        '--config',
                        default="configs/dialect_unlearning.yaml",
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    parser.add_argument('--dialect', type=str, default='sge', choices=['aae','bre','che','ine','sge'])
    # args = parser.parse_args()
    # config = ConfigParser(args.config)
    # return config, args.config
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
