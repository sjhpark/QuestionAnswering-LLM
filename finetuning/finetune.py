"""This script is modified from the work done by the author of the Medium article: 
https://saankhya.medium.com/mistral-instruct-7b-finetuning-on-medmcqa-dataset-6ec2532b1ff1"""
import os
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig, GPTQConfig, TrainingArguments
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

from huggingface_hub.hf_api import HfFolder
txt = open('../model_API_keys/huggingface_write_api.txt')
HfFolder.save_token(txt.readlines()[1])
txt.close()

import sys
sys.path.append('..')
from utils import color_print

def create_tokenizer(model_name:str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None: # this is to add padding token if the tokenizer does not have it (depends on the model used)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # tokenizer.pad_token = tokenizer.unk_token
    # tokenizer.pad_token_id =  tokenizer.unk_token_id
    tokenizer.padding_side = 'left'
    return tokenizer

def load_HF_model(model_name:str):
    color_print(f"Loading Hugging Face model: {model_name}...", "light_green", True)
    # tokenizer
    tokenizer = create_tokenizer(model_name)

    # quantization config
    quantization_config_loading = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
    )

    # model
    model = AutoModelForCausalLM.from_pretrained(
                                model_name,
                                quantization_config=quantization_config_loading,
                                device_map="auto",
                            )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache=False
    model.config.pretraining_tp=1
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    return model

def create_peft_model(model, config:dict):
    peft_config = LoraConfig(
                    r=config['rank'],
                    lora_alpha=config['alpha'],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=config['target_modules'],
                )
    peft_model = get_peft_model(model, peft_config)
    return peft_model, peft_config

def train(peft_model, train_data, peft_config, tokenizer, config:dict):
    training_arguments = TrainingArguments(
                            output_dir=config['save_name'],
                            per_device_train_batch_size=config['train_batch'],
                            gradient_accumulation_steps=1,
                            optim=config['optimizer'],
                            learning_rate=config['lr'],
                            lr_scheduler_type="cosine",
                            save_total_limit=10, # limit the total amount of checkpoints to save
                            save_strategy="epoch",
                            logging_steps=10,
                            num_train_epochs=1,
                            max_steps=config['max_steps'],
                            fp16=True,
                            push_to_hub=True
                        )
    trainer = SFTTrainer(
            model=peft_model,
            train_dataset=train_data,
            peft_config=peft_config,
            dataset_text_field="text",
            args=training_arguments,
            tokenizer=tokenizer,
            packing=False,
            max_seq_length=config['max_seq_length'],
    )

    color_print("Training the model...", "light_magenta", True)
    trainer.train()
    trainer.push_to_hub() # push the trained model to huggingface hub

def load_HF_dataset(dataset_name:str="medmcqa")->pd.DataFrame:
    color_print(f"Loading {dataset_name} dataset...", "light_green", True)
    dataset = load_dataset(dataset_name)
    train_df = pd.DataFrame(dataset['train'])
    val_df = pd.DataFrame(dataset['validation'])
    test_df = pd.DataFrame(dataset['test'])

    train_df['topic_name'] = train_df['topic_name'].fillna('Unknown')
    train_df['exp'] = train_df['exp'].fillna('')
    return train_df, val_df, test_df

def generate_QA(sample):
    """This is for processing MedMCQA dataset."""
    cop = 'Nothing'
    if sample['cop'] == 1:
        cop = sample['opa']
    elif sample['cop'] == 2:
        cop = sample['opb']
    elif sample['cop'] == 3:
        cop = sample['opc']
    elif sample['cop'] == 4:
        cop = sample['opd']
    question = '{}\nOptions:\n1. {}\n2. {}\n3. {}\n4. {}\n'.format(sample['question'], sample['opa'], sample['opb'], sample['opc'], sample['opd'])
    answer = cop
    return question, answer

def generate_prompt(x):
    question, answer = generate_QA(x)
    prompt_template = f"""Solve the following multiple choice question by selecting correct option(s)."""
    prompt = f"""{prompt_template}\nQUESTION: {question}\nANSWER: {answer}"""
    return prompt

if __name__ == "__main__":
    # dataset for training
    dataset = "medmcqa"
    train_df, val_df, test_df = load_HF_dataset(dataset)
    color_print(f"Train dataset: {train_df.shape}", "light_green", True)
    color_print(f"Validation dataset: {val_df.shape}", "light_green", True)
    color_print(f"Test dataset: {test_df.shape}", "light_green", True)

    # add "text" column to the train dataset for training
    train_df['text'] = train_df.apply(lambda x: generate_prompt(x), axis=1)
    train_data = Dataset.from_pandas(train_df)

    # text length statistics
    train_df['text_length'] = train_df['text'].apply(len)
    mean_text_len = train_df['text_length'].mean()
    color_print(f"Mean text length: {mean_text_len}", "light_green", True)
    std_text_len = train_df['text_length'].std()
    color_print(f"Standard deviation of text length: {std_text_len}", "light_green", True)

    # configuration parameters
    config = {  ### Dataset config ###
                "dataset": "medmcqa",
                ### Model config ###
                "model_name": "Qwen/Qwen1.5-0.5B", # text generation model from Hugging Face
                "save_name": "Qwen1.5-0.5B-finetuned-medmcqa", # name of the trained model to be saved
                ### Model Training config ###
                "train_batch": 1,
                "optimizer": "paged_adamw_32bit",
                "lr": 2e-4,
                "max_steps": len(train_df)//100,
                "max_seq_length": int(mean_text_len + 2*std_text_len),
                ### LoRA config ###
                "rank": 16,
                "alpha": 32, # scaler for LoRA weight matrix (rule of thumb is double the rank)
                "target_modules": ["q_proj","k_proj","v_proj","o_proj",
                                   "gate_proj","up_proj","down_proj"] # sometimes "lm_head" is added
                }

    # model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    model_name = config['model_name']
    model = load_HF_model(model_name)
    peft_model, peft_config = create_peft_model(model, config)

    # tokenizer
    tokenizer = create_tokenizer(model_name)

    # model train (fine-tune) w/ dataset
    train(peft_model, train_data, peft_config, tokenizer, config)
