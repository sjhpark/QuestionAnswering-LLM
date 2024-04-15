"""This script is modified from the work done by the author of the Medium article: 
https://saankhya.medium.com/mistral-instruct-7b-finetuning-on-medmcqa-dataset-6ec2532b1ff1"""
from peft import AutoPeftModelForCausalLM
from transformers import GenerationConfig
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
import re
import json
import pandas as pd
from datasets import Dataset
from finetune import generate_prompt, load_HF_dataset

import sys
sys.path.append('..')
from utils import color_print

def get_gt_labels(dataset:pd.DataFrame)->list:
    """This is for collecting ground truth labels (answers) from the MedMCQA dataset."""
    gt_answers = []
    for i in range(len(dataset)):
        sample = dataset.iloc[i]
        cop = "Nothing"
        if sample['cop'] == 1:
            cop = sample['opa']
        elif sample['cop'] == 2:
            cop = sample['opb']
        elif sample['cop'] == 3:
            cop = sample['opc']
        elif sample['cop'] == 4:
            cop = sample['opd']
        gt_answers.append(cop)
    return gt_answers

def QA_inference(question_prompt):
    inputs = tokenizer(question_prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
    outputs = model.generate(**inputs, generation_config=generation_config)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True) # decode tokenized predicted answers to natural language text
    return answer

if __name__ == "__main__":
    dataset = "medmcqa" # openlifescienceai/medmcqa (https://huggingface.co/datasets/openlifescienceai/medmcqa)
    _, val_df, _ = load_HF_dataset(dataset) # we will use validation dataset for evaluation since test dataset doesn't have ground truth answers

    # preprocess the validation dataset
    val_df['text'] = val_df.apply(lambda x: generate_prompt(x, mode='val'), axis=1)
    val_dataset = Dataset.from_pandas(val_df)

    # model_name = "sjhpark/dolly-v2-3b-finetuned-medmcqa" # fine-tuned (via PEFT) model from Hugging Face
    model_name = "sjhpark/Qwen1.5-0.5B-finetuned-medmcqa" # fine-tuned (via PEFT) model from Hugging Face

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoPeftModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="cuda")

    generation_config = GenerationConfig(
        do_sample=True,
        top_k=1,
        temperature=0.1,
        max_new_tokens=25,
        pad_token_id=tokenizer.pad_token_id
    )

    # get ground truth labels (answers)
    gt_answers = get_gt_labels(val_df)

    # get predicted answers
    all_preds = []
    batch = 1
    val_data_prompts = list(val_dataset['text'])
    for i in tqdm(range(0, len(val_data_prompts), batch)):
        question_prompts = val_data_prompts[i:i+batch]
        pred = QA_inference(question_prompts) # predicted answers from model inference
        preds = []
        for text in pred:
            preds.append(re.search(r'ANSWER: \s*(.*)', text).group(1))
        all_preds.extend(preds)

    # evaluation with threshold score
    threshold_score = 0.7 
    matching_scores = [sum(g == t for g, t in zip(gen, truth)) / max(len(gen), len(truth)) for gen, truth in zip(all_preds, gt_answers)]
    correct_count = sum(score >= threshold_score for score in matching_scores)
    color_print(f"Accuracy: {correct_count/len(val_dataset)*100}%", "light_green", True)