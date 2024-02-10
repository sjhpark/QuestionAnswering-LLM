from langchain.llms import GooglePalm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from utils import load_config
import torch

def CFG(): # configuration
    config = load_config('config.yaml') # load config variables
    params = config['params']
    LLM = params['llm'] # LLM

    # configure LLM
    if LLM == "meta-llama/Llama-2-7b-chat-hf":
        task = "text-generation"
        tokenizer = AutoTokenizer.from_pretrained(LLM, use_fast=True)
        LLM = AutoModelForCausalLM.from_pretrained(
            LLM,
            load_in_4bit = params['quantize'],
            device_map = params['llm_device'],
            torch_dtype = torch.float16, # floating point domain can be found from huggingface.co/{model_name}
            trust_remote_code = True
            )
        max_len = 2048
        params['task'] = task
        params['tokenizer'] = tokenizer
        params['max_len'] = max_len
        params['llm'] = LLM
        params['is_hf'] = True # HuggingFace model

    elif LLM == "mistralai/Mistral-7B-v0.1":
        task = "text-generation"
        tokenizer = AutoTokenizer.from_pretrained(LLM, use_fast=True)
        LLM = AutoModelForCausalLM.from_pretrained(
            LLM,
            load_in_4bit = params['quantize'],
            device_map = params['llm_device'],
            torch_dtype = torch.bfloat16, # floating point domain can be found from huggingface.co/{model_name}
            trust_remote_code = True
            )
        max_len = 2048
        params['task'] = task
        params['tokenizer'] = tokenizer
        params['max_len'] = max_len
        params['llm'] = LLM
        params['is_hf'] = True # HuggingFace model
    
    elif LLM == "google/flan-t5-base":
        task = "text2text-generation"
        tokenizer = AutoTokenizer.from_pretrained(LLM, use_fast=True)
        LLM = AutoModelForSeq2SeqLM.from_pretrained(
            LLM,
            load_in_4bit = params['quantize'],
            device_map = params['llm_device'],
            torch_dtype = torch.float32, # floating point domain can be found from huggingface.co/{model_name}
            trust_remote_code = True
            )
        max_len = 2048
        params['task'] = task
        params['tokenizer'] = tokenizer
        params['max_len'] = max_len
        params['llm'] = LLM
        params['is_hf'] = True
    
    elif LLM == "google-palm":
        api_file = open(config['model_API_keys']['palm'], 'r')
        api_key = api_file.readlines()[1]
        api_key = api_key.strip()
        LLM = GooglePalm(google_api_key=api_key)
        params['llm'] = LLM
        params['is_hf'] = False # Not HuggingFace model

    return config