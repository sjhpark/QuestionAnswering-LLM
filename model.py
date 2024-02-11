from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.llms import GooglePalm
import torch
from utils import load_config

def config_hf_llm(LLM_name:str, task:str, float_domain:torch.dtype, params:dict):
    """Configure HuggingFace LLM."""
    if task == "text-generation":
        tokenizer = AutoTokenizer.from_pretrained(LLM_name, use_fast=True)
        LLM = AutoModelForCausalLM.from_pretrained(
            LLM_name,
            load_in_4bit = params['quantize'],
            device_map = params['llm_device'],
            torch_dtype = float_domain, # floating point domain can be found from huggingface.co/{model_name}
            trust_remote_code = True
            )
        params['task'] = task
        params['tokenizer'] = tokenizer
        params['llm'] = LLM_name
        params['is_hf'] = True # HuggingFace model
    elif task == "text2text-generation":
        tokenizer = AutoTokenizer.from_pretrained(LLM_name, use_fast=True)
        LLM = AutoModelForSeq2SeqLM.from_pretrained(
            LLM_name,
            load_in_4bit = params['quantize'],
            device_map = params['llm_device'],
            torch_dtype = torch.float32, # floating point domain can be found from huggingface.co/{model_name}
            trust_remote_code = True
            )
        params['task'] = task
        params['tokenizer'] = tokenizer
        params['llm'] = LLM_name
        params['do_sample'] = True
        params['is_hf'] = True
    return LLM, params

def get_llm_config():
    config = load_config('config.yaml') # load config variables
    params = config['params']
    LLM_name = params['llm'] # LLM

    # configure LLM
    if LLM_name == "meta-llama/Llama-2-7b-chat-hf":
        task = "text-generation"
        float_domain = torch.float16
        LLM, params = config_hf_llm(LLM_name, task, float_domain, params)
    
    elif LLM_name == "meta-llama/Llama-2-7b-chat-t2t":
        task = "text2text-generation"
        float_domain = torch.bfloat16
        LLM, params = config_hf_llm(LLM_name, task, float_domain, params)

    elif LLM_name == "google/flan-t5-base":
        task = "text2text-generation"
        float_domain = torch.float32
        LLM, params = config_hf_llm(LLM_name, task, float_domain, params)
    
    elif LLM_name == "google-palm":
        api_file = open(config['model_API_keys']['palm'], 'r')
        api_key = api_file.readlines()[1]
        api_key = api_key.strip()
        LLM = GooglePalm(google_api_key=api_key)
        params['llm'] = LLM_name
        params['is_hf'] = False # Not HuggingFace model

    return LLM, config

def get_llm():
    LLM, config = get_llm_config()
    params = config['params']
    if params['is_hf']:
        pipe = pipeline(
            task = params['task'],
            model = LLM,
            tokenizer = params['tokenizer'],
            pad_token_id = params['tokenizer'].eos_token_id,
            max_length = params['max_length'],
            temperature = params['temperature'],
            do_sample = params['do_sample'] if params['task'] == 'text2text-generation' else None,
            top_p = params['top_p'] if params['task'] == 'text-generation' else None,
            repetition_penalty = params['repetition_penalty']
            )
        LLM = HuggingFacePipeline(pipeline = pipe)
    return LLM, config