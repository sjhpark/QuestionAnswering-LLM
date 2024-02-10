from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from config import CFG

params = CFG()['params']

# get LLM
if params['is_hf']:
    LLM = params['llm']
    tokenizer = params['tokenizer']
    pipe = pipeline(
        task = params['task'],
        model = LLM,
        tokenizer = tokenizer,
        pad_token_id = tokenizer.eos_token_id,
        max_length = params['max_len'],
        temperature = params['temperature'],
        top_p = params['top_p'],
        repetition_penalty = params['repetition_penalty']
        )
    LLM = HuggingFacePipeline(pipeline = pipe)
else:
    LLM = params['llm']