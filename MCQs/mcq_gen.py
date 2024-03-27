import csv
import os
import pandas as pd
from tqdm import tqdm
from termcolor import colored
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter, NLTKTextSplitter, SpacyTextSplitter
import sys
sys.path.append('..')
from utils import pdf_loader, docs_splitter, get_embeddings, build_database, get_retriever, prompt_template

def get_answer(model, question:str):
    try:
        answer = model.invoke(question)
    except:
        answer = 0
    return answer

def get_llm_config(params:dict):
    LLM_name = "mistral:instruct" # https://ollama.com/library/mistral:instruct
    # install ollama app (https://ollama.com/) and then run "ollama pull mistral:instruct" to get the model first
    LLM = ChatOllama(model=LLM_name, temperature=params['temperature'])
    params['llm'] = LLM_name
    params['is_hf'] = False
    return LLM, params

def get_llm(params:dict):
    LLM, config = get_llm_config(params)
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

if __name__ == "__main__":
    """Input document corpus and split them into chunks and then convert the text chunks to multiple choice questions using the LLM model."""

    # params for huggingface model
    params = {
        'chain_type': 'stuff',
        'chunk_overlap': 0,
        'chunk_size': 500, # number of characters
        'embedding_device': 'cuda',
        'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
        'k': 3,
        'llm': 'google/flan-t5-base',
        'llm_device': 'cuda',
        'max_length': 2000,
        'quantize': True,
        'query_instruction': 'Represent the question for retrieving supporting documents',
        'repetition_penalty': 1.0,
        'search_type': 'similarity',
        'separator': '\n\n',
        'temperature': 0.05,
        'top_p': 1.0
    }

    LLM, _ = get_llm(params)
    prompt = """Convert the following context to 
                only one multiple choice question with 
                one correct answer and two wrong answers.
                Display the multiple choices as A), B), and C).
                Display the answer in the format of just the letter:
                ANSWER: A).
                Context is: """

    pdf = "Omicron Variant Symptoms and Treatment.pdf"
    # pdf = "apple_vision_pro_info.pdf"
    pdf = os.path.join("../data", pdf) # PDF file
    loader = PyPDFLoader(pdf) # PDF loader
    docs = loader.load() # load document

    text_splitter = CharacterTextSplitter(chunk_size=params['chunk_size'], chunk_overlap=params['chunk_overlap'], separator = "\n")
    chunks = text_splitter.split_documents(docs)
    context_chunks = []
    for chunk in tqdm(chunks, desc="Splitting documents..."):
        context_chunks.append(chunk.page_content)
    print(colored(f"Number of context chunks: {len(context_chunks)}", "blue"))

    for context in tqdm(context_chunks, desc="Generating MCQs..."):
        query = prompt + context
        print(colored(f"Query: {query}", "green"))
        answer = get_answer(LLM, query)
        print(colored(f"Response: {answer}", "magenta"))
