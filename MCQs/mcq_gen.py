import os
import argparse
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

def get_answer(model, question:str):
    try:
        answer = model.invoke(question)
    except:
        answer = 0
    return answer

def get_llm_config(params:dict, LLM_name:str):
    LLM = ChatOllama(model=LLM_name, temperature=params['temperature'])
    params['llm'] = LLM_name
    params['is_hf'] = False
    return LLM, params

def get_llm(params:dict, LLM_name:str):
    LLM, config = get_llm_config(params, LLM_name)
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
    """Input document corpus and split them into chunks and then convert the text chunks to multiple choice questions (MCQs) using the LLM model."""
    parser = argparse.ArgumentParser()
    # parser.add_argument("--pdf", type=str, default="Omicron Variant Symptoms and Treatment.pdf", help="PDF file", required=False)
    # parser.add_argument("--pdf", type=str, default="vtol_manual.pdf", help="PDF file", required=False)
    parser.add_argument("--pdf", type=str, default="BMS_tutorial.pdf", help="PDF file", required=False)
    parser.add_argument("--LLM_name", type=str, default="mistral:instruct", help="LLM model name", required=False) # https://ollama.com/library/mistral:instruct
    parser.add_argument("--chunk_size", type=int, default=500, help="Number of characters", required=False)
    parser.add_argument("--chunk_overlap", type=int, default=0, help="Number of characters", required=False)
    args = parser.parse_args()

    pdf = args.pdf
    LLM_name = args.LLM_name
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap

    # model parameters
    params = {
        'chain_type': 'stuff',
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

    # Large Language Model (LLM)
    LLM, _ = get_llm(params, LLM_name)
    print(colored(f"MCQ Generator LLM: {LLM}", "yellow"))

    # engineered prompt template
    prompt = """Convert the following context to 
                only one multiple choice question with 
                1 correct answer and 2 wrong answers.
                Do not create more than one question based on the entire context.
                Do not create additional comments even though the answer is 
                not clear in the context.
                Do not create a question that requires the reader to refer 
                to any document or external source having the context.
                Always display the multiple choices as A), B), and C).
                Always display the correct answer after the answer choice of C)
                and display the correct answer in the format of just the letter: Answer:A.
                Do not make any additional notes or comments about your reasoning or the context.
                Context is: """

    # load document
    pdf = os.path.join("../data", pdf) # PDF file
    loader = PyPDFLoader(pdf) # PDF loader
    docs = loader.load() # load document

    # split document
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator = "\n")
    chunks = text_splitter.split_documents(docs)
    context_chunks = []
    for chunk in tqdm(chunks, desc="Splitting documents..."):
        context_chunks.append(chunk.page_content)
    print(colored(f"Number of context chunks: {len(context_chunks)}", "light_green"))

    # generate MCQs
    contexts = []
    questions = []
    answers = []
    for context in tqdm(context_chunks, desc="Generating MCQs..."):
        query = prompt + context
        print(colored(f"Query: {query}", "light_cyan"))
        response = get_answer(LLM, query)
        print(colored(f"Response: {response}", "light_magenta"))
        if response != 0: # split LLM's response into question and answer
            try:
                response = response.content
                response = response.split("Answer:")
                # # post-process question
                question = response[0].strip()
                question = question.replace("\\n", " ")
                # post-process answer
                answer = response[1].strip()[0]
                # append to lists
                contexts.append(context)
                questions.append(question)
                answers.append(answer)
            except Exception as e:
                print(colored(f"Error: {e}", "red"))
                continue
    df = pd.DataFrame({'Q': questions, 'A': answers, 'Context': contexts})
    df.to_csv(f'{args.pdf.split(".")[0]}_MCQs.csv', index=False)       
    print(colored(f"MCQs have been generated and saved to 'MCQs.csv'", "white"))