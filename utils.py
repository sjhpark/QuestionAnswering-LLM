import os
import yaml
from termcolor import colored
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter, NLTKTextSplitter, SpacyTextSplitter
from langchain.prompts import PromptTemplate

def color_print(text:str, color:str='green', bold:bool=False):
    print(colored(text, color, attrs=['bold'] if bold else None))

def load_config(config_file:str):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def pdf_loader(pdf:str):
    """
    PDF document loader
    - input: pdf file path
    - return: loaded document
    """
    pdf = os.path.join("data", "GAN.pdf") # PDF file
    loader = PyPDFLoader(pdf) # PDF loader
    docs = loader.load() # load document
    color_print(f"{os.path.basename(pdf)} has been loaded. Number of pages: {len(docs)}", "green", True)
    return docs

def doc_splitter(docs:str, chunk_size:int=300, chunk_overlap:int=0, separator:str="\n"):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,  separator=separator)
    chunks = text_splitter.split_documents(docs)
    color_print(f"Document has been split into {len(chunks)} chunks", "green", True)
    return chunks

def get_embeddings(embedder_model, device="cpu", query_instruction:str=None):
    if not query_instruction: query_instruction = "Represent the query for retrieval:"
    embeddings = embedder_model(query_instruction=query_instruction, model_kwargs={"device": device})
    color_print(f"Embeddings have been generated using {embedder_model.__name__}", "green", True)
    return embeddings

def build_database(database_type, chunks, embeddings):
    color_print(f"Building {database_type.__name__} vector database...", "green", True)
    database = database_type.from_documents(documents=chunks, embedding=embeddings)
    color_print(f"{database_type.__name__} vector database has successfully been built", "green", True)
    return database

def get_retriever(database, search_type:str="similarity", k:int=3):
    """Get vector retriever for similarity search"""
    search_type = "similarity" # search type
    retriever = database.as_retriever(search_type=search_type, search_kwargs={"k": k})
    color_print(f"Vector retriever has been created for {search_type} search", "green", True)
    return retriever

def prompt_template():
    """
    Prompt template is to help create parametrized prompts for language models.
    We will use it to prevent the model from generating 
    a response with guess and hallucination.
    """
    prompt_template = """Given the following context and question,
    generate an answer based on the context. If the answer is not
    found in the context, simply respond with "I don't know".
    Please make the answer in a natural human language style without
    making any guess or wrong information.
    CONTEXT: {context}
    QUESTION: {question}"""
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_qa_chain(llm, retriever, chain_type:str="stuff"):
    # Q&A chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type=chain_type, # other options: map_reduce, refine, etc.
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt_template()}
        )
    color_print(f"Q&A chain has been created", "green", True)
    return qa_chain

def get_answer(qa_chain, query:str): # void function
    response = qa_chain(query)
    color_print(f"Question: {query}", "green", True)
    color_print(f"--> Answer: {response['result']}", "green", False)
