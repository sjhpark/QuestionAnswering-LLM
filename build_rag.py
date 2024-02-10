from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Cassandra, Chroma, FAISS # vector database
from model import LLM
from utils import pdf_loader, doc_splitter, get_embeddings, \
                    build_database, get_retriever, get_qa_chain
from config import CFG

params = CFG()['params']

def RAG(docs:str):
    docs = pdf_loader(docs)
    chunks = doc_splitter(docs, chunk_size=params['chunk_size'], chunk_overlap=params['chunk_overlap'], separator=params['separator'])
    embedding_model_wrapper = HuggingFaceInstructEmbeddings
    embedding_model = params['embedding_model']
    embeddings = get_embeddings(embedding_model_wrapper, embedding_model, device=params['embedding_device'], query_instruction=params['query_instruction'])
    db = build_database(FAISS, chunks, embeddings)
    retriever = get_retriever(db, search_type=params['search_type'], k=params['k'])
    qa_chain = get_qa_chain(LLM, retriever, chain_type=params['chain_type'])
    return qa_chain

def build_RAG():
    docs = CFG()['docs']
    assert docs.endswith(".pdf"), "Please provide a PDF document"
    qa_chain = RAG(docs)
    return qa_chain