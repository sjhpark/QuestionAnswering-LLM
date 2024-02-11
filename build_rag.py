from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Cassandra, Chroma, FAISS # vector database
from model import get_llm
from utils import pdf_loader, doc_splitter, get_embeddings, \
                    build_database, get_retriever, get_qa_chain

def RAG():
    LLM, config = get_llm()
    params = config['params']
    docs = config['docs']
    assert docs.endswith(".pdf"), "Please provide a PDF document"
    docs = pdf_loader(docs)
    chunks = doc_splitter(docs, chunk_size=params['chunk_size'], chunk_overlap=params['chunk_overlap'], separator=params['separator'])
    embedding_model_wrapper = HuggingFaceInstructEmbeddings
    embedding_model = params['embedding_model']
    embeddings = get_embeddings(embedding_model_wrapper, embedding_model, device=params['embedding_device'], query_instruction=params['query_instruction'])
    db = build_database(FAISS, chunks, embeddings)
    retriever = get_retriever(db, search_type=params['search_type'], k=params['k'])
    qa_chain = get_qa_chain(LLM, retriever, chain_type=params['chain_type'])
    return qa_chain