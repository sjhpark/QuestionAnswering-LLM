from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter, NLTKTextSplitter, SpacyTextSplitter
from langchain.vectorstores import Cassandra, Chroma, FAISS # vector database
from model import get_llm
from utils import pdf_loader, docs_splitter, get_embeddings, \
                    build_database, get_retriever, get_qa_chain

def RAG():
    # get LLM
    LLM, config = get_llm()
    params = config['params']
    # load document
    docs = config['docs']
    assert docs.endswith(".pdf"), "Please provide a PDF document"
    docs = pdf_loader(docs)
    # split document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=params['chunk_size'], chunk_overlap=params['chunk_overlap'], separators=params['separator'])
    chunks = docs_splitter(text_splitter=text_splitter, docs=docs)
    # get embeddings
    embedding_model_wrapper = HuggingFaceInstructEmbeddings
    embedding_model = params['embedding_model']
    embeddings = get_embeddings(embedding_model_wrapper, embedding_model, device=params['embedding_device'], query_instruction=params['query_instruction'])
    # build database
    database = FAISS
    db = build_database(database, chunks, embeddings)
    # get retriever
    retriever = get_retriever(db, search_type=params['search_type'], k=params['k'])
    # get Q&A chain
    qa_chain = get_qa_chain(LLM, retriever, chain_type=params['chain_type'])
    return qa_chain