# Modified work done by https://github.com/avrabyt/RAG-Chatbot.git

import re
from pypdf import PdfReader
from io import BytesIO
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter, NLTKTextSplitter, SpacyTextSplitter
from langchain.vectorstores import Cassandra, Chroma, FAISS # vector database
from model import get_llm
from utils import get_embeddings, build_database, get_retriever, get_qa_chain

def get_answer(qa_chain, query:str): # void function
    response = qa_chain.invoke(query)
    return response

def pdf_parser(pdf:BytesIO, doc_name:str):
    """
    PDF document loader
    - input: pdf file
    - return: texts & document name
    """
    pdf = PdfReader(pdf)
    texts = []
    for page in pdf.pages:
        text = page.extract_text()
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        text = re.sub(r"\n\s*\n", "\n\n", text)
        texts.append(text)
    return texts, doc_name

def prepare_docs(text_splitter, texts:list, doc_name:str):
    docs = [Document(page_content=page) for page in texts]
    for i, doc in enumerate(docs):
        doc.metadata['page'] = i + 1
    
    chunks_list = []
    for doc in docs:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(page_content=chunk,
                           metadata={'page': doc.metadata['page'], 'chunk': i+1})
            doc.metadata["source"] = f"Page: {doc.metadata['page']}, Chunk #: {doc.metadata['chunk']}"
            doc.metadata['filename'] = doc_name
            chunks_list.append(doc)
    return chunks_list

def RAG(docs, doc_names):
    # get LLM
    LLM, config = get_llm()
    params = config['params']
    # split document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=params['chunk_size'], chunk_overlap=params['chunk_overlap'], separators=params['separator'])
    documents = []
    docs = [doc.getvalue() for doc in docs]
    for doc, doc_name in zip(docs, doc_names):
        text, doc_name = pdf_parser(BytesIO(doc), doc_name)
        documents = documents + prepare_docs(text_splitter, text, doc_name)
    # get embeddings
    embedding_model_wrapper = HuggingFaceInstructEmbeddings
    embedding_model = params['embedding_model']
    embeddings = get_embeddings(embedding_model_wrapper, embedding_model, device=params['embedding_device'], query_instruction=params['query_instruction'])
    # build database
    database = FAISS
    db = build_database(database, documents, embeddings)
    # get retriever
    retriever = get_retriever(db, search_type=params['search_type'], k=params['k'])
    # get Q&A chain
    qa_chain = get_qa_chain(LLM, retriever, chain_type=params['chain_type'])
    return qa_chain