# # for RAG:
from langchain_community.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter, NLTKTextSplitter, SpacyTextSplitter
from langchain_community.vectorstores import Cassandra, Chroma, FAISS # vector database
from model import get_llm
from utils import pdf_loader, docs_splitter, get_embeddings, \
                    build_database, get_retriever, get_qa_chain
# for CRAG:
from langgraph.graph import END, StateGraph

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

def CRAG():
    from utils_CRAG import GraphState, retrieve, grade_documents, generate, transform_query, web_search, decide_to_generate
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("transform_query", transform_query)  # transform_query
    workflow.add_node("web_search", web_search)  # web search

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    # Compile
    app = workflow.compile()
    return app