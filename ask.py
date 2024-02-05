from utils import get_answer, load_config
from build_rag import RAG

def build_RAG():
    config = load_config('config.yaml')
    docs = config['docs']
    assert docs.endswith(".pdf"), "Please provide a PDF document"
    qa_chain = RAG(docs)
    return qa_chain

if __name__ == "__main__":
    qa_chain = build_RAG()
    while True:
        query = input(f"Type in your question:")
        get_answer(qa_chain, query)