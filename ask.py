from build_rag import build_RAG
from utils import get_answer

if __name__ == "__main__":
    qa_chain = build_RAG()
    while True:
        query = input(f"Type in your question:")
        get_answer(qa_chain, query)