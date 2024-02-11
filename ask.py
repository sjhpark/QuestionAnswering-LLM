from build_rag import RAG
from utils import get_answer

if __name__ == "__main__":
    qa_chain = RAG()
    while True:
        query = input(f"Type in your question:")
        get_answer(qa_chain, query)