import argparse
from build_rag import RAG, CRAG
from utils import get_answer
from utils_CRAG import get_censored_answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask questions")
    parser.add_argument("--CRAG", action="store_true", help="Use CRAG")
    args = parser.parse_args()

    if args.CRAG:
        print("Using CRAG")
        app = CRAG()
        while True:
            query = input(f"Type in your question:")
            get_censored_answer(app, query)
    else:
        print("Using RAG")
        qa_chain = RAG()
        while True:
            query = input(f"Type in your question:")
            get_answer(qa_chain, query)
