from model import LLM

def get_answer(model, question:str):
    print(f"Question: {question}")
    try:
        answer = model(question)
        print(f"Answer:\n{answer}")
    except:
        print("Similarity search failed.")

if __name__ == "__main__":
    """Ask directly to the LLM model. Could be prone to hallucination."""
    while True:
        query = input(f"Type in your question:")
        get_answer(LLM, query)