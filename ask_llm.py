from model import get_llm

def get_answer(model, question:str):
    try:
        answer = model.invoke(question)
        print(f"Answer:\n{answer}")
    except:
        print("No answer found")

if __name__ == "__main__":
    """Ask directly to the LLM model. Could be prone to hallucination."""
    LLM, _ = get_llm()
    while True:
        query = input(f"Type in your question:")
        get_answer(LLM, query)