from model import get_llm

def get_answer(model, question:str, is_hf:bool):
    try:
        if is_hf:
            answer = model.invoke(question)
        else:
            answer = model(question)
        print(f"Answer:\n{answer}")
    except:
        print("No answer found")

if __name__ == "__main__":
    """Ask directly to the LLM model. Could be prone to hallucination."""
    LLM, config = get_llm()
    is_hf = config['params']['is_hf']
    while True:
        query = input(f"Type in your question:")
        get_answer(LLM, query, is_hf)