# Question Answering Chatbot
Q&A Chatbot with LLMs via Retrieval Augmented Generation for Hallucination Mitigation

## Setup
```bash
conda create -n QA_llm python=3.10
conda activate QA_llm
pip install -r requirements.txt
```

## Model
Once retrieving an API key for a desired large language model, the user can add the key as an variable under 'model_API_keys' in 'config.yaml' file.
Additionally, the user should modify 'model.py' accordingly to use the desired LLM as a header for this RAG pipeline.

## Model, Input PDF, Hyperparameters
User can change the model, input PDF, and hyperparameters by modifying the arguments inside 'config.yaml' file.

## Additional Requirements
- Save PDFs inside /data to use as non-parametric knowledge sources for the chatbot. Update the 'config.yaml' file accordingly.
- Install Ollama app from [here](https://ollama.com/). Ollama allows users to download LLMs and run them locally.
- Download desired LLMs to use with Ollama (ChatOllama) using "ollama pull [model_name]" command in the terminal. Model names for Ollama can be found [here](https://ollama.com/library).
- Get an API key for Tavily from [here](https://tavily.com/). Tavily is a search engine built specifically for AI agents (LLMs).

## Run Inference (Question Answering)
```bash
python3 ask.py # build a RAG pipeline and then query
python3 ask.py --CRAG # build a Corrective RAG (https://arxiv.org/abs/2401.15884) pipeline and then query
```

## Run APP (Question Answering)
User can upload multiple PDFs and ask questions to the chatbot.
```bash
streamlit run app.py
```