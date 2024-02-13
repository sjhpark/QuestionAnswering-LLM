# Question Answering Chatbot
Retrieval-Augmented Generation (RAG) System via LLM for Question Answering based on PDFs

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

## Run Inference (Question Answering)
User can upload multiple PDFs and ask questions to the chatbot.
```bash
python3 ask.py
```

## Run API (Question Answering)
```bash
streamlit run app.py
```