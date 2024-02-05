# Question Answering Chatbot on Your Terminal
Retrieval-Augmented Generation (RAG) System via LLM for Question Answering based on PDFs

## Setup
```bash
pip install -r requirements.txt
```

## Model
Once retrieving an API key for a desired large language model, the user can add the key as an variable under 'model_API_keys' in 'config.yaml' file.
Additionally, the user should modify 'model.py' accordingly to use the desired LLM as a header for this RAG pipeline.

## Run Inference (Question Answering)
```bash
python3 ask.py
```

## Model, Input PDF, Hyperparameters
User can change the model, input PDF, and hyperparameters by modifying the arguments inside 'config.yaml' file.