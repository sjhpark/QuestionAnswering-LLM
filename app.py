import os
import time
import yaml
import streamlit as st
from utils import load_config
from app_utils import get_answer, RAG, CRAG, get_censored_answer

st.write('# 🤖 Your Q&A Bot in Your Browser')

if st.button("Click Here to Shut Down (supported for Linux and MacOS only)"):
    st.write("Shut down the server. This will free up your resources taken by the server.")
    time.sleep(2)
    os.system("pkill -9 streamlit")

config = load_config(config_file="config.yaml")

params = config['params']
param_keys = list(params.keys())

rag_type = config['RAG']['type']
rag_type_input = st.sidebar.selectbox('RAG Type', ['RAG', 'CRAG'], index=0)
config['RAG']['type'] = rag_type_input

# ask for params as inputs and update config
input = st.sidebar.text_input('Model API Key text file path only if required for your model (e.g. model_API_keys/palm_api.txt)', config['model_API_keys']['palm'])
for key in param_keys:
    # input = st.text_input(key, config['params'][key])
    if key in ['llm']:
        if rag_type_input == 'RAG':
            input = st.sidebar.selectbox(key, ['google/flan-t5-base', 'mistralai/Mistral-7B-v0.1', 'google-palm', 'mistral:instruct'], index=0)
        elif rag_type_input == 'CRAG':
            input = st.sidebar.selectbox(key, ['mistral:instruct'], index=0) # Ollama models
        params[key] = input
    elif key in ['llm_device']:
        input = st.sidebar.selectbox(key, ['cuda', 'cpu'], index=0)
        params[key] = input
    elif key in ['embedding_model']:
        input = st.sidebar.selectbox(key, ['sentence-transformers/all-MiniLM-L6-v2'], index=0)
        params[key] = input
    elif key in ['embedding_device']:
        input = st.sidebar.selectbox(key, ['cuda', 'cpu'], index=0)
        params[key] = input
    elif key in ['repetition_penalty']:
        input = st.sidebar.slider(key, min_value=0.0, max_value=2.0, value=config['params'][key], step=0.05)
        params[key] = input
    elif key in ['temperature', 'top_p']:
        input = st.sidebar.slider(key, min_value=0.0, max_value=1.0, value=config['params'][key], step=0.05)
        params[key] = input
    elif key in ['chunk_size']:
        input = st.sidebar.slider(key, min_value=0, max_value=1000, value=config['params'][key], step=10)
        params[key] = input
    elif key in ['chunk_overlap']:
        input = st.sidebar.slider(key, min_value=0, max_value=100, value=config['params'][key], step=5)
        params[key] = input
    elif key in ['k']:
        input = st.sidebar.slider(key, min_value=0, max_value=10, value=config['params'][key], step=1)
        params[key] = input
    elif key in ['max_length']:
        input = st.sidebar.slider(key, min_value=800, max_value=4000, value=config['params'][key], step=16)
        params[key] = input
    elif key in ['search_type']:
        input = st.sidebar.selectbox(key, ['similarity', 'mmr'], index=0) # TODO: implement 'similarity_score_threshold'
        params[key] = input
    elif key in ['quantize']:
        input = st.sidebar.selectbox('4-bit Quantization?\n(Applies to Huggingface models only)', [True, False], index=0)
        params[key] = input
    elif key in ['separator']:
        input = st.sidebar.selectbox(key, ['\\n\\n', '\\n'], index=0)
        params[key] = f"{input}"
    elif key in ['query_instruction']:
        input = st.sidebar.text_input(key, config['params'][key])
        params[key] = input
    elif key in ['chain_type']:
        input = st.sidebar.selectbox(key, ['stuff'], index=0) #TODO: implement 'map_reduce', 'refine', 'map_rerank'
        params[key] = input

pdfs = st.file_uploader("Upload PDF and Start Building Q&A System", type='pdf', accept_multiple_files=True)
if pdfs:
    doc_names = [pdf.name for pdf in pdfs]
    config['docs'] = doc_names

# update config
with open("config.yaml", 'w') as f:
    yaml.dump(config, f)
    st.info("Configuration is Updated")

query = st.chat_input("Ask me anything")

if query:
    if rag_type_input == 'RAG':
        # build Q&A chain
        st.info(f"Building Q&A chain with {rag_type} pipeline...")
        qa_chain = RAG(docs=pdfs, doc_names=doc_names)
        st.info("Q&A chain has been built")
        # get answer
        st.info("Getting answer...")
        response = get_answer(qa_chain, query)
        st.write("Answer:")
        st.write(response['result'])
        st.write("Source:")
        for source_docs in response['source_documents']:
            source = source_docs.metadata['source']
            doc_name = source_docs.metadata['filename']
            st.write(f"{doc_name.split('/')[-1]}, Source: {source}")
    elif rag_type_input == 'CRAG':
        # build Q&A chain
        st.info(f"Building Q&A chain with {rag_type} pipeline...")
        qa_chain = CRAG(docs=pdfs, doc_names=doc_names)
        # get answer
        st.info("Getting answer...")
        response = get_censored_answer(qa_chain, query)
        st.write("Answer:")
        st.write(response["keys"]["generation"])
        st.write("Source:")
        for source_docs in response['keys']['documents']:
            # if 'documents' in response['keys']:
            if 'source' in source_docs.metadata:
                source = source_docs.metadata['source']
                doc_name = source_docs.metadata['filename']
                st.write(f"{doc_name.split('/')[-1]}, Source: {source}")
            else:
                st.write("Web Search via Tavily")