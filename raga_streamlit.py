import streamlit as st
#import json
import numpy as np
import pandas as pd
import faiss
from ragatouille import RAGPretrainedModel
import os.path as osp
#import requests

@st.cache_data
def load_data():
    df_explode = pd.read_csv('esg_explode.csv')
    return df_explode[df_explode.index<50]

@st.cache_data
def load_model():
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    return RAG

df = load_data()
RAG = load_model()

#RAG.model.config.index_root = ''

index_path = RAG.index(index_name='esg_index', 
                       collection=df.chunk)


# # UI
# # Title
st.title('Search engine')

# # Query
query = st.text_input(
     label = ":blue[Query]",
     placeholder="Please search the news with..."
 )
search_button = st.button(label='Run',type='primary')


if query:
    if search_button:
        results = RAG.search(query=query, k=10)

        for res in results:
            st.markedown(f'## {res["rank"]}, {res["document_id"]}')
            st.write(f'{res["score"]}')
            st.write(f'{res["content"]}')
