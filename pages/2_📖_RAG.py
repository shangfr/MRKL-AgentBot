# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 15:58:09 2023

@author: shangfr
"""
import streamlit as st
from vectorstores import vectordb, qa_retrieval

st.set_page_config(page_title="RAG", page_icon="📖")
st.header("🦜🔗 RAG问答")
st.caption("🚀 Retrieval Augmented Generation (RAG)")

@st.cache_data(ttl=60)
def ask_qa(question, sk, collection_name):
    @st.cache_resource
    def update_qa(sk, collection_name):
        return qa_retrieval(sk, collection_name=collection_name)
    qa = update_qa(sk, collection_name)
    response = qa({"query": question})
    return response

@st.cache_resource
def get_vectordb(**kwargs):
    return vectordb(**kwargs)
    
    
with st.sidebar:
    col1,col2 = st.columns(2)
    on = col1.toggle('New Collection')
    if on:
        collection_name = col2.text_input('Name', "agent")
    else:
        db = get_vectordb()
        collections = db._client.list_collections()
        collections = [c.name for c in collections]
        collection_name = col2.radio("Select Collection to Retrieve",
                                   options=collections,
                                   index=0,
                                   horizontal=True
                                   )
    chunk_size = st.slider("Chunk Size", 100, 384, 300, help="openai:text-embedding-ada-002 & 8191")
    uploaded_file = st.file_uploader(
        label="📖 上传资料", accept_multiple_files=False
    )
    if uploaded_file is None:
        db = get_vectordb(collection_name = collection_name)
        st.info("上传文档资料，提升问答质量。")
    else:
        db = get_vectordb(file=uploaded_file, chunk_size=chunk_size, collection_name = collection_name)
        cnt = db._collection.count()
        collections = db._client.list_collections()
        st.info(f"Vector DB 新增数据成功！  {cnt}")
        collections = [c.name for c in collections]
        st.info("文件名:", uploaded_file.name)
    if st.button("重置", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()

sk = st.slider("Number of Documents to Return", 1, 5, key='k')

question = st.text_input(
    "Ask something about the article",
    placeholder="请对文章进行摘要总结。",
)

if question:
    response = ask_qa(question, sk, collection_name)
    st.success(response["result"])
    with st.expander("📖 Show Source Documents"):
        st.json(response["source_documents"])
