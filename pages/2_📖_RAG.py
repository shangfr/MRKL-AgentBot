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
    db = vectordb(**kwargs)
    #collections = db._client.list_collections()
    #cnt = db._collection.count()
    #st.info(f"Vector DB 新增数据成功！  {cnt}")
    return db
    
    
with st.sidebar:
    col1, col2 = st.columns(2)
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
    chunk_size = st.slider("Chunk Size", 100, 290, 290, help="openai:text-embedding-ada-002 & 8191")
    uploaded_file = st.file_uploader(
        label="📖 上传资料", accept_multiple_files=False
    )
    urls = []
    if uploaded_file is None:
        _ = get_vectordb(collection_name = collection_name)
        st.info("👆上传文档👇输入网址，提升问答质量。")
    else:
        st.info(f"文件名:{uploaded_file.name}")

    url = st.text_input('输入网址：', "",help='L网页URL：动态或静态网页（通过Chromium服务渲染）')
    if url:
        urls = url.strip().replace(" ","").split("\n")

    if urls==[] and uploaded_file is None:
        st.warning('没有可分析的文本。', icon="⚠️")
    else:
        if st.button("入库", use_container_width=True):
            _ = get_vectordb(file=uploaded_file, urls=urls, chunk_size=chunk_size, collection_name = collection_name)  
    if st.button("重置", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()

col0, col1 = st.columns([3, 1])
sk = col1.number_input("返回文档数", 1, 5 ,3, key='k')
question = col0.text_input(
    "Ask something about the article",
    placeholder="请对文章进行摘要总结。",
)

if question:
    response = ask_qa(question, sk, collection_name)
    st.success(response["result"])
    with st.expander("📖 Show Source Documents"):
        st.json(response["source_documents"])

 