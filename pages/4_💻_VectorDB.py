# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:44:53 2023

@author: shangfr
"""
import streamlit as st
from peek import ChromaPeek

st.set_page_config(page_title="VectorDB", page_icon="💻")
st.header("Chroma Peek 👀")
st.caption("💡 Chroma makes it easy to build LLM apps by making knowledge, facts, and skills pluggable for LLMs.")

# load collections
path = "chroma"
peeker = ChromaPeek(path)

## create radio button of each collection
#col1, col2 = st.columns([1,3])
collections = peeker.get_collections()
if not collections:
    st.link_button("创建本地向量数据库", "/RAG")
    st.stop()

with st.sidebar:
    collection_selected=st.radio("Select Collection To View",
             options=collections,
             index=0,
             )
    
df  = peeker.get_collection_data(collection_selected, dataframe=True)
size = df.shape[0]

if size == 0:
    st.info(f"集合{collection_selected}共有{size}个向量")
    st.stop()
    
st.info(f"集合{collection_selected}共有{size}个向量")


edited_df = st.data_editor(df,column_config={
                            "delete": st.column_config.CheckboxColumn(
                                "Delete vector?",
                                help="Select your **delete** rows",
                                default=False,
                            )
                        },
                        disabled=['ids', 'metadatas', 'documents'],
                        hide_index=True)
                        

delete_ids = edited_df.loc[edited_df["delete"]==True]["ids"].tolist()
if delete_ids:
    if st.button("确定？删除", type="primary"):
        peeker.delete(delete_ids, collection_selected)
        st.rerun()
        #st.markdown(f"Delete ids: **{delete_ids}** 🎈")
    else:
        st.markdown(f"Delete ids: **{delete_ids}** 🎈")

st.divider()

query = st.text_input("文档查询", placeholder="按相似度返回前3个")
if query:
    result_df = peeker.query(query, collection_selected, dataframe=True)
    
    st.dataframe(result_df, use_container_width=True)


        