# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 17:45:24 2023

@author: shangfr
"""
import os
import pandas as pd
import streamlit as st

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.llms import QianfanLLMEndpoint

file_formats = {
    "csv": pd.read_csv,
    "xls": pd.read_excel,
    "xlsx": pd.read_excel,
    "xlsm": pd.read_excel,
    "xlsb": pd.read_excel,
}


def clear_submit():
    """
    Clear the Submit Button State
    Returns:

    """
    st.session_state["submit"] = False


@st.cache_data(ttl="2h")
def load_data(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1][1:].lower()
    except:
        ext = uploaded_file.split(".")[-1]
    if ext in file_formats:
        return file_formats[ext](uploaded_file)
    else:
        st.error(f"Unsupported file format: {ext}")
        return None


st.set_page_config(page_title="LangChain: Chat with pandas DataFrame", page_icon="🦜")
st.title("🦜 LangChain: Chat with pandas DataFrame")

uploaded_files = st.file_uploader(
    "Upload a Data file",
    type=list(file_formats.keys()),
    help="Various File formats are Support",
    on_change=clear_submit,
    accept_multiple_files = True
)

if not uploaded_files:
    st.warning(
        "这个应用程序使用了**LangChain**的**PythonAstREPLTool**，它存在任意代码执行的漏洞。在部署和分享此应用程序时，请谨慎行事。"
    )
df_lst = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        df = load_data(uploaded_file)
        df_lst.append(df)

    with st.expander("查看数据"):
        sk = st.number_input("选择", 0, len(df_lst)-1)
        st.dataframe(df_lst[sk ])
        
openai_api_key = st.sidebar.text_input("API Key", type="password")


llm = QianfanLLMEndpoint(streaming=True)
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
msgs = StreamlitChatMessageHistory()

memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", chat_memory=msgs, return_messages=True)

pandas_df_agent = create_pandas_dataframe_agent(
    llm,
    df_lst,
    verbose=True,
    agent_executor_kwargs={"handle_parsing_errors":True,"memory": memory},
    max_iterations=3,
    early_stopping_method="generate",
)
    

if len(msgs.messages) == 0 or st.sidebar.button("清空聊天记录", use_container_width=True):
    msgs.clear()

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if content := st.chat_input(placeholder="What is this data about?"):
    st.chat_message("user").write(content)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        response = pandas_df_agent.run(content, callbacks=[st_cb])
