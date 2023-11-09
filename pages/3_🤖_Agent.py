# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:12:50 2023

@author: shangfr
"""
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from vectorstores import qa_retrieval,summarize
from modules import custom_react_agent
from database import create_research_db, insert_research, read_research_table

st.set_page_config(page_title="MRKL", page_icon="🤖")
st.header("🤖 Research MRKL Agent Bot")
st.caption("🚀 Agents是大模型(LLM)、记忆(Memory)、任务规划(Planning Skills)以及工具使用(Tool Use)的集合。")


class Document:
    def __init__(self, content, topic):
        self.page_content = content
        self.metadata = {"Topic": topic}


def qa_agent(question):
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        answer = st.session_state.agent.run(question, callbacks=[st_callback])
        #st.write(answer)
    return answer


def chat_research():
    if content := st.chat_input(placeholder="请提问"):
        _ = qa_agent(content)


def generate_research():

    if user_input := st.chat_input(placeholder="输入企业名称", key="chat02"):

        st.info("Generating Introduction")
        question1 = f'查找关于{user_input}的企业介绍，尤其是在绿色环保主题下的一些内容。'
        intro = qa_agent(question1)

        st.info("Generating Statistical Facts")
        question2 = f'''查找关于{user_input}企业最近1年经营活动产生的污染物排放、治理情况材料；
            \n判断{user_input}是否是绿色企业，生成关于{user_input}污染物排放、治理情况的3到5个定量事实的列表；评估出具体数值；
        '''
        quant_facts = qa_agent(question2)

        st.info("Green Enterprise Standard Matching")
        with st.spinner("Retrieval & Matching"):
            green_matching = st.session_state.qa.run(f'''
                \n查找绿色企业标准，参考{intro}\n\n{quant_facts}\n\n分析{user_input}是否满足这些标准。
            ''')
            st.success(green_matching)

        insert_research(user_input, intro, quant_facts, green_matching)

def md_report(df):
    md_str = f'''# {df.user_input[0]}-绿色报告\n
📝 简介：{df.intro[0]}\n
⭐ 污染与治理：{df.quant_facts[0]}\n
🖊️ 绿色判断：{df.green_matching[0]}\n
'''
    return summarize(md_str)
    
    

def generate_history():

    st.dataframe(read_research_table())
    col1,col2 = st.columns([9,1])
    selected_input = col1.selectbox(label="Previous User Inputs", options=[
                                  i for i in read_research_table().user_input])

    if col2.button("生成企业报告") and selected_input:
        with st.expander("Rendered Previous Research", expanded=True):
            selected_df = read_research_table()
            selected_df = selected_df[selected_df.user_input == selected_input].reset_index(
                drop=True)
            md_str = md_report(selected_df)
            st.markdown(md_str)
    
                
            
msgs = StreamlitChatMessageHistory()

if "agent" not in st.session_state:
    create_research_db()
    st.session_state.agent = custom_react_agent(msgs)
    st.session_state.qa = qa_retrieval(rsd=False, collection_name="test")


def update(tools, collection_name):
    st.session_state.qa = qa_retrieval(
        rsd=False, collection_name=collection_name)
    st.session_state.agent = custom_react_agent(msgs, tools.append(collection_name))


with st.sidebar.form('update'):

    collections = st.session_state.qa.retriever.vectorstore._client.list_collections()
    collections = [c.name for c in collections]

    options = st.multiselect(
        '工具箱',
        ['网络搜索', '工商信息', '企业专利', '企业项目'],
        ['网络搜索', '工商信息'])


    collection_name = st.radio("Select Collection to Retrieve",
                                 options=collections,
                                 index=0,
                                 horizontal=True
                                 )

    st.form_submit_button('Update Agent', on_click=update, use_container_width=True,
                          args=[options, collection_name])


mts = ["Agent Chat", "Generate Research", "History"]
method = st.selectbox(
    "功能选择",
    mts,
    index=None,
    placeholder="Select contact method...",
)

if method == mts[2]:
    generate_history()
    
else:
    if method == mts[0]:
        chat_research()

    elif method == mts[1]:
        generate_research()

    if len(msgs.messages) == 0 or st.sidebar.button("清空聊天记录", use_container_width=True):
        msgs.clear()

    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)
        
        