# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:12:50 2023

@author: shangfr
"""
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from vectorstores import qa_retrieval
from modules import react_agent
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
        st.write(answer)
    return answer


def chat_research():
    if content := st.chat_input(placeholder="请提问"):
        response = qa_agent(content)


def generate_research():
    st.subheader("Generate Research")

    if userInput := st.chat_input(placeholder="输入企业名称", key="chat02"):

        st.subheader("Generating Introduction:")
        question1 = f'查找关于{userInput}的企业介绍，尤其是在绿色环保主题下的一些内容。'
        intro = qa_agent(question1)

        st.subheader("Generating Statistical Facts:")
        question2 = f'''
            参考输入: {userInput}和介绍内容：{intro}
            \n判断{userInput}是否是绿色企业，生成关于{userInput}是或不是绿色企业的3到5个定量事实的列表； 
            \n仅返回定量事实列表；
        '''
        quantFacts = qa_agent(question2)

        prev_ai_research = ""

        st.subheader("Previous Related AI Research:")
        with st.spinner("Researching Pevious Research"):
            prev_ai_research = st.session_state.qa.run(f'''
                \n参考绿色企业标准，写下: {userInput}是否满足这些标准。
            ''')
            st.write(prev_ai_research)

        st.subheader("Generating Recent Publications:")
        question3 = f'''
            参考输入: "{userInput}".
            \n参考简介: "{intro}",
            \n参考事实: "{quantFacts}"
            \n生成3-5个关于{userInput}的主要内容.
            \n包含 标题, 链接, 摘要. 
        '''
        papers = qa_agent(question3)

        st.subheader("Generating Reccomended Books:")
        question4 = f'''
            参考输入: "{userInput}".
            \n参考简介: "{intro}",
            \n参考事实: "{quantFacts}"
            \n生成3-5个关于{userInput}的一些判断数据.
        '''
        readings = qa_agent(question4)

        ytlinks = "www.baidu.com"
        insert_research(userInput, intro, quantFacts,
                        papers, readings, ytlinks, prev_ai_research)
        # research_text = [userInput, intro, quantFacts,
        #                papers, readings, ytlinks, prev_ai_research]

        #db = vectordb(research_text, collection_name=collection_name)


def generate_history():
    st.subheader("Generate History")

    st.dataframe(read_research_table())
    selected_input = st.selectbox(label="Previous User Inputs", options=[
                                  i for i in read_research_table().user_input])
    if st.button("Render Research") and selected_input:
        with st.expander("Rendered Previous Research", expanded=True):
            selected_df = read_research_table()
            selected_df = selected_df[selected_df.user_input == selected_input].reset_index(
                drop=True)

            st.subheader("User Input:")
            st.write(selected_df.user_input[0])

            st.subheader("Introduction:")
            st.write(selected_df.introduction[0])

            st.subheader("Quantitative Facts:")
            st.write(selected_df.quant_facts[0])

            st.subheader("Previous Related AI Research:")
            st.write(selected_df.prev_ai_research[0])

            st.subheader("Recent Publications:")
            st.write(selected_df.publications[0])

            st.subheader("Recommended Books:")
            st.write(selected_df.books[0])

            st.subheader("Web Links:")
            st.write(selected_df.ytlinks[0])


msgs = StreamlitChatMessageHistory()

if "agent" not in st.session_state:
    create_research_db()
    st.session_state.agent = react_agent(msgs)
    st.session_state.qa = qa_retrieval(rsd=False, collection_name="test")


def update(tools, collection_name):
    st.session_state.qa = qa_retrieval(
        rsd=False, collection_name=collection_name)
    st.session_state.agent = react_agent(msgs, tools.append(collection_name))


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
        msgs.add_ai_message("我会根据以上标准帮你对企业进行评价。")

    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)


