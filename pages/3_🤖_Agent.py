# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:12:50 2023

@author: shangfr
"""
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

from vectorstores import qa_retrieval, summarize
from modules import custom_react_agent
from database import create_db, read_table, insert_table

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
        # st.write(answer)
    return answer


def chat_research():
    if content := st.chat_input(placeholder="请提问"):
        _ = qa_agent(content)


def generate_research():

    if cmp := st.chat_input(placeholder="输入企业名称", key="chat02"):

        st.info("1. 企业信息收集")
        question1 = f"{cmp}是一家企业吗？这家企业的主要产品和项目有哪些？给出详细说明。"
        intro = qa_agent(question1)

        st.info("2. 绿色企业判断")
        with st.spinner("Retrieval & Matching"):
            green_matching = st.session_state.qa.run(
                f"参考{intro}企业信息，查找相关的绿色标准，分析{cmp}是否满足这些标准。")
        with st.expander("√ Complete!"):
            st.success(green_matching)

        st.info("3. 绿色企业评价")
        question1 = f"参考绿色企业的标准，{cmp}公司是否是一家绿色企业？如果是，给出该公司在符合绿色企业标准的相关信息。"
        valuation = qa_agent(question1)

        st.info("4. 构建评分模型")
        question2 = f"根据绿色企业标准的相关信息，请给出{cmp}在每项标准下的得分，并给出最终评分。"
        rate = qa_agent(question2)

        insert_table((cmp, intro, green_matching, valuation, rate))


def md_report(value_dict):
    research_id = value_dict["research_id"]
    cmp = value_dict["cmp"]
    intro = value_dict["intro"]
    green_matching = value_dict["green_matching"]
    valuation = value_dict["valuation"]
    rate = value_dict["rate"]

    md_str = f'''# {cmp}绿色评估报告\n
📝 **简介：**{intro}\n
🖊️ 绿色判断：{green_matching}\n
⭐ 绿色评估：{valuation}\n
⭐ 绿色评分：{rate}\n
'''
    report = summarize(md_str)
    insert_table((research_id, cmp, report))

    return report


def generate_history():
    df = read_table()
    st.dataframe(df, hide_index=True)
    col1, col2 = st.columns([9, 1])
    options = df["cmp"].tolist()
    options_id = df["research_id"].tolist()
    index = col1.selectbox("Previous User Inputs", range(
        len(options)), format_func=lambda x: options[x])

    if index:
        df_r = read_table(options_id[index])
        for rstr in df_r["report"]:
            with st.expander(f"📝 查看{options[index]}报告"):
                st.markdown(rstr)
    if col2.button("生成企业报告"):
        with st.expander(f"⭐ 生成{options[index]}报告", expanded=True):
            selected_df = df.loc[df["research_id"] == options_id[index]]
            md_str = md_report(selected_df.to_dict('records')[0])
            st.markdown(md_str)


msgs = StreamlitChatMessageHistory()

if "agent" not in st.session_state:
    create_db()
    st.session_state.agent = custom_react_agent(msgs)
    st.session_state.qa = qa_retrieval(rsd=False, collection_name="test")


def update(tools, collection_name):
    st.session_state.qa = qa_retrieval(
        rsd=False, collection_name=collection_name)
    st.session_state.agent = custom_react_agent(
        msgs, tools.append(collection_name))


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
    
    if len(msgs.messages) == 0 or st.sidebar.button("清空聊天记录", use_container_width=True):
        msgs.clear()

    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)
        
    if method == mts[0]:
        chat_research()

    elif method == mts[1]:
        generate_research()

