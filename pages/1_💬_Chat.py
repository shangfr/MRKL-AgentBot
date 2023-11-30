# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:35:34 2023

@author: shangfr
"""
import streamlit as st
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from utils import StreamHandler
from modules import llm_chain

st.set_page_config(page_title="Chat", page_icon="💬")
st.header("🍏 企业绿识别")
st.caption("🚀 文本内容总长度不能超过7000 token")
with st.sidebar:
    with st.expander("角色设置"):
        role_st = st.text_area(
            "",
            "你是一位研究绿色企业方面的专家。绿色企业是指以可持续发展为己任，将环境利益和对环境的管理纳入企业经营管理全过程，并取得成效的企业。",
            height=100)
    with st.expander("标准设置"):
        reference_st = st.text_area(
            "",
            "绿色企业的标准可以分为三个层次：基础标准、核心标准和高度标准。\n\n1. 基础标准：主要关注企业的环境合规性，包括企业的环境管理体系、环境法律法规的遵守情况、环境信息公开情况等。\n\n2. 核心标准：主要关注企业的绿色产品、绿色技术和绿色生产过程，包括企业的产品是否符合环保标准、技术是否具有节能减排特点、生产过程是否采用环保技术和清洁生产方式等。\n\n3. 高度标准：主要关注企业的绿色管理、绿色文化和社会责任，包括企业的环境战略、环保宣传和教育、员工的环保意识和参与度、企业的社会责任报告等。\n\n判断一家绿色企业通常会从上述方面进行综合评估，并结合企业的具体情况进行分析。",
            height=100)

    with st.expander("评分设置"):
        score_st = st.text_area(
            "",
            '''按照三个层次标准，可以分别对企业在各项标准上的表现进行评分，最后进行综合评分。每项标准在1~100分之间评分。\n\n绿色企业评分 = 基础标准 + 核心标准 + 高度标准''',
            height=100, help="总长度不能超过1000 token")

    on = st.toggle('自动执行')

template = f"{role_st}\n\n{reference_st}\n\n{score_st}"
msgs = StreamlitChatMessageHistory()
conversation = llm_chain(template, msgs)


if len(msgs.messages) == 0 or st.sidebar.button("清空聊天记录", use_container_width=True):
    msgs.clear()

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if cmp := st.chat_input(placeholder="提问或输入企业名称"):
    st.chat_message("user").write(cmp)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())

        if on:
            st.caption('🍏 绿识别已激活')

            question1 = f"{cmp}公司是否是一家绿色企业？参考绿色企业的标准，给出该公司在符合绿色企业标准的相关信息。"
            question2 = f"根据你掌握的信息，请对{cmp}公司建立一个绿色企业评分模型"
            question3 = f"请按上面评分标准对{cmp}公司进行评分，只需要给出各项分值。"

            response1 = conversation.run(
                {"question": question1}, callbacks=[stream_handler])
            response2 = conversation.run(
                {"question": question2}, callbacks=[stream_handler])
            response3 = conversation.run(
                {"question": question3}, callbacks=[stream_handler])
        else:
            conversation.run({"question": cmp}, callbacks=[stream_handler])
