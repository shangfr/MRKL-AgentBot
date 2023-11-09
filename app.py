# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:35:34 2023

@author: shangfr
"""
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
st.set_page_config(page_title="Agent Bot", page_icon="🤖")
st.title("🤖 MRKL Agent Bot Research")
st.caption(
    "🦜🔗 LangChain is a framework for developing applications powered by language models.")

## styles ##
st.markdown(""" <style>
            #MainMenu {
                visibility: hidden;
            }
            footer {
                visibility: hidden;
            }
            </style> """,
            unsafe_allow_html=True)
############

with st.sidebar:
    st.link_button("Go to VectorDB", "/VectorDB", use_container_width=True)


'''
Agent System 是什么？

> Agents定义为大模型(LLM)、记忆(Memory)、任务规划(Planning Skills)以及工具使用(Tool Use)的集合，其中LLM是核心大脑，Memory、Planning Skills以及Tool Use等则是Agents系统实现的三个关键组件。构建AI Agent的工具箱已经相对完善，但仍存在一些限制，例如上下文长度、长期规划和任务分解，以及LLM能力的稳定性等。

1. 规划(Planning)

- **子目标和分解**：Agents能够将大型任务分解为较小的、可管理的子目标，以便高效的处理复杂任务；

- **反思和细化**：Agents可以对过去的行为进行自我批评和反省，从错误中吸取经验教训，并为接下来的行动进行分析、总结和提炼，这种反思和细化可以帮助Agents提高自身的智能和适应性，从而提高最终结果的质量。

2. 记忆(Memory)

- **短期记忆**：所有上下文学习都是依赖模型的短期记忆能力进行的；

- **长期记忆**：这种设计使得Agents能够长期保存和调用无限信息的能力，一般通过外部载体存储和快速检索来实现。

3. 工具使用(Tooluse)

- **Agents**可以学习如何调用外部API，以获取模型权重中缺少的额外信息，这些信息通常在预训练后很难更改，包括当前信息、代码执行能力、对专有信息源的访问等。

'''
