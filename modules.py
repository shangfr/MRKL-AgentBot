# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:17:50 2023

@author: shangfr
"""
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.llms import QianfanLLMEndpoint
from langchain.chains import LLMChain

from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool, BaseTool, DuckDuckGoSearchRun
from typing import Union
from vectorstores import qa_retrieval


class CircumferenceTool(BaseTool):
    name = "企业评分模型"
    description = "当需要计算企业绿色得分时，请使用此工具。"

    def _run(self, score: Union[int, float]):
        if score > 80:
            return "优秀"
        elif score > 60:
            return "一般"
        else:
            return "较差"

    def _arun(self, radius: int):
        raise NotImplementedError("This tool does not support async")


llm = QianfanLLMEndpoint(streaming=True)


def get_tools_lst(options):

    collection_name = options[-1]

    # 工具本地检索
    qa = qa_retrieval(rsd=False, collection_name=collection_name)
    search = DuckDuckGoSearchRun()

    my_tools = [
        Tool(
            name="网络搜索",
            func=search.run,
            description="使用此功能从网络搜索中查找企业信息，始终用作第一个工具。",
        ),
        Tool(
            name='政策信息查找工具',
            func=qa.run,
            description='使用此功能从文档存储中查找政策信息，只有在使用网络搜索工具后才可以用。'

        )]

    my_tools.append(CircumferenceTool())

    return my_tools


def _handle_error(error) -> str:
    return str(error)[:50]


def react_agent(msgs=None, options=["test"]):

    tools = get_tools_lst(options)

    if msgs:
        memory = ConversationBufferWindowMemory(k=5,
                                                memory_key="chat_history", chat_memory=msgs, return_messages=True)
    else:
        memory = ConversationBufferWindowMemory(k=5,
                                                memory_key="chat_history", return_messages=True)

    return initialize_agent(tools,
                            llm,
                            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                            verbose=True,
                            memory=memory,
                            handle_parsing_errors=True,
                            max_iterations=2,
                            early_stopping_method="generate",
                            )


def llm_chain(template, msgs=None):
    # Prompt
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(template),
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    # Notice that we `return_messages=True` to fit into the MessagesPlaceholder
    # Notice that `"chat_history"` aligns with the MessagesPlaceholder name
    if msgs:
        memory = ConversationBufferWindowMemory(k=5,
                                                memory_key="chat_history", chat_memory=msgs, return_messages=True)
    else:
        memory = ConversationBufferWindowMemory(k=5,
                                                memory_key="chat_history", return_messages=True)

    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )
    return conversation
