# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:17:50 2023

@author: shangfr
"""
import re
from typing import List, Union
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
from langchain.agents import AgentExecutor, LLMSingleActionAgent, AgentOutputParser

from langchain.tools import Tool, BaseTool, DuckDuckGoSearchRun
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish, OutputParserException

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

    # my_tools.append(CircumferenceTool())

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


class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\n感知: {observation}\n思考: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        llm_output = llm_output.replace("：", ":")
        if "答案:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("答案:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"行动\s*\d*\s*:(.*?)\n执行\s*\d*\s*:(.*?)\n观察\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if match:
            #raise OutputParserException(f"无法解析大模型输出: `{llm_output}`")
            action = match.group(1).strip().strip('"')
            action_input = match.group(2).strip().strip('"')
            # Return the action and action input
            return AgentAction(tool=action, tool_input=action_input, log=llm_output)

        else:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": "🍃"+llm_output},
                log=llm_output,
            )

def custom_react_agent(msgs=None, options=["test"]):

    tools = get_tools_lst(options)
    tool_names = [tool.name for tool in tools]
    # Set up the base template
    template = """针对提问，你可以选择使用以下工具：

    {tools}

    请使用以下格式回答问题:

    提问: 回答问题
    思考: 根据给定的工具，考虑该做什么
    行动: 要采取的行动，应该是[{tool_names}]
    执行: 行动的输入
    观察: 执行的结果
    最终答案: 针对最初的提问，给出最终答案，包含相关知识及判断依据等。

    开始！在你给出最终答案后请立即停止，保证回答正确。

    之前对话的历史记录:
    {chat_history}

    提问: {input}
    {agent_scratchpad}"""
    prompt = CustomPromptTemplate(
        template=template,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "intermediate_steps", "chat_history"]
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    output_parser = CustomOutputParser()
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\n观察:"],
        allowed_tools=tool_names
    )

    if msgs:
        memory = ConversationBufferWindowMemory(
            k=5, memory_key="chat_history", chat_memory=msgs, return_messages=True)

    else:
        memory = ConversationBufferWindowMemory(k=5,
                                                memory_key="chat_history", return_messages=True)

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=False, memory=memory)

    return agent_executor


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
