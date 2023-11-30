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
from langchain_experimental.tools import PythonREPLTool
from vectorstores import qa_retrieval


class ScoreTool(BaseTool):
    name = "企业评分模型"
    description = "计算企业绿色得分。"

    def _run(self, score: Union[int, float]):
        if score > 80:
            return "优秀"
        elif score > 60:
            return "一般"
        else:
            return "较差"

    def _arun(self, radius: int):
        raise NotImplementedError("This tool does not support async")


class MyTool(BaseTool):
    name = "问题增强"
    description = "增强提问能力"

    def _run(self, question: str):
        res = llm(f"根据这个问题：{question}? \n 给出是3个相关问题。")
        return res
    
    
    def _arun(self, radius: int):
        raise NotImplementedError("This tool does not support async")


llm = QianfanLLMEndpoint(model="ERNIE-Bot",streaming=True)

def get_tools_lst(options):

    collection_name = options[-1]

    # 工具本地检索
    qa_green = qa_retrieval(k=3, rsd=False, collection_name="green")
    qa = qa_retrieval(k=3, rsd=False, collection_name=collection_name)
    search = DuckDuckGoSearchRun()
    python3 = PythonREPLTool()

    my_tools = [
        Tool(
            name="网络搜索",
            func=search.run,
            description="从网络搜索中搜索各种信息。",
           # return_direct=True,
        ),
        Tool(
            name='绿产目录',
            func=qa_green.run,
            description='输入产品名称，查找产品是否满足绿色产业目录要求。'

        ),
        Tool(
            name='新闻查找工具',
            func=qa.run,
            description='查找企业新闻。'

        ),
        Tool(
            name='python',
            func=python3.run,
            description='运行python程序。'

        ),  
        Tool(
            name='问题增强',
            func=MyTool().run,
            description='增强提问能力'
        ),   
        ]

    #my_tools.append()

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
        #print(intermediate_steps)
        thoughts = ""
        for action, observation in intermediate_steps:
            #thoughts += action.log
            if "not a valid tool" not in observation:
                thoughts += f"\n🍃信息：{observation}\n"

        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts

        if len(intermediate_steps)>2:
            kwargs["tools"] = "结合以上内容，按以下格式返回内容：\n初始问题：*** \n最终答案：*** "
            kwargs["tool_names"] = ""
            
        else:
            mytools = "\n".join(
                [f"{tool.name}: {tool.description}" for tool in self.tools])
            kwargs["tools"] = f"请从下面选择一项工具：\n\n{mytools}\n\n回答这个新问题，按以下格式返回内容：\n问题：*** \n工具：*** \n解释：***  \n；"
            kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
               
        ttm = self.template.format(**kwargs)

        return ttm


class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        #print("%%%%%%%%%%%"+llm_output+"%%%%%%%%")
        # Parse out the action and action input
        llm_output = llm_output.replace(":", "：")
        
        if "答案：" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("答案：")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"问题\s*\d*\s*：[\s]*(.*?)\n+工具\s*\d*\s*：[\s]*(.*)\n+解释\s*\d*\s*：[\s]*(.*)"

        matches = re.findall(regex, llm_output)
        if not matches:
            #raise ValueError(f"🍃Could not parse LLM output: `{llm_output}`")
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.strip()},
                log=llm_output,
            )
        match = matches[-1]
        action = match[1]
        action_input = match[0]#+"--"+match[2]
        
        tnames = ["网络搜索", "绿产目录", "新闻查找工具", "python", "问题增强"]
        if action not in tnames:
            for regex in tnames:
                if re.search(regex, action):
                    action = regex
                    break

        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)



def custom_react_agent(msgs=None, options=["test"]):
    tools = get_tools_lst(options)
    tool_names = [tool.name for tool in tools]
    # Set up the base template
    template = """对话记录：\n{chat_history}
{agent_scratchpad}
根据这个问题：{input}? \n 给出是1个相关问题。
{tools}
"""
    
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
        stop=["解释"],
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
        memory=memory,
        verbose=False,
    )
    return conversation
