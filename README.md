# 🤖 MRKL-AgentBot

> 🦜🔗 **LangChain**是一个方便开发人员使用大型语言模型 (LLM) 构建端到端应用程序的开源框架。它提供了一套工具、组件和接口，使LLM具有上下文感知和语言推理能力。

## Agent System 是什么？

> 🚀 **Agents**是大模型(LLM)、记忆(Memory)、任务规划(Planning Skills)以及工具使用(Tool Use)的集合。

1. 规划（Planning）

    • 子目标和分解：AI Agents 能够将大型任务分解为较小的、可管理的子目标，以便高效的处理复杂任务；

    • 反思和细化：Agents 可以对过去的行为进行自我批评和反省，从错误中吸取经验教训，并为接下来的行动进行分析、总结和提炼，这种反思和细化可以帮助 Agents 提高自身的智能和适应性，从而提高最终结果的质量。

2. 记忆 （Memory）

    • 短期记忆：所有上下文学习都是依赖模型的短期记忆能力进行的；

    • 长期记忆：这种设计使得 AI Agents 能够长期保存和调用无限信息的能力，一般通过外部载体存储和快速检索来实现。

3. 工具使用（Tool use）

    • AI Agents 可以学习如何调用外部 API，以获取模型权重中缺少的额外信息，这些信息通常在预训练后很难更改，包括当前信息、代码执行能力、对专有信息源的访问等。

## MRKL 架构

[MRKL（Modular Reasoning, Knowledge and Language）](https://zhuanlan.zhihu.com/p/664133752)即“模块化推理、知识和语言”，是一种用于自主代理的大语言模型知识问答架构。MRKL架构的设计中使用ReAct(即推理+动作) 方法得到结果，就像人类从事一项需要多个步骤的才能完成的任务时，在每一步之间往往会有一个推理过程；使用LLM扮演工具选择的角色，通过工具描述找到最合适的工具API获取数据。在MRKL系统中，LLM通过与外部API交互以检索信息。当提出问题时，LLM可以选择执行操作以检索信息，最后根据检索到的信息回答问题。

## MRKL-AgentBot应用

**LLM**是开创性的，基于Agents的LLM系统所展现出的创造力带来了大量令人兴奋的新应用；使用**LangChain**框架提供的**Agent**模块可以快速开发一个**MRKL应用系统**。

我在使用LangChain框架的过程中发现，由于框架内模块(Chat、RetrievalQA、Agent)的PromptTemplate用的都是英语，可以很好的支持ChatGPT的语言环境和英文对话。但如果用国内的LLMs(百度千帆、阿里通义)API基于LangChain框架开发中文对话应用程序，则会出现中英文混合导致的模型输出质量严重下降。因此，必须对LangChain框架内涉及内置英文PromptTemplate的组件进行定制化修改。所以我用Streamlit开发了一个MRKL-AgentBot应用，通过研究对比，发现用中文替换PromptTemplate会极大的提升中文大模型输出的质量。同时加入一些Prompt 技巧，则会事半功倍。

### 什么是 Prompt 技巧？

Prompt 技巧是一种通过为模型提供更加具体和准确的上下文信息，来提高模型回答质量的方法。它允许我们在模型请求答案之前，向模型提供额外的信息或问题，以帮助模型更好地理解和回答问题。

### 如何使用 Prompt 技巧提高 LLMs 的回答准确率？

1. 明确问题：在提问时，尽可能明确和具体化你的问题。例如，如果您想要求解数学问题，请确保提供足够的信息，如题目、公式等。
2. 质量要求：在提问时，要求LLMs仔细检查答案，以确保其准确性和完整性。
3. 提供上下文：为了帮助LLMs更好地理解问题，可以提供相关的背景知识或上下文信息。例如，如果您询问有关科学领域的问题，可以提供相关的概念、理论或实验数据等。
4. 使用引导词：在问题中使用引导词，如“请问……”、“如何……”等，以帮助LLMs更好地理解您的问题并给出准确的回答。
5. 逐层提问：如果需要询问较为复杂的问题，建议逐层提问，避免一次性提出过多信息导致LLMs理解困难。



以上的几个技巧，在Agent在运行过程中可以自主解决，但目前效果没有手动设置的好。
比如，提供上下文的手动设置方式：本地知识向量化，通过[**🦜️🔗 LangChain**](https://python.langchain.com/)快速将本地`原数据` - `模型` - `向量数据库`-`检索`整个**Pipline流程**贯通。

![data_connection_diagram](https://python.langchain.com/assets/images/data_connection-c42d68c3d092b85f50d08d4cc171fc25.jpg)

![vector store diagram](https://python.langchain.com/assets/images/vector_stores-9dc1ecb68c4cb446df110764c9cc07e0.jpg)

### Usage 
####  Installation and requirements

1. 申请[千帆大模型API](https://cloud.baidu.com/product/wenxinworkshop?track=pinzhuanqianfan) 在demo.env里添加环境变量后，重命名demo.env 为 .env
2. `pip install -r requirements.txt`
3. `streamlit run app.py`

### 👇Resources

##### The Best Vector Databases for Storing Embeddings

- [Chroma](https://safjan.com/the-best-vector-databases-for-storing-embeddings/#chroma)

- [Faiss by Facebook](https://safjan.com/the-best-vector-databases-for-storing-embeddings/#faiss-by-facebook)
- [Milvus](https://safjan.com/the-best-vector-databases-for-storing-embeddings/#milvus)
- [pgvector](https://safjan.com/the-best-vector-databases-for-storing-embeddings/#pgvector)
- [Pinecone](https://safjan.com/the-best-vector-databases-for-storing-embeddings/#pinecone)
- [Supabase](https://safjan.com/the-best-vector-databases-for-storing-embeddings/#supabase)


#### SentenceTransformers

word2vec、glove是两种静态的词向量模型，即每个词语只有一个固定的向量表示。但在不同语境中，词语的语义会发生变化，按道理词向量也应该动态调整。相比word2vec、glove生成的静态词向量， BERT、ERNIE是一种动态的技术，可以根据上下文情景，得到语义变化的词向量。HuggingFace网站提供了简易可用的数据集、丰富的预训练语言模型， 通过sentence-transformer库，可以使用预训练模型，得到不同情景的文本的语义向量。