# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:12:50 2023

@author: shangfr
"""
import os
import tempfile
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.llms import QianfanLLMEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

llm = QianfanLLMEndpoint()


def parse_url(urls):
    loader = AsyncHtmlLoader(urls)
    docs_html = loader.load()

    html2text = Html2TextTransformer()
    docs = html2text.transform_documents(docs_html)

    return docs


def doc_splits(file, chunk_size=1000, urls=[]):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=int(chunk_size/10))

    if urls:
        docs = parse_url(urls)
        splits = text_splitter.split_documents(docs)
    else:
        # Read documents
        temp_dir = tempfile.TemporaryDirectory()

        temp_filepath = os.path.join(temp_dir.name, file.name)

        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        if file.name.endswith('.csv'):
            from langchain.document_loaders.csv_loader import CSVLoader
            loader = CSVLoader(temp_filepath, encoding='utf-8')
            splits = loader.load()
            for spl in splits:
                spl.metadata['source'] = file.name
        else:
            loader = UnstructuredFileLoader(temp_filepath)
            docs = loader.load()
            docs[0].metadata['source'] = file.name

            # Split documents
            splits = text_splitter.split_documents(docs)

    return splits


def vectordb(file=None, splits=None, urls=[], chunk_size=1000, collection_name="test"):
    persist_directory = "./chroma"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if file:
        splits = doc_splits(file, chunk_size)

    if urls:
        splits = doc_splits(file, chunk_size, urls)

    if splits:
        if isinstance(splits[0], str):
            db = Chroma.from_texts(
                splits, embeddings, collection_name=collection_name, persist_directory=persist_directory)
        else:
            db = Chroma.from_documents(
                splits, embeddings, collection_name=collection_name, persist_directory=persist_directory)
        print(f"Vector DB 更新成功！  collection_name = {collection_name}")
    else:
        db = Chroma(collection_name, embeddings, persist_directory)
        print(f"Vector DB 加载成功！  collection_name = {collection_name}")

    return db


def qa_retrieval(k=1, rsd=True, prompt_template="", db=None, collection_name="test"):

    if not prompt_template:
        prompt_template = """将问题分解成若干个简单问题，参考以下内容然后逐个回答。确保答案正确，不要太啰嗦。
        
        {context}
        
        问题: {question}
        """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    # Define retriever
    # 加载本地数据库
    if not db:
        db = vectordb(collection_name=collection_name)

    retriever = db.as_retriever(search_type="mmr", search_kwargs={
                                "k": k, "fetch_k": 10})

    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever,
                                     chain_type_kwargs=chain_type_kwargs, return_source_documents=rsd)
    #qa({"query": query})
    return qa


#qa = qa_retrieval()
#results = qa({"query": query})


def summarize(docs):
    if len(docs) > 5000:
        print("ERNIE-Bot-turbo当前只支持6k输入")
        #docs = docs[:3]

    # Define prompt
    prompt_template = """现在你是一位资深投资人，参考以下内容，写一篇分析报告，要求包含以下内容。其中在企业绿色评价部分，你需要从可持续发展、成长性、收益性等方面切入。需要结合权威机构发布的数据。内容详实，有层次结构。
    "内容：{text}"
    报告:"""
    prompt = PromptTemplate.from_template(prompt_template)

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define StuffDocumentsChain
    # stuff_chain = StuffDocumentsChain(llm_chain=llm_chain)
    return llm_chain.run(docs)


def qa_keywords(query,text):
    # Define prompt
    prompt_template = """参考以下内容，请回答{query}。
    "内容：{text}"
    回答:"""
    prompt = PromptTemplate.from_template(prompt_template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    return llm_chain({"query":query,"text":text})