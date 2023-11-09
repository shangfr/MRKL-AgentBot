# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:12:50 2023

@author: shangfr
"""

import os
import tempfile
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.llms import QianfanLLMEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

llm = QianfanLLMEndpoint()


def doc_splits(file, chunk_size=1000, collection_name="test"):
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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=int(chunk_size/10))
        splits = text_splitter.split_documents(docs)

    return splits


def vectordb(file=None, splits=None, chunk_size=1000, collection_name="test"):
    persist_directory = "./chroma"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if file:
        splits = doc_splits(file, chunk_size, collection_name)

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
        prompt_template = """使用以下内容用中文回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。
        
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
    if len(docs) > 3:
        print("ERNIE-Bot-turbo当前只支持6k输入")
        docs = docs[:3]

    # Define prompt
    prompt_template = """请对以下各部分内容进行摘要总结：:
    "{text}"
    摘要总结:"""
    prompt = PromptTemplate.from_template(prompt_template)

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain)

    return stuff_chain.run(docs)
