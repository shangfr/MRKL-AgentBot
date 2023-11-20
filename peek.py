# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:17:40 2023

@author: shangfr
"""
import chromadb
import jieba.analyse
import pandas as pd


class ChromaPeek:
    def __init__(self, path):
        self.client = chromadb.PersistentClient(path)

    # function that returs all collection's name
    def get_collections(self):
        collections = []

        for i in self.client.list_collections():
            collections.append(i.name)

        return collections

    # function to return documents/ data inside the collection
    def get_collection_data(self, collection_name, dataframe=False):
        data = self.client.get_collection(name=collection_name).get(
            include=['metadatas', 'documents'])
        if dataframe:
            df = pd.DataFrame(data)
            df["delete"] = df.shape[0] * [False]
            orders = ["delete", 'documents', 'metadatas', 'ids']
            return df[orders]
        return data

    # function to query the selected collection
    def query(self, query_str, collection_name, k=3, dataframe=False):
        
        keywords = jieba.analyse.extract_tags(query_str, topK=3)
        
        if len(keywords)>1:
            key_lst = [{"$contains": kw} for kw in keywords]
            where_document = {"$or": key_lst}
        else:
            where_document = {"$contains": query_str}
            
        collection = self.client.get_collection(collection_name)
        res = collection.query(
            query_texts=[query_str], n_results=k,
            where_document=where_document
        )
        out = {}
        for key, value in res.items():
            if value:
                out[key] = value[0]
            else:
                out[key] = value
        if dataframe:
            df = pd.DataFrame(out)
            orders = ['documents', 'metadatas', 'ids']
            return df[orders]
        return out

    def delete(self, ids, collection_name):
        collection = self.client.get_collection(collection_name)
        collection.delete(ids)
