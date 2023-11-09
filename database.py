# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 17:38:58 2023

@author: shangfr
"""
import sqlite3
import pandas as pd

db_path = "chroma/chroma.sqlite3"
def create_research_db():
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Research (
                research_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_input TEXT,
                introduction TEXT,
                quant_facts TEXT,
                publications TEXT,
                books TEXT,
                ytlinks TEXT,
                prev_ai_research TEXT
            )
        """)


def create_messages_db():
    pass


def read_research_table():
    with sqlite3.connect(db_path) as conn:
        query = "SELECT * FROM Research"
        df = pd.read_sql_query(query, conn)
    return df



def insert_research(user_input, introduction, quant_facts, publications, books, ytlinks, prev_ai_research):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO Research (user_input, introduction, quant_facts, publications, books, ytlinks, prev_ai_research)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_input, introduction, quant_facts, publications, books, ytlinks, prev_ai_research))


