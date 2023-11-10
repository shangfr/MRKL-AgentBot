# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 17:38:58 2023

@author: shangfr
"""
import sqlite3
import pandas as pd

db_path = "chroma/chroma.sqlite3"


def create_db():
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS research (
                research_id INTEGER PRIMARY KEY AUTOINCREMENT,
                cmp TEXT,
                intro TEXT,
                green_matching TEXT,
                valuation TEXT,
                rate TEXT
            )
        """)

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS report (
                report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                research_id INTEGER,
                cmp TEXT,
                report TEXT
            )
        """)


def read_table(ids=None):
    if ids:
        query = f"SELECT * FROM report WHERE research_id={ids}"
    else:
        query = "SELECT * FROM research"

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)
    return df


def insert_table(tuple_data):

    if len(tuple_data) == 5:
        sql = """
            INSERT INTO research (cmp, intro, green_matching, valuation, rate)
            VALUES (?, ?, ?, ?, ?)
        """
    elif len(tuple_data) == 3:
        sql = """
             INSERT INTO report (research_id, cmp, report)
             VALUES (?, ?, ?)
         """

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql, tuple_data)
