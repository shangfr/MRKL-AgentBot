# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 10:00:31 2023

@author: shangfr
"""
import fitz
import uuid
import pandas as pd

doc = fitz.open("files/广州绿色企业认定指引.pdf")  # open example file

tab_dict = {}
state_dict = {}

def extract2dict(key,tab):
    extract = tab.extract() 
    if key in tab_dict:
        if extract[0] == tab_dict[key][0]:
            # 跨页表格，表头字段名连续
            extract = extract[1:]
        tab_dict[key].extend(extract)
    else:    
        tab_dict[key] = extract

def calculate_key(p_num,tab_names):
    key = str(uuid.uuid3(uuid.NAMESPACE_DNS, f"{str(tab_names)}"))
    if tab_dict:
        if p_num <= state_dict["p"]+1 and len(tab_names) == state_dict["cnt"]: 
            # 跨页表格，表头字段名不连续
            p_num = state_dict["p"]
            key = state_dict["names"]
            
    state_dict["p"] = p_num
    state_dict["cnt"] = len(tab_names)
    state_dict["names"] = key
            
    return f"{p_num}_"+key
                  

for page in doc:
    print(page.number)
    tabs = page.find_tables()
    if len(tabs.tables) > 0:
        for tab in tabs:
            key = calculate_key(page.number,tab.header.names)
            extract2dict(key,tab)


def cell2df(cell):
    df = pd.DataFrame(cell).ffill().bfill()
    df.columns = df.iloc[0]
    df.drop(0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df.replace(r"[\n ]", "",regex=True)



cells = list(tab_dict.values())
df = cell2df(cells[0])
df1 = cell2df(cells[1])

df.to_csv("files/test.csv", index=False)
df1.to_csv("files/test1.csv", index=False)
