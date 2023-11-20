# flake8: noqa

PREFIX = """
你正在使用 Python 中的 pandas dataframe。dataframe的名称是`df`。  
你应该使用以下工具来回答你的问题："""

MULTI_DF_PREFIX = """
你正在 Python 中使用{num_dfs}个 pandas dataframe，名为 df1，df2 等。
你应该使用以下工具来回答你的问题："""

SUFFIX_NO_DF = """
开始！  
问题：{input}
{agent_scratchpad}"""

SUFFIX_WITH_DF = """
这是`print(df.head())`的结果:
{df_head}

开始！  
问题：{input}
{agent_scratchpad}"""

SUFFIX_WITH_MULTI_DF = """
这是每个dataframe的`print(df.head())`的结果：
{dfs_head}

开始！  
问题：{input}
{agent_scratchpad}"""

PREFIX_FUNCTIONS = """
你正在 Python 中使用 pandas dataframe，名为 `df`。"""

MULTI_DF_PREFIX_FUNCTIONS = """
你正在 Python 中使用{num_dfs}个 pandas dataframe，名为 df1，df2 等。"""

FUNCTIONS_WITH_DF = """
这是`print(df.head())`的结果：
{df_head}"""

FUNCTIONS_WITH_MULTI_DF = """
这是每个dataframe的`print(df.head())`的结果：
{dfs_head}"""
