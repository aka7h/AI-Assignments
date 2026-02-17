# %%
# !pip install langgraph langchain langchain-community langchain-groq

# %%
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from IPython.display import Image, Markdown


# %%
from getpass import getpass
GROQ_API_KEY = getpass("Enter your Groq API key: ")

# %%
TAVILY_API_KEY = getpass('Enter Tavily Search API Key: ')

# %%
import os
os.environ['GROQ_API_KEY'] = GROQ_API_KEY
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

# %%
#defining systemprompt

# prompt_temp = """
# You are a business research assistant for a Max Fashion store. 
# Given a location you need to gather the competitor brand in that location, their foot fall trend, 
# peak hours and offers they provide. Based on these information you need to provide a short report to the store manager. 
# Get the most recent inforamtion available on the internet also the date. Always use the tools provided. 

# You must provide report only based on the information you have gathered. If user ask anything outside the scope, just reply "Not in Scope".
# If you don't have information about a particular aspect, you can say "Not Found". if the given location doesnt exist then reply "Location doesnt exist".
# Dont hallucinate and make up information. 
# """

prompt_temp = """
Act as a Max Fashion research assistant. For a given location, use tools to find: competitor brands, footfall trends, peak hours, and current offers. 
Provide a short report for the store manager including the data source date. If info is missing, say 'Not Found'. 
If the location is invalid, say 'Location doesn't exist'. For topics outside this scope, reply 'Not in Scope'. Do not hallucinate. Be concise."""

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",prompt_temp),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# %%
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
# llama-3.1-8b-instant
# openai/gpt-oss-120b


tavily_search = TavilySearchResults(max_results=3,max_tokens=1000)
tools = [tavily_search]
# llm_mod = (prompt_template | llm)

llm_mod = (prompt_template | llm.bind_tools(tools))

tool_node = ToolNode(tools)

# %%
def call_model(state: MessagesState):
    response = llm_mod.invoke(state['messages'])

    return {'messages': [response]}

# %%
graph = StateGraph(MessagesState)
graph.add_node("agent",call_model)
graph.add_node("tools",tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent",tools_condition, ["tools",END])
graph.add_edge("tools","agent")


# %%
r_graph = graph.compile()

# %%
Image(r_graph.get_graph().draw_mermaid_png())

# %%
output = r_graph.invoke({'messages': [
    ('user', 'Location : Marathahalli')
]})

Markdown(output['messages'][-1].content)

# %%
# output

# %%

output = r_graph.invoke({'messages': [
    ('user', 'Location : Jaya nagar, bangalore')
]})

Markdown(output['messages'][-1].content)


# %%
# output

# %%


# %%


# %%



