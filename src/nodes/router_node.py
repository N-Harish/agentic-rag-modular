from .schema import AgentState
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from typing import Literal
from langsmith import traceable


_llm = None
_model_name = None

def get_groq_agent(groq_model_name: str = "openai/gpt-oss-20b"):
    """
    Lazily initialize and return a ChatGroq instance.
    If a different model name is requested than the one currently in use,
    re-initialize the client with the new model.
    """
    global _llm, _model_name

    # If we have an instance and the model requested is the same, return it
    if _llm is not None and _model_name == groq_model_name:
        return _llm

    _llm = ChatGroq(model=groq_model_name, temperature=0)
    _model_name = groq_model_name
    return _llm


@traceable
def router_node(state: AgentState) -> AgentState:
    llm = get_groq_agent("openai/gpt-oss-20b")
    query = state["query"]
    # prompt = ChatPromptTemplate(
    #     [
    #         ("system", """You are a routing assistant. Analyze the user query and determine if it's:
    #          - "weather": Questions about weather, temperature, climate conditions, forecast
    #          - "pdf": Questions about document content, information from uploaded files, summarizing documents
    #         Respond with ONLY one word: either "weather" or "pdf"."""),
    #         ("human", "{query}")
    
    #     ]
    # )

    prompt = ChatPromptTemplate(
        [
            ("system", """You are a routing assistant. Analyze the user query and determine if it's:
             - "weather": Questions about weather, temperature, climate conditions, forecast, humidity, precipitation, storms, wind, atmospheric conditions
             - "pdf": Questions about document content, information from uploaded files, summarizing documents, extracting text from files
            
            Respond with ONLY one word: either "weather" or "pdf"."""),
            ("human", "{query}")
        ]
    )
    chain = prompt | llm
    response = chain.invoke({"query": query})
    intent = response.content.strip().lower()
    return AgentState(
        query=state["query"],
        pdf_context=state["pdf_context"],
        weather_data=state["weather_data"],
        final_response=state["final_response"],
        intent=intent
    )


@traceable
def route_query(state: AgentState) -> Literal["weather_tool", "rag_tool"]:
    intent = state.get("intent", "pdf")
    
    if intent == "weather":
        return "weather_tool"
    else:
        return "rag_tool"

# query = "What's the name of person who found America"
# chain = prompt | llm
# response = chain.invoke({"query": query})
# intent = response.content.strip().lower()
# print(intent)