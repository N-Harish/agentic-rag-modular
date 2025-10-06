from .schema import AgentState
from langchain import hub
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor, create_tool_calling_agent
import os
from dotenv import load_dotenv
from langchain_core.tools import tool, Tool
import requests
import json
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable


# load_dotenv(dotenv_path=".env")

@tool
def get_weather(city:str)->str:
    """
      This function fetches the current weather data for a given city
    """
    weather_api = os.getenv('OPENWEATHERMAP_API_KEY')
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api}&units=metric"
    response = requests.get(url)
    return response.json()


def weather_agent(groq_model_name: str = "openai/gpt-oss-20b"):
    llm = ChatGroq(
        model=groq_model_name,
        temperature=0
    )

    tools = [get_weather]
    print("Available tools:", [t.name for t in tools])
    
    # Create a simple prompt for tool calling
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", "You are a helpful assistant that can check weather information."),
    #     ("human", "{input}"),
    #     ("placeholder", "{agent_scratchpad}"),
    # ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful weather assistant. When users ask about weather:
            1. Extract the city name from their question
            2. Use the get_weather tool with the city name to fetch current weather data
            3. Present the weather information in a clear, friendly format

        If no city is mentioned, ask the user which city they want to check.
        Always use the get_weather tool to provide accurate, real-time weather information."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
        
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    agent = create_tool_calling_agent(llm_with_tools, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        verbose=True
    )
    
    return agent_executor


@traceable
def weather_tool_node(state: AgentState) -> AgentState:
    query = state['query']
    agent_executor =  weather_agent("openai/gpt-oss-120b")
    print(query)
    response = agent_executor.invoke({"input": query})
    return AgentState(
        query=state['query'],
        intent=state['intent'],
        pdf_context=state['pdf_context'],
        final_response=state['final_response'],
        weather_data=response
    )


# load_dotenv(dotenv_path="../.env")
# agent = weather_agent("openai/gpt-oss-20b")
# response = agent.invoke({"input": "Current weather in mumbai"})
# print(response)