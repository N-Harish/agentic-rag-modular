from .schema import AgentState
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv
import os
from langsmith import traceable



# load_dotenv(find_dotenv())
# os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')


# def get_groq_agent(groq_model_name: str = "openai/gpt-oss-20b"):
#     llm = ChatGroq(
#             model=groq_model_name
#         )
#     return llm


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

    _llm = ChatGroq(model=groq_model_name)
    _model_name = groq_model_name
    return _llm

@traceable
def generate_response_node(state: AgentState) -> AgentState:
    llm = get_groq_agent()

    query = state["query"]
    intent = state["intent"]

    if intent == "weather":
        weather_data = state.get("weather_data", {})
        if "error" in weather_data:
            state["final_response"] = f"Error fetching weather data: {weather_data['error']}"
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful weather assistant. Provide a natural, conversational response about the weather data."),
                ("human", "Query: {query}\n\nWeather Data: {weather_data}\n\nProvide a helpful response.")
            ])
            
            chain = prompt | llm
            response = chain.invoke({"query": query, "weather_data": str(weather_data)})
            return AgentState(
                query=state["query"],
                intent=state["intent"],
                weather_data=state["weather_data"],
                final_response=response.content,
                pdf_context=state["pdf_context"]
            )
    elif intent == "pdf":
        context = state.get("pdf_context", "")
        print(f"context:- {context}")

        relevance_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a relevance checker. Determine if the provided context contains information that can answer the question.

        Respond with ONLY 'RELEVANT' or 'NOT_RELEVANT'.

        Guidelines:
        - If the context has information directly related to the question: RELEVANT
        - If the context is about completely different topics: NOT_RELEVANT
        - If unsure or context is vague: NOT_RELEVANT"""),
            ("human", "Context: {context}\n\nQuestion: {query}\n\nRelevance:")
        ])

        relevance_chain = relevance_prompt | llm | StrOutputParser()
        is_relevant = relevance_chain.invoke({
            "context": context, 
            "query": state["query"]
        }).strip().upper()

        print(f"Relevance check: {is_relevant}")

        # Step 2: Answer only if relevant
        if "NOT_RELEVANT" in is_relevant:
            return AgentState(
                query=state["query"],
                intent=state["intent"],
                weather_data=state["weather_data"],
                final_response="I don't have sufficient information in the provided context to answer this question.",
                pdf_context=state["pdf_context"]
            )   

        # prompt = ChatPromptTemplate.from_messages([
        #     ("system", "You are a helpful assistant. Answer the question based on the provided context."),
        #     ("human", "Context: {context}\n\nQuestion: {query}\n\nProvide a detailed answer.")
        # ])
        SYSTEM = """You are an assistant that MUST ONLY use the provided Context variable.
        Do NOT use external knowledge or invent facts. Produce a concise factual SUMMARY
        based only on the Context. The summary should be 3-6 sentences maximum.
        Do NOT add examples, speculation, or any facts that are not present in the Context.
        If you quote text from Context, wrap it in double quotes and prefix with (quote).
        Return only the summary text (no extra commentary)."""

        HUMAN = "Context: {context}\n\nQuestion: {query}\n\nPlease provide the SUMMARY as requested."

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SYSTEM),
            HumanMessagePromptTemplate.from_template(HUMAN),
        ])
        
        chain = prompt | llm
        response = chain.invoke({"query": query, "context": context})
        return AgentState(
            query=state["query"],
            intent=state["intent"],
            weather_data=state["weather_data"],
            final_response=response.content,
            pdf_context=state["pdf_context"]
        )
