from .weather_node import weather_tool_node
from .rag_node import rag_tool_node
from .final_response_node import generate_response_node
from .router_node import route_query, router_node
from langgraph.graph import StateGraph, END
from .schema import AgentState
from langsmith import traceable


def create_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("weather_tool", weather_tool_node)
    workflow.add_node("rag_tool", rag_tool_node)
    workflow.add_node("generate_response", generate_response_node)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add conditional edges from router
    workflow.add_conditional_edges(
        "router",
        route_query,
        {
            "weather_tool": "weather_tool",
            "rag_tool": "rag_tool"
        }
    )
    
    # Add edges to response generation
    workflow.add_edge("weather_tool", "generate_response")
    workflow.add_edge("rag_tool", "generate_response")
    
    # Add edge to end
    workflow.add_edge("generate_response", END)
    
    # Compile the graph
    app = workflow.compile()
    return app


@traceable
def invoke_runnable(agent, prompt, intent = "", weather_data = {}, pdf_contexxt = "", final_response = ""):
    result = agent.invoke({
                "query": prompt,
                "intent": intent,
                "weather_data": weather_data,
                "pdf_context": pdf_contexxt,
                "final_response": final_response
            })
    return result