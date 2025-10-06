from typing import TypedDict


class AgentState(TypedDict):
    query: str
    intent: str
    weather_data: dict
    pdf_context: str
    # pdf_context: list
    final_response: str