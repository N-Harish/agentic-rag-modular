from core.vectorstore import VectorStoreQdrant
from core.reranker import CEReranker
from .schema import AgentState
from langchain_nomic import NomicEmbeddings
import os
from dotenv import load_dotenv, find_dotenv
from langsmith import traceable

# load_dotenv(dotenv_path=".env", override=True, verbose=True)
print("RAG Node", os.getenv('QDRANT_URL'))
vs = VectorStoreQdrant(url=os.getenv('QDRANT_URL'))
ce_ranker = CEReranker()


def get_retriever():
    return vs.get_retriever(top_k=10)


@traceable
def rag_tool_node(state: AgentState) -> AgentState:
    retriever = get_retriever()
    query = state["query"]
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    # context = docs
    return AgentState(
        query=state["query"],
        intent=state["intent"],
        final_response=state["final_response"],
        weather_data=state["weather_data"],
        pdf_context=context
    )
