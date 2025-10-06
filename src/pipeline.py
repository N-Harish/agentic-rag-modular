# from nodes import graph
# import os
# from dotenv import load_dotenv
# from core.doc_processor import DocProcessor
# from core.vectorstore import VectorStoreQdrant


# load_dotenv(override=True)
# print(os.getenv('QDRANT_URL'))

# dp = DocProcessor()
# docs = dp.process_pdf(r"C:\Users\haris\OneDrive\Desktop\agents_pipeline\pdf_and_weather_rag\Introduction to LangChain.pdf")
# vs = VectorStoreQdrant(url=os.getenv('QDRANT_URL'))
# vs.upsert_doc(docs)

# agent = graph.create_graph()

# agent.get_graph().print_ascii()
# result = agent.invoke({
#         "query": "What's the weather like in Dombivli?",
#         "intent": "",
#         "weather_data": {},
#         "pdf_context": "",
#         "final_response": ""
# })

# print("Weather Query Result:")
# print(result["final_response"])
# print("\n" + "="*50 + "\n")

# # Test with PDF query
# result = agent.invoke({
#     "query": "What is langchain?",
#     "intent": "",
#     "weather_data": {},
#     "pdf_context": "",
#     "final_response": ""
# })

# print("PDF Query Result:")
# print(result["final_response"])
