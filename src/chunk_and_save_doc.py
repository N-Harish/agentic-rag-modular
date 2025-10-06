from nodes import graph
import os
from dotenv import load_dotenv
from core.doc_processor import DocProcessor
from core.vectorstore import VectorStoreQdrant


load_dotenv(override=True)
# print(os.getenv('QDRANT_URL'))

def insert_doc(pdf_file_path: str):
    dp = DocProcessor()
    docs = dp.process_pdf(pdf_file_path)
    vs = VectorStoreQdrant(url=os.getenv('QDRANT_URL'))
    vs.upsert_doc(docs)
