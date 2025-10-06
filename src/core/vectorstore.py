from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_nomic import NomicEmbeddings
from langchain_core.documents import Document
from typing import List
import os
from .reranker import CEReranker


class VectorStoreQdrant:
    def __init__(self,  url: str , collection_name: str = "pdf-rag", prefer_grpc: bool = False, rerank: bool = True):
        self.url = url
        self.collection_name = collection_name
        self.prefer_grpc = prefer_grpc
        self.rerank = rerank
        self.embeddings = NomicEmbeddings(
            model="nomic-embed-text-v1.5",
            )

    def upsert_doc(self, docs: List[Document]):
        qdrant = QdrantVectorStore.from_documents(
            docs,
            self.embeddings,
            url=self.url,
            prefer_grpc=self.prefer_grpc,
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name=self.collection_name,
        )
        return qdrant
    
    def get_existing_vectorestore(self):
        print("Using QDrant URL :-", self.url)
        qdrant = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            collection_name=self.collection_name,
            url=self.url,
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=self.prefer_grpc,
        )
        return qdrant

    def similarity_search(self, text: str, vectorstore: QdrantVectorStore, top_k: int = 20):
        results = vectorstore.similarity_search(
            text, k=top_k
        )
        return results

    def get_retriever(self, top_k: int = 20, reranked_k: int = 5):
        vs = self.get_existing_vectorestore()
        if self.rerank:
            ce = CEReranker(top_n=reranked_k)
            return ce.get_reranked_retriever(vs.as_retriever(k=top_k, search_kwargs={'k':top_k}, search_type="mmr"))
        return vs.as_retriever(k=reranked_k, search_kwargs={'k':top_k}, search_type="mmr")
