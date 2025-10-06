from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.retrievers import BaseRetriever


class CEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", top_n: int =  5):
        model = HuggingFaceCrossEncoder(model_name=model_name)
        self.compressor = CrossEncoderReranker(model=model, top_n=top_n)
    
    def get_reranked_retriever(self, retriever: BaseRetriever) -> ContextualCompressionRetriever:
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor, base_retriever=retriever
        )
        return compression_retriever