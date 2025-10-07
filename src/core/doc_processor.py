from langchain_unstructured import UnstructuredLoader
import os
import getpass
from dotenv import load_dotenv


class DocProcessor:
    def __init__(self, strategy: str = 'hi_res', partition_via_api: bool = True, coordinates: bool = True):
        self.strategy = strategy
        self.partition_via_api = partition_via_api
        self.coordinates = coordinates

    def load_pdf_splitter(self, filepath: str):
        loader = UnstructuredLoader(
            file_path=filepath,
            strategy=self.strategy,
            partition_via_api=self.partition_via_api,
            coordinates=self.coordinates,
        )
        return loader
    
    def process_pdf(self, filepath: str):
        loader = self.load_pdf_splitter(filepath=filepath)
        docs = []
        for doc in loader.lazy_load():
            docs.append(doc)
        return docs
