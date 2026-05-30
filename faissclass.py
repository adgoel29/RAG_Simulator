from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from search import SearchEngine
class RagClass:
    def __init__(self,model_name):
        self.vectorstore = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name#"sentence-transformers/all-MiniLM-L6-v2"
        )
    def ingest(self, chunks):
        self.vectorstore = FAISS.from_texts(
            texts=chunks,
            embedding=self.embeddings
        )
    def search(self, query, topk=5):
        if self.vectorstore is None:
            raise ValueError("Index not built. Call ingest() first.")
        search = SearchEngine(self.vectorstore)
        method = input(
            "Choose search method (similarity(1) / mmr(2)): "
        ).strip().lower()

        docs = search.search(
            query=query,
            method=method
        )
        return docs