from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
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