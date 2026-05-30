from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
class RagClass:
    def __init__(self):
        self.vectorstore = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    def ingest(self, chunks):
        self.vectorstore = FAISS.from_texts(
            texts=chunks,
            embedding=self.embeddings
        )
    def search(self, query, topk=5):
        if self.vectorstore is None:
            raise ValueError("Index not built. Call ingest() first.")
        docs = self.vectorstore.similarity_search(query, k=topk)
        return [doc.page_content for doc in docs]