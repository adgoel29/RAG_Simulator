from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from search import SearchEngine
from reranker import Reranker
from chunking import chunk
class RagClass:
    def __init__(self,model_name):
        self.vectorstore = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name#"sentence-transformers/all-MiniLM-L6-v2"
        )
        self.tochunk=chunk(self.embeddings)
        self.tosearch=None
    def ingest(self, chunks):
        self.vectorstore = FAISS.from_texts(
            texts=chunks,
            embedding=self.embeddings
        )
    def searching(self, query,method, topk=5):
        if self.vectorstore is None:
            raise ValueError("Index not built. Call ingest() first.")
        self.tosearch = SearchEngine(self.vectorstore)
        docs = self.tosearch.search_rag(
            query=query,
            topk=topk,
            method=method
        )
        return [doc.page_content for doc in docs]
    
    def chunking(self,path,method,**kwargs):
        with open(path,'r',encoding="utf-8") as f:
            con=f.read()
        content=con
        if method=="semantic":
            chunks=self.tochunk.semantic_chunking(content)
        elif method=="fixed":
            chunks=self.tochunk.fixed(content,**kwargs)
        else:
            pass
        return chunks
    
    def rerank(self,query,docs,topk=3,model_name=None):
        if not query or not docs or not model_name:
            raise ValueError("query,docs and model_name are required")
        obj=Reranker(model_name=model_name)
        reranked_docs=obj.reranking_docs(docs,query,topk)
        return reranked_docs