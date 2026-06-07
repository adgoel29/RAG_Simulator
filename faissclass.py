from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from search import SearchEngine
from reranker import Reranker
from chunking import chunk

import logging


logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RagClass:
    def __init__(self,model_name):
        self.vectorstore = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name#"sentence-transformers/all-MiniLM-L6-v2"
        )
        self.tochunk=chunk(self.embeddings)
        self.tosearch=None
        logger.info(
            "RAG initialized successfully"
        )
    def ingest(self, chunks):
        try:

            if not chunks:
                raise ValueError("No chunks provided")

            if not isinstance(chunks, list):
                raise TypeError(
                    "Chunks should be a list"
                )


            self.vectorstore = FAISS.from_texts(
                texts=chunks,
                embedding=self.embeddings
            )

            logger.info(
            f"Stored {len(chunks)} chunks"
        )


        except (ValueError,TypeError):

            logger.exception(
                "Invalid ingestion input"
            )

            raise

        except Exception as e:
            logger.exception(f"FAISS indexing failed: {e}")
            raise

    def searching(self, query,method, topk=5):
        try:

            if self.vectorstore is None:
                raise ValueError("Index not built. Call ingest() first.")
            self.tosearch = SearchEngine(self.vectorstore)

            if not query:
                raise ValueError(
                    "Empty query"
                )
            
            if topk <= 0:
                raise ValueError(
                    "Invalid topk"
                )
            
            logger.info(
                f"Searching for query: {query} with method: {method} and topk: {topk}"
            )
            docs = self.tosearch.search_rag(
                query=query,
                topk=topk,
                method=method
            )
            if not docs:
                logger.warning(
                    "No documents retrieved"
                )

                return []
            
            logger.info(
                f"Retrieved {len(docs)} documents for query: {query}"
            )
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.exception(f"Search failed: {e}")
            return []
        
    def chunking(self,path,method,**kwargs):
        try:

            with open(path,'r',encoding="utf-8") as f:
                content=f.read()
            if not content:
                raise ValueError("No content provided")
            if not isinstance(content, str):
                raise TypeError("Content must be string")
            logger.info(
                f"Chunking content from {path} using method: {method}"
            )
            if method=="semantic":
                chunks=self.tochunk.semantic_chunking(content)
            elif method=="fixed":
                chunks=self.tochunk.fixed(content,**kwargs)
            else:
                pass
            return chunks
        except FileNotFoundError:

            logger.exception(
                "File missing"
            )

            return []

        except Exception as e:
            logger.exception(f"Chunking failed: {e}")
            return []
        
        
    def rerank(self,query,docs,topk=3,model_name=None):
        if not query or not docs or not model_name:
            raise ValueError("query,docs and model_name are required")
        obj=Reranker(model_name=model_name)
        reranked_docs=obj.reranking_docs(docs,query,topk)
        return reranked_docs