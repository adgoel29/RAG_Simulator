from langchain_community.vectorstores import FAISS
import logging
from Faiss.Fsearch import FAISSSearchEngine

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 

class FAISSBackend:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vectorstore = None
        self.search_engine = None

    def build_index(self, chunks):
        try:
            if not chunks:
                raise ValueError("No chunks provided")
            if not isinstance(chunks, list):
                raise TypeError("Chunks must be a list")
            self.vectorstore = FAISS.from_texts(texts=chunks, embedding=self.embeddings)
            self.search_engine = FAISSSearchEngine(self.vectorstore)
            logger.info(f"FAISS: indexed {len(chunks)} chunks")
        except (ValueError, TypeError):
            logger.exception("Invalid input")
            raise
        except Exception as e:
            logger.exception(f"FAISS indexing failed: {e}")
            raise

    def search(self, query, method, topk):
        try:
            if self.vectorstore is None:
                raise ValueError("Call ingest() first")
            if not query:
                raise ValueError("Empty query")
            if topk <= 0:
                raise ValueError("topk must be > 0")
            if self.search_engine is None:
                raise ValueError("Search engine not initialized")
            return self.search_engine.search_rag(query, topk, method)
        except ValueError:
            logger.exception("Invalid search input")
            raise
        except Exception as e:
            logger.exception(f"FAISS search failed: {e}")
            raise