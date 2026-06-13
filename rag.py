from langchain_huggingface import HuggingFaceEmbeddings
from Faiss.faiss_backend import FAISSBackend
from Qdrant.qdrant_backend import QdrantBackend
from chunking import chunk
from reranker import Reranker
import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class RagClass:
    def __init__(self, model_name, backend="faiss", **kwargs):
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
            self.tochunk = chunk(self.embeddings)

            if backend == "faiss":
                self.backend = FAISSBackend(self.embeddings)
            elif backend == "qdrant":
                self.backend = QdrantBackend(self.embeddings, **kwargs)
            else:
                raise ValueError(f"Unknown backend '{backend}'. Choose faiss or qdrant.")

            logger.info(f"RAG initialized with backend='{backend}'")
        except Exception as e:
            logger.exception(f"RAG init failed: {e}")
            raise

    def ingest(self, chunks):
        try:
            if not chunks:
                raise ValueError("No chunks provided")
            if not isinstance(chunks, list):
                raise TypeError("Chunks must be a list")
            self.backend.build_index(chunks)
            logger.info(f"Ingested {len(chunks)} chunks")
        except (ValueError, TypeError):
            logger.exception("Invalid ingestion input")
            raise
        except Exception as e:
            logger.exception(f"Ingestion failed: {e}")
            raise

    def searching(self, query, method="similarity", topk=5):
        try:
            if not query:
                raise ValueError("Empty query")
            if topk <= 0:
                raise ValueError("topk must be > 0")
            logger.info(f"Searching query='{query}' method='{method}' topk={topk}")
            docs = self.backend.search(query, method, topk)
            if not docs:
                logger.warning("No documents retrieved")
                return []
            return [doc.page_content for doc in docs]
        except ValueError:
            logger.exception("Invalid search input")
            raise
        except Exception as e:
            logger.exception(f"Search failed: {e}")
            return []

    def chunking(self, path, method, **kwargs):
        try:
            with open(path, 'r', encoding="utf-8") as f:
                content = f.read()
            if not content:
                raise ValueError("File is empty")
            if not isinstance(content, str):
                raise TypeError("Content must be a string")
            logger.info(f"Chunking '{path}' with method='{method}'")
            if method == "semantic":
                return self.tochunk.semantic_chunking(content)
            elif method == "fixed":
                return self.tochunk.fixed(content, **kwargs)
            else:
                raise ValueError(f"Unknown chunking method '{method}'")
        except FileNotFoundError:
            logger.exception("File not found")
            return []
        except (ValueError, TypeError):
            logger.exception("Invalid chunking input")
            return []
        except Exception as e:
            logger.exception(f"Chunking failed: {e}")
            return []

    def rerank(self, query, docs, topk=3,
               model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            if not query or not docs:
                raise ValueError("query and docs are required")
            return Reranker(model_name=model_name).reranking_docs(docs, query, topk)
        except ValueError:
            logger.exception("Invalid rerank input")
            raise
        except Exception as e:
            logger.exception(f"Reranking failed: {e}")
            raise