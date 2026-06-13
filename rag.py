from langchain_huggingface import HuggingFaceEmbeddings
from Faiss.faiss_backend import FAISSBackend
from Qdrant.qdrant_backend import QdrantBackend
from loadfile import get_content
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
    def __init__(self, model_name):
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
            self.tochunk = chunk(self.embeddings)
            self._backends = {}
            logger.info("RAG initialized successfully")
        except Exception as e:
            logger.exception(f"RAG init failed: {e}")
            raise

    def get_backend(self, backend_name: str):
        backend_name = backend_name.lower()
        if backend_name not in self._backends:
            if backend_name == "faiss":
                self._backends["faiss"] = FAISSBackend(self.embeddings)
            elif backend_name == "qdrant":
                self._backends["qdrant"] = QdrantBackend(self.embeddings)
            else:
                raise ValueError(f"Unknown backend '{backend_name}'. Choose faiss or qdrant.")
        return self._backends[backend_name]

    def ingest(self, chunks, backend_name="faiss"):
        try:
            if not chunks:
                raise ValueError("No chunks provided")
            if not isinstance(chunks, list):
                raise TypeError("Chunks must be a list")
            backend = self.get_backend(backend_name)
            backend.build_index(chunks)
            logger.info(f"Ingested {len(chunks)} chunks into {backend_name}")
        except (ValueError, TypeError):
            logger.exception("Invalid ingestion input")
            raise
        except Exception as e:
            logger.exception(f"Ingestion failed: {e}")
            raise

    def searching(self, query, method="similarity", topk=5, backend_name="faiss"):
        try:
            if not query:
                raise ValueError("Empty query")
            if topk <= 0:
                raise ValueError("topk must be > 0")
            logger.info(f"Searching query='{query}' method='{method}' topk={topk} using backend={backend_name}")
            backend = self.get_backend(backend_name)
            docs = backend.search(query, method, topk)
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
            # with open(path, 'r', encoding="utf-8") as f:
            #     content = f.read()
            content = get_content(path)
            logger.info(f"Content type: {type(content)} ")
            # print(f"Content {content[:100]}...")
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
