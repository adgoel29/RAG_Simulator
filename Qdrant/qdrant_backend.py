from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.documents import Document
from  Qdrant.Qsearch import QdrantSearchEngine
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 

class QdrantBackend:
    def __init__(self, embeddings, mode="memory", url="http://localhost:6333"):
        self.embeddings = embeddings
        self.chunks = []
        self.vectorstore = None
        self.search_engine = None

        try:
            self.client = QdrantClient(":memory:") if mode == "memory" else QdrantClient(url=url)
            logger.info(f"Qdrant client initialized in '{mode}' mode")
        except Exception as e:
            logger.exception(f"Qdrant client init failed: {e}")
            raise

    def _make_collection(self, name, vector_size):
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

    def build_index(self, chunks):
        try:
            if not chunks:
                raise ValueError("No chunks provided")
            if not isinstance(chunks, list):
                raise TypeError("Chunks must be a list")

            self.chunks = chunks
            vector_size = len(self.embeddings.embed_query("test"))
            self._make_collection("rag", vector_size)

            # add texts manually — avoids the internal client conflict
            vectors = self.embeddings.embed_documents(chunks)
            self.client.upload_collection(
                collection_name="rag",
                vectors=vectors,
                payload=[{"page_content": c} for c in chunks]
            )

            # now construct vectorstore pointing at the existing collection
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name="rag",
                embedding=self.embeddings,
            )
            logger.info(f"Qdrant: indexed {len(chunks)} chunks")
            self.search_engine = QdrantSearchEngine(self.vectorstore)

        except (ValueError, TypeError):
            logger.exception("Invalid input")
            raise
        except Exception as e:
            logger.exception(f"Qdrant indexing failed: {e}")
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

        except (ValueError, ImportError):
            logger.exception("Search input error")
            raise
        except Exception as e:
            logger.exception(f"Qdrant search failed: {e}")
            raise