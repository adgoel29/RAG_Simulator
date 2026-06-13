import os
import logging
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def build_embeddings():
    """
    Default embedding model for a local/demo setup.
    Swap this with OpenAIEmbeddings or your own embedding object if needed.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    model_name = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    return HuggingFaceEmbeddings(model_name=model_name)


class QdrantSearchEngine:
    def __init__(self, vectorstore: QdrantVectorStore):
        self.vectorstore = vectorstore

    def search_rag(self, query: str, topk: int = 5, method: str = "similarity"):
        if method == "similarity":
            return self.vectorstore.similarity_search(query, k=topk)

        if method == "mmr":
            return self.vectorstore.max_marginal_relevance_search(query, k=topk)

        return self.vectorstore.similarity_search(query, k=topk)


class QdrantBackend:
    def __init__(
        self,
        embeddings,
        mode: str = "server",
        url: str = "http://localhost:6333",
    ):
        self.embeddings = embeddings
        self.vectorstore: Optional[QdrantVectorStore] = None
        self.search_engine: Optional[QdrantSearchEngine] = None

        try:
            if mode == "memory":
                self.client = QdrantClient(":memory:")
            else:
                self.client = QdrantClient(url=url)

            logger.info(f"Qdrant client initialized in '{mode}' mode")
        except Exception as e:
            logger.exception(f"Qdrant client init failed: {e}")
            raise

    def list_collections(self) -> List[str]:
        try:
            collections = self.client.get_collections().collections
            return [c.name for c in collections]
        except Exception as e:
            logger.exception(f"Failed to list collections: {e}")
            raise

    def collection_exists(self, name: str) -> bool:
        try:
            self.client.get_collection(collection_name=name)
            return True
        except Exception:
            return False

    def _make_collection(
        self,
        name: str,
        vector_size: int,
        recreate: bool = False,
    ):
        try:
            exists = self.collection_exists(name)

            if exists and recreate:
                self.client.delete_collection(collection_name=name)
                exists = False

            if not exists:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"Created collection: {name}")
            else:
                logger.info(f"Using existing collection: {name}")

        except Exception as e:
            logger.exception(f"Failed to prepare collection '{name}': {e}")
            raise

    def _get_vectorstore(self, collection_name: str) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )

    def ensure_collection(self, collection_name: str, recreate: bool = False):
        vector_size = len(self.embeddings.embed_query("test"))
        self._make_collection(
            name=collection_name,
            vector_size=vector_size,
            recreate=recreate,
        )

        self.vectorstore = self._get_vectorstore(collection_name)
        self.search_engine = QdrantSearchEngine(self.vectorstore)

    def ingest_chunks(
        self,
        chunks: List[str],
        collection_name: str,
        recreate: bool = False,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        try:
            if not chunks:
                raise ValueError("No chunks provided")
            if not isinstance(chunks, list):
                raise TypeError("chunks must be a list of strings")

            self.ensure_collection(collection_name, recreate=recreate)

            docs: List[Document] = []
            for i, chunk in enumerate(chunks):
                if not chunk or not chunk.strip():
                    continue

                metadata = {"chunk_id": i}
                if metadatas and i < len(metadatas) and metadatas[i]:
                    metadata.update(metadatas[i])

                docs.append(Document(page_content=chunk.strip(), metadata=metadata))

            if not docs:
                raise ValueError("No valid non-empty chunks found")

            assert self.vectorstore is not None
            self.vectorstore.add_documents(docs)

            logger.info(f"Ingested {len(docs)} chunks into '{collection_name}'")

        except Exception as e:
            logger.exception(f"Ingest failed: {e}")
            raise

    def search(
        self,
        query: str,
        collection_name: str,
        method: str = "similarity",
        topk: int = 5,
    ):
        try:
            if not query or not query.strip():
                raise ValueError("Empty query")
            if topk <= 0:
                raise ValueError("topk must be > 0")

            self.ensure_collection(collection_name, recreate=False)

            assert self.search_engine is not None
            return self.search_engine.search_rag(query=query, topk=topk, method=method)

        except Exception as e:
            logger.exception(f"Qdrant search failed: {e}")
            raise

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        try:
            info = self.client.get_collection(collection_name=collection_name)
            if hasattr(info, "model_dump"):
                raw = info.model_dump()
            else:
                raw = info.dict()
            return raw
        except Exception as e:
            logger.exception(f"Failed to fetch stats for '{collection_name}': {e}")
            raise
# from langchain_qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams
# from langchain_core.documents import Document
# from  Qdrant.Qsearch import QdrantSearchEngine
# import logging

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO) 

# class QdrantBackend:
#     def __init__(self, embeddings, mode="memory", url="http://localhost:6333"):
#         self.embeddings = embeddings
#         self.chunks = []
#         self.vectorstore = None
#         self.search_engine = None

#         try:
#             self.client = QdrantClient(":memory:") if mode == "memory" else QdrantClient(url=url)
#             logger.info(f"Qdrant client initialized in '{mode}' mode")
#         except Exception as e:
#             logger.exception(f"Qdrant client init failed: {e}")
#             raise

#     def _make_collection(self, name, vector_size):
#         if self.client.collection_exists(name):
#             self.client.delete_collection(name)
#         self.client.create_collection(
#             collection_name=name,
#             vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
#         )

#     def build_index(self, chunks):
#         try:
#             if not chunks:
#                 raise ValueError("No chunks provided")
#             if not isinstance(chunks, list):
#                 raise TypeError("Chunks must be a list")

#             self.chunks = chunks
#             vector_size = len(self.embeddings.embed_query("test"))
#             self._make_collection("rag", vector_size)

#             # add texts manually — avoids the internal client conflict
#             vectors = self.embeddings.embed_documents(chunks)
#             self.client.upload_collection(
#                 collection_name="rag",
#                 vectors=vectors,
#                 payload=[{"page_content": c} for c in chunks]
#             )

#             # now construct vectorstore pointing at the existing collection
#             self.vectorstore = QdrantVectorStore(
#                 client=self.client,
#                 collection_name="rag",
#                 embedding=self.embeddings,
#             )
#             logger.info(f"Qdrant: indexed {len(chunks)} chunks")
#             self.search_engine = QdrantSearchEngine(self.vectorstore)

#         except (ValueError, TypeError):
#             logger.exception("Invalid input")
#             raise
#         except Exception as e:
#             logger.exception(f"Qdrant indexing failed: {e}")
#             raise

#     def search(self, query, method, topk):
#         try:
#             if self.vectorstore is None:
#                 raise ValueError("Call ingest() first")
#             if not query:
#                 raise ValueError("Empty query")
#             if topk <= 0:
#                 raise ValueError("topk must be > 0")

#             if self.search_engine is None:
#                 raise ValueError("Search engine not initialized")
#             return self.search_engine.search_rag(query, topk, method)

#         except (ValueError, ImportError):
#             logger.exception("Search input error")
#             raise
#         except Exception as e:
#             logger.exception(f"Qdrant search failed: {e}")
#             raise
