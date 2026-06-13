from pydantic import BaseModel,Field 
from typing import List, Dict, Any, Optional
class ChunkMetadata(BaseModel):
    chunk_id: int
    text: str
    start_char: int
    end_char: int
class EmbeddingMetadata(BaseModel):
    method: str
    vector_dimension: int
    embedding: List[float] = Field(default_factory=list)
class RetrievalMetadata(BaseModel):
    similarity_score: float
    rank: int
class RerankerMetadata(BaseModel):
    score_before: float
    score_after: float
    rank_before: int
    rank_after: int
class ChunkTrace(BaseModel):
    chunk: ChunkMetadata
    embedding: Optional[EmbeddingMetadata] = None
    retrieval: Optional[RetrievalMetadata] = None
    reranker: Optional[RerankerMetadata] = None
    extra: Dict[str, Any] = Field(default_factory=dict)
class F_to_B(BaseModel):
    chunk_id:int
    chunk:str
    embedding_method:str
    vector_store:str
    embedding:List[float] = Field(default_factory=list)
    before_reranker:Optional[Dict[str,Any]]=None
    after_reranker:Optional[Dict[str,Any]]=None