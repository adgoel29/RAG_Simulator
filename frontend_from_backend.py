from pydantic  import BaseModel,Field 
from typing import List,Dict,Any,Optional
class F_to_B(BaseModel):
    chunk_id:int
    chunk:str
    embedding_method:str
    vector_store:str
    embedding:List[float] = Field(default_factory=list)
    before_reranker:Optional[Dict[str,Any]]=None
    after_reranker:Optional[Dict[str,Any]]=None
    


   