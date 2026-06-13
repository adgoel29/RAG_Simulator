import os
import shutil
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import RagClass
from reranker import Reranker
from frontend_from_backend import F_to_B
from loadfile import get_content

app = FastAPI(
    title="RAG Simulator API",
    description="Visual RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_MODEL = "all-MiniLM-L6-v2"
rag_instance = RagClass(DEFAULT_MODEL)

chunks_cache = []
embeddings_cache = []

class ChunkRequest(BaseModel):
    text: str
    method: str = "fixed" 
    chunk_size: int = 50
    chunk_overlap: int = 5
    separate_keywords: str = "recursive"
    sep_value: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    method: str = "similarity"
    topk: int = 5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_topk: int = 3
    backend: str = "faiss"

@app.post("/api/chunk")
def api_chunk(req: ChunkRequest):
    global chunks_cache, embeddings_cache
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    temp_path = "temp_input.txt"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(req.text)
        
        kwargs = {}
        if req.method == "fixed":
            kwargs = {
                "separate_keywords": req.separate_keywords,
                "sep_value": req.sep_value,
                "chunk_size": req.chunk_size,
                "chunk_overlap": req.chunk_overlap
            }
            
        chunks = rag_instance.chunking(temp_path, req.method, **kwargs)
        
        chunks_cache = chunks
        embeddings_cache = []
        
        return {"chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    temp_dir = "temp_uploads"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    file_path = os.path.join(temp_dir, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        content = get_content(file_path)
        return {"filename": file.filename, "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

@app.post("/api/ingest", response_model=List[F_to_B])
def api_ingest(backend: str = "faiss"):
    global chunks_cache, embeddings_cache
    if not chunks_cache:
        raise HTTPException(status_code=400, detail="No chunks available to ingest. Please run chunking first.")
    
    try:
        embeddings = rag_instance.embeddings.embed_documents(chunks_cache)
        embeddings_cache = embeddings
        
        rag_instance.ingest(chunks_cache, backend_name=backend)
        
        response_data = []
        for idx, (chunk_text, emb) in enumerate(zip(chunks_cache, embeddings_cache)):
            response_data.append(F_to_B(
                chunk_id=idx,
                chunk=chunk_text,
                embedding_method=DEFAULT_MODEL,
                vector_store=backend.upper(),
                embedding=emb,
                before_reranker=None,
                after_reranker=None
            ))
            
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=List[F_to_B])
def api_query(req: QueryRequest):
    global chunks_cache, embeddings_cache
    
    try:
        backend = rag_instance.get_backend(req.backend)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    if not chunks_cache or backend.vectorstore is None:
        raise HTTPException(status_code=400, detail=f"Vector store '{req.backend}' not initialized. Please ingest chunks first.")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    try:
        docs = backend.search(req.query, req.method, req.topk)
        docs_with_scores = [(doc, float(req.topk - i)) for i, doc in enumerate(docs)]
            
        retrieved_map = {}
        for rank, (doc, score) in enumerate(docs_with_scores):
            text = doc.page_content
            retrieved_map[text] = {
                "rank": rank + 1,
                "score": float(score)
            }
            
        retrieved_texts = list(retrieved_map.keys())
        reranked_map = {}
        if retrieved_texts:
            reranker_instance = Reranker(req.reranker_model)
            pairs = [(req.query, text) for text in retrieved_texts]
            rerank_scores = reranker_instance.reranker.predict(pairs)
            
            reranked_docs = sorted(
                zip(retrieved_texts, rerank_scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            for rank, (text, score) in enumerate(reranked_docs[:req.rerank_topk]):
                reranked_map[text] = {
                    "rank": rank + 1,
                    "score": float(score)
                }

        response_data = []
        for idx, chunk_text in enumerate(chunks_cache):
            emb = embeddings_cache[idx] if idx < len(embeddings_cache) else []
            
            before = None
            if chunk_text in retrieved_map:
                before = {
                    "rank": retrieved_map[chunk_text]["rank"],
                    "score": retrieved_map[chunk_text]["score"]
                }
                
            after = None
            if chunk_text in reranked_map:
                after = {
                    "rank": reranked_map[chunk_text]["rank"],
                    "score": reranked_map[chunk_text]["score"]
                }
                
            response_data.append(F_to_B(
                chunk_id=idx,
                chunk=chunk_text,
                embedding_method=DEFAULT_MODEL,
                vector_store=req.backend.upper(),
                embedding=emb,
                before_reranker=before,
                after_reranker=after
            ))
            
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
