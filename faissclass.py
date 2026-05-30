from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
model=SentenceTransformer("all-MiniLM-L6-v2")

class Ragclass:
    def __init__(self):
        self.index=None
        self.chunks=None
    def getembedding(self,chunks):
        embed=model.encode(chunks)
        embed=np.array(embed)
        dim=embed.shape[1]
        return embed,dim
    
    def ingest(self,chunks):
        # embed=model.encode(chunks)
        self.chunks=chunks
        embed,dim=self.getembedding(chunks)
        self.index=faiss.IndexFlatL2(dim)
        self.index.add(embed)

    def search(self,query,topk):
        # tossearch=[query]
        embed,_=self.getembedding([query])
        dis,ind=self.index.search(embed,topk)
        ans=""
        for i in ind[0]:
            ans+=self.chunks[i]
        return ans
        

