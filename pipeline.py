from faissclass import RagClass
from chunking import chunking
from sentence_transformers import SentenceTransformer
path=r"C:\Users\aditya\Desktop\pracsesh\sample.txt"
model = SentenceTransformer("all-MiniLM-L6-v2")
test=RagClass("all-MiniLM-L6-v2")
query="faiss"
chunk=chunking(path)
test.ingest(chunk)
ans=test.search(query,3)
print(ans)

