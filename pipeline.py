from faissclass import RagClass
# from chunking import chunking
from sentence_transformers import SentenceTransformer
from search import SearchEngine
path="sample.txt"
model = SentenceTransformer("all-MiniLM-L6-v2")
test=RagClass("all-MiniLM-L6-v2")
chunk=test.chunking(path,"semantic")
print(chunk)
test.ingest(chunk)
query = input("Enter query: ")
ans = test.search(query)
print(ans)

