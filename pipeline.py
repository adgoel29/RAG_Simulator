from faissclass import RagClass
from chunking import chunking
from sentence_transformers import SentenceTransformer
from search import SearchEngine
path="sample_text.txt"
model = SentenceTransformer("all-MiniLM-L6-v2")
test=RagClass("all-MiniLM-L6-v2")
chunk=chunking(path)
test.ingest(chunk)
query = input("Enter query: ")
ans = test.search(query)
print(ans)

