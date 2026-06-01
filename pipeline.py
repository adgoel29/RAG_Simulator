from faissclass import RagClass
# from chunking import chunking
from sentence_transformers import SentenceTransformer
from search import SearchEngine
path="sample.txt"
model = SentenceTransformer("all-MiniLM-L6-v2")
test=RagClass("all-MiniLM-L6-v2")
#semantic or fixed,parameters of fixed are changeable
chunk=test.chunking(path,"semantic")
print(chunk)
test.ingest(chunk)
query = input("Enter query: ")
#mmr or similarity ,default is similarity,can  use top_k as well.
ans = test.searching(query,"mmr")
print(ans)

