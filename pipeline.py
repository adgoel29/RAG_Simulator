from faissclass import RagClass
from chunking import chunking
from sentence_transformers import SentenceTransformer
from search import SearchEngine
path="sample_text.txt"
model = SentenceTransformer("all-MiniLM-L6-v2")
test=RagClass("all-MiniLM-L6-v2")
chunk=chunking(path)
test.ingest(chunk)
search = SearchEngine(test.vectorstore)
query = input("Enter query: ")
vectorized_value = test.embeddings.embed_query(query)
method = input(
    "Choose search method (similarity(1) / mmr(2)): "
).strip().lower()

docs = search.search(
    query=query,
    method=method
)
ans = search.search(query=query, method=method)
print(vectorized_value[:5])
for doc in ans:
    vector = test.embeddings.embed_query(doc)
    print(doc)

    # print("\nFirst 10 Values:")
    # print(vector)

