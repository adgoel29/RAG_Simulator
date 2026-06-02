from faissclass import RagClass
# from chunking import chunking
from sentence_transformers import SentenceTransformer
from search import SearchEngine
from reranker import Reranker
path="sample.txt"
model = SentenceTransformer("all-MiniLM-L6-v2")
test=RagClass("all-MiniLM-L6-v2")
#semantic or fixed,parameters of fixed are changeable
chunk=test.chunking(path,"semantic")
print(chunk)
test.ingest(chunk)
query = input("Enter query: ")
#mmr or similarity ,default is similarity,can  use top_k as well.
docs= test.searching(query,"mmr")
print(docs)
model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker_instance=Reranker(model_name)
reranked_docs=reranker_instance.reranking_docs(docs,query,topk=3)
print([reranked_doc.page_content for reranked_doc in reranked_docs ])