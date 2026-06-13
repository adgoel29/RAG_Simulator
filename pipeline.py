from rag import RagClass   # change name according to your file


# -------------------------
# CONFIG
# -------------------------

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BACKEND = "qdrant"

FILE_PATH = "/home/aditya/Desktop/RAG_Simulator/chemistry project.docx"

CHUNK_METHOD = "semantic"


# -------------------------
# INITIALIZE RAG
# -------------------------

rag = RagClass(
    model_name=MODEL_NAME
)


# -------------------------
# INGESTION PIPELINE
# -------------------------

print("Creating chunks...")

chunks = rag.chunking(
    path=FILE_PATH,
    method=CHUNK_METHOD
)


print(f"Chunks created: {len(chunks)}")


print("Adding chunks to Qdrant...")

rag.ingest(
    chunks,
    backend_name=BACKEND
)


print("Ingestion complete")


# -------------------------
# QUERY PIPELINE
# -------------------------

while True:

    query = input("\nEnter query: ")

    if query.lower() in ["exit", "quit"]:
        break


    # Retrieve from Qdrant

    retrieved_docs = rag.searching(
        query=query,
        method="similarity",
        topk=5,
        backend_name=BACKEND
    )


    if not retrieved_docs:
        print("No documents found")
        continue


    print("\nRetrieved docs:")
    print("-" * 50)

    for doc in retrieved_docs:
        print(doc[:300])
        print()



    # Reranking

    reranked_docs = rag.rerank(
        query=query,
        docs=retrieved_docs,
        topk=3
    )


    print("\nAfter reranking:")
    print("-" * 50)

    for i, doc in enumerate(reranked_docs, 1):
        print(f"\nResult {i}")
        print(doc)