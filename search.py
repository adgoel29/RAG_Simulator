
class SearchEngine:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def search(self, query, topk=5, method="similarity"):

        if method == "1":
            docs = self.vectorstore.similarity_search(query, k=topk)

        elif method == "2":
            docs = self.vectorstore.max_marginal_relevance_search(query,k=topk)
        
        else:
            raise ValueError(f"Unknown search method: {method}")

        return [doc.page_content for doc in docs]