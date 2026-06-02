class SearchEngine:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def search_rag(self, query, topk=5, method="similarity"):

        if method == "similarity":
            docs = self.vectorstore.similarity_search_with_score(query, k=topk)

        elif method == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(query,k=topk)
        
        else:
            docs = self.vectorstore.similarity_search(query, k=topk)

        return docs