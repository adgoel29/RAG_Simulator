from sentence_transformers import CrossEncoder
class Reranker:
    def __init__(self,model_name):
        self.reranker_instance=CrossEncoder(model_name)
    def reranking_docs(self,docs,query,topk=3):
        pairs=[(query,doc)for doc in docs]
        scores=self.reranker_instance.predict(pairs)
        reranked_docs=sorted(
            zip(docs,scores),
            key=lambda x: x[1],
            reverse=True
        )
        return [doc for doc, _ in reranked_docs[:topk]]
