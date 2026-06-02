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
"""
# =========================
# MS MARCO CROSS-ENCODERS
# =========================

# Fast, lightweight, great baseline
# ~22M params
# Good CPU performance
"cross-encoder/ms-marco-MiniLM-L-6-v2"

# Smaller and faster than L6
# ~12M params
"cross-encoder/ms-marco-MiniLM-L-2-v2"

# Better accuracy than L6
# ~33M params
"cross-encoder/ms-marco-MiniLM-L-12-v2"

# Extremely fast
# Tiny model
"cross-encoder/ms-marco-TinyBERT-L-2-v2"

# Strong accuracy
# Electra architecture
"cross-encoder/ms-marco-electra-base"

# Highest accuracy among MS MARCO variants
# Slower and larger
"cross-encoder/ms-marco-electra-large"


# =========================
# BGE RERANKERS
# =========================

# Strong open-source reranker
# Recommended for most RAG systems
"BAAI/bge-reranker-base"

# Higher accuracy version
# Requires more memory and compute
"BAAI/bge-reranker-large"


# =========================
# MULTILINGUAL RERANKERS
# =========================

# Supports many languages
"cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Multilingual passage reranking
"amberoad/bert-multilingual-passage-reranking-msmarco"


# =========================
# QA / NLI CROSS-ENCODERS
# =========================

# General sentence relevance scoring
"cross-encoder/stsb-roberta-base"

# Strong semantic similarity model
"cross-encoder/stsb-distilroberta-base"

# Natural Language Inference
# Useful for contradiction / entailment scoring
"cross-encoder/nli-distilroberta-base"

# Larger NLI model
"cross-encoder/nli-roberta-base"
"""