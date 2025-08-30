from sentence_transformers import CrossEncoder
import numpy as np

class ThirukkuralReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize CrossEncoder reranker.
        You can switch to a multilingual one if needed:
          - 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
          - 'cross-encoder/stsb-xlm-r-multilingual'
        """
        self.reranker = CrossEncoder(model_name)

    def rerank(self, query, candidates, topk=5):
        """
        Rerank retrieved results.

        Args:
            query (str): User query (English/Tamil/Hindi).
            candidates (list): List of dicts from retrievers 
                               [{"text":..., "kural_id":..., "lang":..., "score":...}, ...]
            topk (int): Number of reranked results to return.

        Returns:
            list: Reranked results as dicts with rerank_score.
        """
        if not candidates:
            return []

        #Prepare pairs for reranker
        pairs = [(query, c["text"]) for c in candidates]

        #Predict scores
        scores = self.reranker.predict(pairs)

        # Attach reranker scores
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])

        #rerank_score
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:topk]


def get_final_kurals(reranked_results, normalize_scores=True):
    """
    Convert reranked results into display-ready Kurals.

    Args:
        reranked_results (list): Output of rerank() (list of dicts).
        normalize_scores (bool): Whether to normalize scores to [0,1].

    Returns:
        list: List of dicts [{ "kural_id":..., "lang":..., "text":..., "score":... }]
    """
    if not reranked_results:
        return []

    if normalize_scores:
        scores = np.array([r["rerank_score"] for r in reranked_results])
        # Normalize to [0,1]
        min_s, max_s = scores.min(), scores.max()
        if max_s > min_s:
            norm_scores = (scores - min_s) / (max_s - min_s)
        else:
            norm_scores = np.ones_like(scores)  
    else:
        norm_scores = [r["rerank_score"] for r in reranked_results]

    final = []
    for r, s in zip(reranked_results, norm_scores):
        final.append({
            "kural_id": r["kural_id"],
            "lang": r["lang"],
            "text": r["text"],
            "score": float(s)
        })
    return final

#Example usage:
if __name__ == "__main__":
    # Initialize retrievers
    sparse = ThirukkuralBM25Retriever("thirukkural_corpus.json")
    dense = ThirukkuralDenseRetriever("thirukkural_corpus.json")
    hybrid = ThirukkuralHybridRetriever("thirukkural_corpus.json", alpha=0.6)

    # Initialize reranker
    reranker = ThirukkuralReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

    query = "What does Thirukkural say about guidance on learning?"

    print("\n🔎 Sparse + Final:")
    candidates = sparse.retrieve(query, lang="en", topk=10)
    reranked = reranker.rerank(query, candidates, topk=3)
    final = get_final_kurals(reranked)
    for r in final:
        print(f"[{r['lang']}] Kural ID: {r['kural_id']} | Score: {r['score']:.3f}\n{r['text']}\n---")

    print("\n🔎 Dense + Final:")
    candidates = dense.retrieve(query, lang="en", topk=10)
    reranked = reranker.rerank(query, candidates, topk=3)
    final = get_final_kurals(reranked)
    for r in final:
        print(f"[{r['lang']}] Kural ID: {r['kural_id']} | Score: {r['score']:.3f}\n{r['text']}\n---")

    print("\n🔎 Hybrid + Final:")
    candidates = hybrid.retrieve(query, lang="en", topk=10)
    reranked = reranker.rerank(query, candidates, topk=3)
    final = get_final_kurals(reranked)
    for r in final:
        print(f"[{r['lang']}] Kural ID: {r['kural_id']} | Score: {r['score']:.3f}\n{r['text']}\n---")


