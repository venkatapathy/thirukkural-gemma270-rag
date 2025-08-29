import json
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import re


def simple_tokens(text):
    return re.findall(r"\b\w+\b", text.lower())


class ThirukkuralHybridRetriever:
    def __init__(self, corpus_path="thirukkural_corpus.json", model_name="all-MiniLM-L6-v2", alpha=0.5):
        """
        alpha: weight for dense retriever. (0.5 means equal weight between BM25 and dense)
        """
        self.alpha = alpha

        # Loading the corpus
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        # Storing the docs per language
        self.docs_en, self.docs_ta, self.docs_hi = [], [], []

        for entry in corpus:
            kural_id = entry["kural_id"]

            # English
            en_text = " ".join([
                entry["english"].get("line1", ""),
                entry["english"].get("line2", ""),
                entry["english"].get("translation", ""),
                entry["english"].get("paal", ""),
                entry["english"].get("iyal", ""),
                entry["english"].get("adhigaram", "")
            ])
            self.docs_en.append({"id": kural_id, "lang": "en", "text": en_text})

            # Tamil
            ta_text = " ".join([
                entry["tamil"].get("line1", ""),
                entry["tamil"].get("line2", ""),
                entry["tamil"].get("paal", ""),
                entry["tamil"].get("iyal", ""),
                entry["tamil"].get("adhigaram", "")
            ])
            self.docs_ta.append({"id": kural_id, "lang": "ta", "text": ta_text})

            # Hindi
            hi_text = " ".join([
                entry["tamil"].get("line1", ""),
                entry["tamil"].get("line2", ""),
                entry["hindi"].get("explanation", ""),
                entry["hindi"].get("paal", ""),
                entry["hindi"].get("iyal", ""),
                entry["hindi"].get("adhigaram", "")
            ])
            self.docs_hi.append({"id": kural_id, "lang": "hi", "text": hi_text})

        # Build the BM25 indexes
        self.bm25_en = BM25Okapi([simple_tokens(doc["text"]) for doc in self.docs_en])
        self.bm25_ta = BM25Okapi([simple_tokens(doc["text"]) for doc in self.docs_ta])
        self.bm25_hi = BM25Okapi([simple_tokens(doc["text"]) for doc in self.docs_hi])

        # Loading the  dense model
        self.model = SentenceTransformer(model_name)

        # Precomputation of dense embeddings
        self.emb_en = self.model.encode([d["text"] for d in self.docs_en], convert_to_tensor=True, normalize_embeddings=True)
        self.emb_ta = self.model.encode([d["text"] for d in self.docs_ta], convert_to_tensor=True, normalize_embeddings=True)
        self.emb_hi = self.model.encode([d["text"] for d in self.docs_hi], convert_to_tensor=True, normalize_embeddings=True)

    def retrieve(self, query, lang="en", topk=12):
        # Tokenizing the  query for BM25
        qtok = simple_tokens(query)

        # Encode query for dense retriever
        query_emb = self.model.encode(query, convert_to_tensor=True, normalize_embeddings=True)

        if lang == "en":
            bm25_scores = np.array(self.bm25_en.get_scores(qtok))
            dense_scores = util.cos_sim(query_emb, self.emb_en)[0].cpu().numpy()
            docs = self.docs_en
        elif lang == "ta":
            bm25_scores = np.array(self.bm25_ta.get_scores(qtok))
            dense_scores = util.cos_sim(query_emb, self.emb_ta)[0].cpu().numpy()
            docs = self.docs_ta
        elif lang == "hi":
            bm25_scores = np.array(self.bm25_hi.get_scores(qtok))
            dense_scores = util.cos_sim(query_emb, self.emb_hi)[0].cpu().numpy()
            docs = self.docs_hi
        else:
            raise ValueError("lang must be 'en', 'ta', or 'hi'")

        # Normalize scores
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        if dense_scores.max() > 0:
            dense_scores = dense_scores / dense_scores.max()

        # Hybrid score = weighted sum
        hybrid_scores = (1 - self.alpha) * bm25_scores + self.alpha * dense_scores

        # Rank results
        order = np.argsort(-hybrid_scores)[:topk]

        results = []
        for i in order:
            results.append({
                "kural_id": docs[i]["id"],
                "lang": docs[i]["lang"],
                "score": float(hybrid_scores[i]),
                "text": docs[i]["text"]
            })
        return results


# Example usage
if __name__ == "__main__":
    retriever = ThirukkuralHybridRetriever("thirukkural_corpus.json", alpha=0.6)

    print("🔎 English query:")
    results = retriever.retrieve("How can I apply Thirukkural's teaching on providence / rain & sustenance today?	", lang="en", topk=5)
    for r in results:
        print(f"[{r['lang']}] Kural ID: {r['kural_id']} | Score: {r['score']:.3f}\n{r['text']}\n---")

    print("\n🔎 Tamil query:")
    results = retriever.retrieve("மருந்து", lang="ta", topk=5)
    for r in results:
        print(f"[{r['lang']}] Kural ID: {r['kural_id']} | Score: {r['score']:.3f}\n{r['text']}\n---")

    print("\n🔎 Hindi query:")
    results = retriever.retrieve("सच", lang="hi", topk=5)
    for r in results:
        print(f"[{r['lang']}] Kural ID: {r['kural_id']} | Score: {r['score']:.3f}\n{r['text']}\n---")
