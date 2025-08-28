import json
import numpy as np
from rank_bm25 import BM25Okapi
import re

def simple_tokens(text):
    return re.findall(r"\b\w+\b", text.lower())


class ThirukkuralBM25Retriever:
    def __init__(self, corpus_path="thirukkural_corpus.json"):
        # Load corpus
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        # Build docs per language
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
                entry["hindi"].get("explanation", ""),
                entry["hindi"].get("paal", ""),
                entry["hindi"].get("iyal", ""),
                entry["hindi"].get("adhigaram", "")
            ])
            self.docs_hi.append({"id": kural_id, "lang": "hi", "text": hi_text})

        # Build BM25 indexes
        self.bm25_en = BM25Okapi([simple_tokens(doc["text"]) for doc in self.docs_en])
        self.bm25_ta = BM25Okapi([simple_tokens(doc["text"]) for doc in self.docs_ta])
        self.bm25_hi = BM25Okapi([simple_tokens(doc["text"]) for doc in self.docs_hi])

    def retrieve(self, query, lang="en", topk=12):
        qtok = simple_tokens(query)

        if lang == "en":
            scores = self.bm25_en.get_scores(qtok)
            docs = self.docs_en
        elif lang == "ta":
            scores = self.bm25_ta.get_scores(qtok)
            docs = self.docs_ta
        elif lang == "hi":
            scores = self.bm25_hi.get_scores(qtok)
            docs = self.docs_hi
        else:
            raise ValueError("lang must be 'en', 'ta', or 'hi'")

        order = np.argsort(-scores)[:topk]

        results = []
        for i in order:
            results.append({
                "kural_id": docs[i]["id"],
                "lang": docs[i]["lang"],
                "score": float(scores[i]),
                "text": docs[i]["text"]
            })
        return results

# Example usage

if __name__ == "__main__":
    retriever = ThirukkuralBM25Retriever("thirukkural_corpus.json")

    print("🔎 English query:")
    results = retriever.retrieve("what does thirukkural say about friendship", lang="en", topk=5)
    for r in results:
        print(f"[{r['lang']}] Kural ID: {r['kural_id']} | Score: {r['score']:.3f}\n{r['text']}\n---")

    print("\n🔎 Tamil query:")
    results = retriever.retrieve("மருந்து", lang="ta", topk=5)
    for r in results:
        print(f"[{r['lang']}] Kural ID: {r['kural_id']} | Score: {r['score']:.3f}\n{r['text']}\n---")

    print("\n🔎 Hindi query:")
    results = retriever.retrieve("प्रेम", lang="hi", topk=5)
    for r in results:
        print(f"[{r['lang']}] Kural ID: {r['kural_id']} | Score: {r['score']:.3f}\n{r['text']}\n---")
