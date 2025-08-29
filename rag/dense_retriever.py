import json
import numpy as np
from sentence_transformers import SentenceTransformer, util


class ThirukkuralDenseRetriever:
    def __init__(self, corpus_path="thirukkural_corpus.json", model_name="all-MiniLM-L6-v2"):
        
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        # Store docs per language
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

        # Load dense model
        self.model = SentenceTransformer(model_name)

        # Precompute embeddings
        self.emb_en = self.model.encode([d["text"] for d in self.docs_en], convert_to_tensor=True, normalize_embeddings=True)
        self.emb_ta = self.model.encode([d["text"] for d in self.docs_ta], convert_to_tensor=True, normalize_embeddings=True)
        self.emb_hi = self.model.encode([d["text"] for d in self.docs_hi], convert_to_tensor=True, normalize_embeddings=True)

    def retrieve(self, query, lang="en", topk=12):
        # Encode query
        query_emb = self.model.encode(query, convert_to_tensor=True, normalize_embeddings=True)

        if lang == "en":
            scores = util.cos_sim(query_emb, self.emb_en)[0].cpu().numpy()
            docs = self.docs_en
        elif lang == "ta":
            scores = util.cos_sim(query_emb, self.emb_ta)[0].cpu().numpy()
            docs = self.docs_ta
        elif lang == "hi":
            scores = util.cos_sim(query_emb, self.emb_hi)[0].cpu().numpy()
            docs = self.docs_hi
        else:
            raise ValueError("lang must be 'en', 'ta', or 'hi'")

        # Rank
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
    retriever = ThirukkuralDenseRetriever("thirukkural_corpus.json")

    print("🔎 English query:")
    results = retriever.retrieve("what does thirukkural say about education", lang="en", topk=5)
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
