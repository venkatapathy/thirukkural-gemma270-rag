import pymysql
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

# ======================
# CONFIG
# ======================
GROQ_API_KEY = "***"   # 🔑 Replace with your Groq key
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Embedding model (for FAISS)
GROQ_MODEL = "llama-3.3-70b-versatile" # LLM model (for generation)
TABLE_NAME = "kurals"
TOP_K = 3

# ======================
# STEP 1: Connect to MySQL
# ======================
conn = pymysql.connect(
    host="localhost",
    user="root",
    password="***",  # 🔑 Replace with your MySQL password
    database="kural"
)
cursor = conn.cursor()

cursor.execute("SELECT ID, Kural, Couplet, Vilakam FROM kurals")
rows = cursor.fetchall()

# ======================
# STEP 2: Create embeddings & FAISS index
# ======================
model = SentenceTransformer(EMBEDDING_MODEL)

texts = [f"{row[1]} {row[2]} {row[3]}" for row in rows]
ids = [row[0] for row in rows]

embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)



# ======================
# STEP 3: Retriever function
# ======================
def retrieve(query, top_k=TOP_K):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, top_k)

    results = []
    for idx, score in zip(I[0], D[0]):
        row_id = ids[idx]
        cursor.execute("SELECT ID, Kural, Couplet, Vilakam FROM kurals WHERE ID=%s", (row_id,))
        record = cursor.fetchone()
        if record:
            results.append({
                "ID": record[0],
                "Kural": record[1],
                "Couplet": record[2],
                "Vilakam": record[3],
                "score": float(score)
            })
    return results

# ======================
# STEP 4: Generator using Groq API
# ======================
def generate_response(query, retrieved_docs):
    context = "\n\n".join(
        [f"ID: {doc['ID']}\nKural: {doc['Kural']}\nCouplet: {doc['Couplet']}\nVilakam: {doc['Vilakam']}" 
         for doc in retrieved_docs]
    )

    prompt = f"""
You are a helpful assistant that answers queries about Thirukkural.
Use the following retrieved passages as context:

{context}

User Query: {query}

Answer in a clear and accurate way.
Give explaination in a single line for all the three kurals seperated by "*".
Do not include any other sentences other than the particular answer
    """

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()

    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    else:
        return f"❌ API Error: {data}"

# ======================
# STEP 5: Example usage
# ======================







def out(query):
    retrieved = retrieve(query, top_k=3)

    # Generate one combined answer (contains 3 explanations separated by "*")
    answer_text = generate_response(query, retrieved)
    answer_lines = [a.strip() for a in answer_text.split("*") if a.strip()]

    final_output = []

    for i, r in enumerate(retrieved):
        # Fetch full details for each Kural (without Porul)
        cursor.execute(
            "SELECT ID, Kural, Couplet, Vilakam, Adhigaram, Transliteration FROM kurals WHERE ID=%s",
            (r['ID'],)
        )
        extra = cursor.fetchone()

        if extra:
            details = {
                "ID": extra[0],
                "Kural": extra[1],
                "Couplet": extra[2],
                "Vilakam": extra[3],
                "Adhigaram": extra[4],
                "Transliteration": extra[5]
            }

            kural_answer = answer_lines[i] if i < len(answer_lines) else ""
            final_output.append([details, kural_answer])

    return final_output
