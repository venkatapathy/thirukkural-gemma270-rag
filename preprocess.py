import pandas as pd
import json

file_path = "thirukkural_multi_language.xlsx"  

tamil_df = pd.read_excel(file_path, sheet_name="Tamil")
english_df = pd.read_excel(file_path, sheet_name="English")
hindi_df = pd.read_excel(file_path, sheet_name="Hindi")

tamil_df = tamil_df.rename(columns={"Kural no": "kural_id"})
english_df = english_df.rename(columns={"kural no": "kural_id"})
hindi_df = hindi_df.rename(columns={"g": "kural_id"})  

merged = pd.merge(tamil_df, english_df, on="kural_id", suffixes=("_ta", "_en"))
merged = pd.merge(merged, hindi_df, on="kural_id", suffixes=("", "_hi"))

corpus = []
for _, row in merged.iterrows():
    entry = {
        "kural_id": int(row["kural_id"]),
        "tamil": {
            "line1": str(row.get("Line 1_ta", "")),
            "line2": str(row.get("Line 2_ta", "")),
            "paal": str(row.get("Paal_ta", "")),
            "iyal": str(row.get("Iyal_ta", "")),
            "adhigaram": str(row.get("Adhigaram_ta", ""))
        },
        "english": {
            "line1": str(row.get("Line 1_en", "")),
            "line2": str(row.get("Line 2_en", "")),
            "translation": str(row.get("Translation", "")),
            "paal": str(row.get("Paal_en", "")),
            "iyal": str(row.get("Iyal_en", "")),
            "adhigaram": str(row.get("Adhigaram_en", ""))
        },
        "hindi": {
            "explanation": str(row.get("Explanation", "")),
            "paal": str(row.get("Hindi Paal", "")),
            "iyal": str(row.get("Hindi Iyal", "")),
            "adhigaram": str(row.get("Himdi  Adhigaram", ""))
        }
    }
    corpus.append(entry)


output_path = "thirukkural_corpus.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(corpus, f, ensure_ascii=False, indent=2)

print(f"Corpus saved as {output_path} with {len(corpus)} entries.")
