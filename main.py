from fastapi import FastAPI, UploadFile, File
import uvicorn
import pickle
import json
import numpy as np
import os

from pdfminer.high_level import extract_text
from docx import Document
import openpyxl

from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

vectorizer = pickle.load(open("tfidf_model.pkl", "rb"))

with open("vector_data.json", "r") as f:
    data = json.load(f)

document_ids = data["ids"]
stored_vectors = np.array(data["vectors"])

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_text(file_path)
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    elif ext in [".xlsx", ".xls"]:
        wb = openpyxl.load_workbook(file_path)
        text = ""
        for sheet in wb:
            for row in sheet.iter_rows(values_only=True):
                text += " ".join([str(cell) if cell else "" for cell in row]) + "\n"
        return text
    else:
        raise ValueError("Unsupported file type")

def compare_vectors(uploaded_vector):
    similarities = cosine_similarity([uploaded_vector], stored_vectors)
    best_score = float(np.max(similarities))
    best_index = int(np.argmax(similarities))
    return {
        "matched_document_id": document_ids[best_index],
        "similarity_score": best_score,
        "is_duplicate": best_score > 0.8
    }

@app.post("/check-duplicate")
async def check_duplicate(file: UploadFile = File(...)):
    temp_filename = f"temp_uploaded_file{os.path.splitext(file.filename)[1]}"
    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    try:
        extracted_text = extract_text_from_file(temp_filename)
        vector = vectorizer.transform([extracted_text]).toarray()[0]
        result = compare_vectors(vector)
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    return result
