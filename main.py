from fastapi import FastAPI, UploadFile, File
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text
from docx import Document
import openpyxl

app = FastAPI()

# Load model and embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = np.load("embeddings.npy")
with open("ids.json", "r") as f:
    ids = json.load(f)

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text(file_path)
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext in [".xlsx", ".xls"]:
        wb = openpyxl.load_workbook(file_path)
        text = ""
        for sheet in wb:
            for row in sheet.iter_rows(values_only=True):
                text += " ".join([str(cell) if cell else "" for cell in row]) + "\n"
        return text
    else:
        raise ValueError("Unsupported file type")

@app.post("/check-similar-documents")
async def check_similar_documents(file: UploadFile = File(...), threshold: float = 0.75):
    ext = os.path.splitext(file.filename)[1]
    temp_file = f"temp{ext}"

    with open(temp_file, "wb") as f:
        f.write(await file.read())

    try:
        text = extract_text_from_file(temp_file)
        Documents_similaritys = compare_content(text); # type: ignore
        return Documents_similaritys;

    except Exception as e:
        return {"error": str(e)}

    finally:
        os.remove(temp_file)