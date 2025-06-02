from fastapi import FastAPI, UploadFile, File
#from train_model import compare_content  # Assuming this function is defined in train_model.py
#import numpy as np
import os
import json
from pdfminer.high_level import extract_text
from docx import Document
import openpyxl

from pyexpat import model
from sentence_transformers import SentenceTransformer
import numpy as np
import textwrap
import re
import pyodbc
# Calculate cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()


def create_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model
def train_model():
   
    conn_str = (
                         # Use 'dbmssocn' for TCP/IP
     )

    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    cursor.execute("Select DocumentID, Title, DataText from Documents where IsTemplate='false' and DataText Is Not Null")
    rows = cursor.fetchall()

    model = create_model()
    my_dict = {}
    
    for row in rows:
        documentId_Title= str(row[0]) + "_" + row[1]
        my_dict[documentId_Title]= get_document_embedding(row[2], model)

    conn.close()
    # Example Usage:
   
   
    # This function is a placeholder for any training logic you might want to implement.
    # For now, we are using a pre-trained model, so no training is needed.
    return my_dict

def split_document_into_chunks(document_text, max_tokens=512, overlap=50, tokenizer=None):
    if tokenizer is None:
        # Fallback if no specific tokenizer is provided, just use rough character count
        # This is less accurate than a real tokenizer but works as a simple heuristic
        words = document_text.split()
        chunk_size_words = int(max_tokens * 0.75) # Rough estimate, as tokens are sub-word units
        overlap_words = int(overlap * 0.75)
        chunks = []
        i = 0
        while i < len(words):
            chunk = ' '.join(words[i : i + chunk_size_words])
            chunks.append(chunk)
            i += chunk_size_words - overlap_words
            if i < 0: # Handle cases where overlap is larger than chunk_size_words
                i = 0
        return chunks

    # Better: use the model's tokenizer for accurate splitting
    tokens = tokenizer.encode(document_text, add_special_tokens=False)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunks.append(tokenizer.decode(chunk_tokens))
        i += max_tokens - overlap
        if i < 0: # Prevent going into negative indices if overlap is too large
            i = 0
    return chunks

def get_document_embedding(document_text, model):
    # Get the tokenizer from the SBERT model
    tokenizer = model.tokenizer
    max_tokens = model.max_seq_length # Get the max sequence length of the loaded model

    # Split into chunks (e.g., with overlap for better context)
    # Adjust max_tokens and overlap based on your chosen model and experimentation
    chunks = split_document_into_chunks(document_text, max_tokens=max_tokens, overlap=max_tokens//10, tokenizer=tokenizer)

    if not chunks:
        return np.zeros(model.get_sentence_embedding_dimension()) # Return zero vector for empty doc

    # Encode all chunks
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)

    # Aggregate embeddings (mean pooling is common and effective)
    document_embedding = np.mean(chunk_embeddings.cpu().numpy(), axis=0) # Move to CPU and convert to numpy

    return document_embedding  
    
def get_similarity_score(doc1_embedding, doc2_embedding):
    return cosine_similarity(doc1_embedding.reshape(1, -1), doc2_embedding.reshape(1, -1))[0][0]

def compare_content(newDocumentContent):
    my_dict = train_model()
    Documents_similaritys= {}
    newDocumentEmbedding = get_document_embedding(newDocumentContent, create_model())
    for key in my_dict:
        similarity = get_similarity_score(my_dict[key],newDocumentEmbedding)
        #print(similarity)
        # if similarity >= 0.75:
        Documents_similaritys[key] = round(float(similarity), 4)
        #print(f"Document '{key}' is similar to the new document with a similarity score of {similarity:.4f}")
    return Documents_similaritys

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
        Documents_similaritys = compare_content(text)
        json_response = json.dumps(Documents_similaritys)
        return json_response

    except Exception as e:
        return {"error": str(e)}

    finally:
        os.remove(temp_file)