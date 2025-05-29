from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import json

# Sample dataset
documents = [
    {"id": "doc1", "content": "Yes, there are third-party libraries that can assist with data manipulation, including reading data from tables and documents, and comparing content within a .NET environment. For data manipulation and analysis, DataTable and DataSet in ADO.NET are fundamental, and libraries like Newtonsoft.Json (for JSON serialization/deserialization) and potentially Pandas (if you need to work with more complex data structures and algorithms, though it's typically used with Python) can be helpful. For document processing, libraries like iTextSharp (for PDF manipulation), Open XML SDK (for working with Office documents), or specialized OCR libraries (for image-based text extraction) can be considered. These can be integrated into a .NET application to scan data tables, create datasets, and compare document content. "},
    {"id": "doc2", "content": "Payment was received on time."},
    {"id": "doc3", "content": "The client paid the invoice in full."}
]

# Extract contents
texts = [doc["content"] for doc in documents]
ids = [doc["id"] for doc in documents]

# Create TF-IDF model
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(texts)

# Save the vectorizer model
with open("tfidf_model.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save the document vectors
with open("vector_data.json", "w") as f:
    json.dump({
        "ids": ids,
        "vectors": vectors.toarray().tolist()
    }, f)

print("✅ Model and vector data saved.")
