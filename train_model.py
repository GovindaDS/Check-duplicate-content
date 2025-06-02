from pyexpat import model
from sentence_transformers import SentenceTransformer
import numpy as np
import textwrap
import re
# Calculate cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
print("train_model.py loaded")
docTest="""The challenge with large documents is that transformer models like BERT and
    its derivatives have a fixed maximum input length, typically 512 tokens.
    If a document exceeds this, it gets truncated, losing crucial information.
    Therefore, we need strategies like chunking the document into smaller pieces
    and then aggregating their embeddings. Mean pooling is a common way to do this,
    where we take the average of all chunk embeddings to represent the whole document."""
 
def create_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model
def train_model():
    # Example Usage:
    large_document_1 = """
    This is a very long document. It contains multiple paragraphs and spans
    several pages. We need to process this document to find its semantic
    similarity with other documents in our learning management system (LMS).
    Traditional keyword-based methods often fail to capture the true meaning,
    especially when documents use synonyms or rephrase concepts.

    For example, consider a research paper on machine learning. It might discuss
    "neural networks," "deep learning algorithms," or "artificial intelligence models."
    These phrases are semantically similar but lexically different. An AI-based
    tool needs to understand this underlying meaning.

    The challenge with large documents is that transformer models like BERT and
    its derivatives have a fixed maximum input length, typically 512 tokens.
    If a document exceeds this, it gets truncated, losing crucial information.
    Therefore, we need strategies like chunking the document into smaller pieces
    and then aggregating their embeddings. Mean pooling is a common way to do this,
    where we take the average of all chunk embeddings to represent the whole document.

    Another aspect to consider is identifying outdated documents. This can be
    based on version numbers, last modified dates, or even detecting references
    to old policies or data. The AI system would combine semantic similarity
    with metadata analysis to achieve this.
    """

    large_document_2 = """
    A comprehensive guide to handling extensive texts in NLP, particularly
    for semantic understanding and information retrieval in large datasets.
    This includes techniques for breaking down long articles into manageable
    sections, generating vector representations using advanced models, and
    combining these representations to summarize the overall content.

    Modern natural language processing (NLP) architectures, such as those
    based on the Transformer paradigm, are highly effective for tasks like
    semantic search and document clustering. However, their quadratic
    complexity concerning input length poses a significant challenge for
    documents containing thousands of words. Methods such as sliding window
    tokenization with overlapping segments are crucial for mitigating context
    loss during the chunking process.

    Furthermore, the process of identifying redundant or obsolete educational
    materials within a learning management system (LMS) can greatly benefit
    from automated tools. By leveraging semantic embeddings, systems can
    detect documents that convey the same core information, even if their
    wording differs. This is vital for maintaining a clean and current
    repository of learning resources.
    """
    model = create_model()
    my_dict = {}
    my_dict["large_document_1"]= get_document_embedding(large_document_1, model)
    my_dict["large_document_2"]= get_document_embedding(large_document_2, model)
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
        print(similarity)
        if similarity >= 0.75:
            Documents_similaritys[key] = similarity
            print(f"Document '{key}' is similar to the new document with a similarity score of {similarity:.4f}")
    return Documents_similaritys
