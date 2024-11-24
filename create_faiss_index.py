from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
import pandas as pd 

def create_and_store_dense_index_approx(documents, model_name, output_dir, max_seq_length=256, nlist=100, nprobe=10):
    """
    Creates an approximate nearest neighbor dense index using Sentence-Transformers and FAISS.

    Parameters:
    - documents (list of str): The list of documents to index.
    - model_name (str): Hugging Face model name to use for encoding (e.g., 'sentence-transformers/LaBSE').
    - output_dir (str): Directory where the index and metadata will be stored.
    - max_seq_length (int): Maximum sequence length for the model (default is 128 for LaBSE).
    - nlist (int): Number of clusters for the FAISS IndexIVFFlat.

    Returns:
    - None
    """
    # Step 1: Load the model and adjust the max_seq_length
    model = SentenceTransformer(model_name)
    #model.max_seq_length = max_seq_length  # Enforce maximum sequence length

    # Step 2: Encode the documents
    print("Encoding documents...")
    embeddings = model.encode(documents, show_progress_bar=True)

    # Step 3: Create a FAISS IndexIVFFlat
    print("Creating FAISS IndexIVFFlat...")
    dimension = embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(dimension)  # Inner product quantizer
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

    # Train the index with document embeddings
    print("Training FAISS index...")
    index.train(embeddings)
    index.add(embeddings)  # Add embeddings to the index
    
    index.nprobe = nprobe
    
    # Step 4: Normalize embeddings for Dot Product (optional)
    # FAISS inner product is equivalent to cosine similarity if vectors are normalized
    faiss.normalize_L2(embeddings)

    # Step 5: Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Step 6: Save the FAISS index
    index_path = os.path.join(output_dir, "faiss_index_ivf")
    print(f"Saving FAISS index to {index_path}...")
    faiss.write_index(index, index_path)

    # Step 7: Save the metadata (documents list)
    metadata_path = os.path.join(output_dir, "metadata.pkl")
    print(f"Saving metadata to {metadata_path}...")
    with open(metadata_path, "wb") as f:
        pickle.dump(documents, f)

    print("Index and metadata successfully saved.")




    
    
