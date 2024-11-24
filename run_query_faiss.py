from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
import pandas as pd 
from dataloader import get_data
import json
import argparse
import numpy as np
import pytrec_eval
from tqdm import tqdm 

def query_dense_index(query, model, index_path, metadata_path, top_k=5, max_seq_length=128):
    """
    Queries a dense FAISS index with a given query string.

    Parameters:
    - query (str): The query string.
    - model_name (str): Hugging Face model name for encoding (e.g., 'sentence-transformers/LaBSE').
    - index_path (str): Path to the stored FAISS index.
    - metadata_path (str): Path to the stored metadata file.
    - top_k (int): Number of nearest neighbors to retrieve.
    - max_seq_length (int): Maximum sequence length for the model (default is 128).

    Returns:
    - list of tuples: Top-k matches with their respective distances.
    """

    # Encode the query
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)  # Normalize query for inner product

    # Load the FAISS index
    index = faiss.read_index(index_path)

    # Load the metadata
    with open(metadata_path, "rb") as f:
        documents = pickle.load(f)

    # Search the index
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve results
    results = [(idx, distances[0][i]) for i, idx in enumerate(indices[0]) if idx != -1]
    return results

