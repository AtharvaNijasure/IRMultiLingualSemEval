import argparse
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from dataloader import get_data
from utils import read_json
from metrics import calculate_ndcg
import torch

from transformers import DPRConfig, DPRContextEncoder, DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from torch.utils.data import DataLoader, TensorDataset

import faiss
from tqdm import tqdm
import pytrec_eval
import sys
import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Check for Apple M2 GPU and set device
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use Metal Performance Shaders (MPS) for Apple GPUs
    print("Using Apple M2 GPU (MPS)")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

batch_size = 1

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='monolingual', help='monolingual or crosslingual')
parser.add_argument('--lang', type=str, default='eng', help='Language')
parser.add_argument('--split', type=str, default='train', help='train, dev')
args = parser.parse_args()



TASK = "monolingual" #args.task
LANG = "eng" # args.lang
SPLIT = "train" # args.split

print(f"Task: {TASK}, Language: {LANG}, Split: {SPLIT}")

df_fact_checks, df_posts, df_fact_check_post_mapping = get_data('./data')
tasks = read_json(f"./data/tasks.json")

posts_split = tasks[TASK][LANG][f'posts_{SPLIT}']
print(f"Number of posts in {SPLIT} set:", len(posts_split))

fact_checks = tasks[TASK][LANG]['fact_checks']
print("Number of fact checks:", len(fact_checks))

## filter dataframes
df_posts_split = df_posts[df_posts.index.isin(posts_split)]
assert len(df_posts_split) == len(posts_split)

df_fact_checks = df_fact_checks[df_fact_checks.index.isin(fact_checks)]
assert len(df_fact_checks) == len(fact_checks)



## DPR pre-processing

# concat all OCR text from source language (0th index)
df_posts_split['ocr_all_srclang'] = df_posts_split['ocr'].apply(lambda x: ' '.join([i[0] for i in x]) if x else "")

# extract text from source language (0th index)
df_posts_split['text_srclang'] = df_posts_split['text'].apply(lambda x: x[0] if x else "")

# query: OCR + text -- is the Social Media Post SMP
df_posts_split['query'] = df_posts_split['ocr_all_srclang'] + ' ' + df_posts_split['text_srclang']

# extract claim and title from source language (0th index)
df_fact_checks['claim_srclang'] = df_fact_checks['claim'].apply(lambda x: x[0] if x else "")
df_fact_checks['title_srclang'] = df_fact_checks['title'].apply(lambda x: x[0] if x else "")

# doc: claim + title --- FC doc
df_fact_checks['doc'] = df_fact_checks['claim_srclang'] + ' ' + df_fact_checks['title_srclang']




## DPR

# get doc embeddings i.e FC docs
tokenizerDoc = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
modelDocEnc = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device)



# Tokenize documents
tokenized_docs = tokenizerDoc(list(df_fact_checks['doc']), return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
# Create a dataset and data loader for batching
dataset = TensorDataset(tokenized_docs["input_ids"])
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


# print("here")
# Compute embeddings in batches
# embeddingsFCs_list = []
# modelDocEnc.eval()  # Set the model to evaluation mode
# with torch.no_grad():  # No need for gradients during inference
#     for batch in tqdm(data_loader, desc="Processing batches", total=len(data_loader)):
#         torch.cuda.empty_cache()
#         input_ids_batch = batch[0]  # Extract the input IDs from the batch
#         embeddings_batch = modelDocEnc(input_ids_batch).pooler_output
#         embeddingsFCs_list.append(embeddings_batch.cpu())
        

# # Concatenate all embeddings into a single tensor
# embeddingsFCs = torch.cat(embeddingsFCs_list, dim=0)

# torch.save(embeddingsFCs, '/gypsum/work1/allan/anijasure/multilingual_semeval/IRMultiLingualSemEval/embeddingsFCs.pt')

embeddingsFCs = torch.load('/gypsum/work1/allan/anijasure/multilingual_semeval/IRMultiLingualSemEval/embeddingsFCs.pt')



# Step 2: Create a FAISS index for document embeddings
d = embeddingsFCs.shape[1]  # Dimensionality of embeddings
nlist = 100  # Number of clusters (adjust based on data size)
m = 32  # Number of connections per node in HNSW (controls trade-off between accuracy and speed)

# quantizer = faiss.IndexFlatL2(d)  # L2 quantizer for clustering
# index = faiss.IndexHNSWFlat(d, m)  # HNSW index with Flat quantizer
# index.hnsw.efConstruction = 40  # Controls the quality of the graph during construction
# index.hnsw.efSearch = 50  # Controls the number of neighbors explored during a search

index = faiss.IndexFlatL2(d)  # Exact search using L2 distance


# quantizer = faiss.IndexFlatIP(d)  # L2 quantizer for clustering
# index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)  # HNSW index with Flat quantizer
# index.nprobe = 10  # Controls the quality of the graph during construction
# # index.hnsw.efSearch = 50  # Controls the number of neighbors explored during a search



# Add document embeddings to the FAISS index
index.add(embeddingsFCs)  # Add document embeddings to the FAISS index




# get Social Media Post embeddings    
tokenizerSMP = tokenizerDoc # DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
modelSMP = modelDocEnc #  DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").to(device)

# input_ids_SMP = tokenizerSMP(list(df_posts_split['query']), return_tensors="pt", padding= True )["input_ids"].to(device)
# embeddingsSMP = modelSMP(input_ids_SMP).pooler_output.cpu()


# Tokenize documents
input_ids_SMP = tokenizerDoc(list(df_posts_split['query']), return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

# Create a dataset and data loader for batching
dataset = TensorDataset(input_ids_SMP["input_ids"])
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Compute embeddings in batches
# embeddingsSMPs_list = []
# modelSMP.eval()  # Set the model to evaluation mode
# with torch.no_grad():  # No need for gradients during inference
#     for batch in data_loader:
#         input_ids_batch = batch[0]  # Extract the input IDs from the batch
#         embeddings_batch = modelSMP(input_ids_batch).pooler_output
#         embeddingsSMPs_list.append(embeddings_batch.cpu())

# # Concatenate all embeddings into a single tensor
# embeddingsSMP = torch.cat(embeddingsSMPs_list, dim=0)

# torch.save(embeddingsSMP, '/gypsum/work1/allan/anijasure/multilingual_semeval/IRMultiLingualSemEval/embeddingsSMPs.pt')

# embeddingsSMP = torch.load('/gypsum/work1/allan/anijasure/multilingual_semeval/IRMultiLingualSemEval/embeddingsSMPs.pt')


# Retrieve the nearest fact-check documents for each post
# k = 10  # Number of nearest neighbors
# distances, indices = index.search(embeddingsSMP, k )


fact_check_ids = df_fact_checks.index.tolist()
    
metrics=['P_3', 'P_5', 'P_10', 'map_cut_3', 'map_cut_5', 'map_cut_10', 'ndcg_cut_3', 'ndcg_cut_5', 'ndcg_cut_10']
average_scores = {metric: 0.0 for metric in metrics}

# Ensure FAISS index and embeddings are already created
# embeddingsFCs: Loaded document embeddings
# index: FAISS index built on embeddingsFCs
# embeddingsSMP: Social Media Post (query) embeddings
# df_posts_split: DataFrame containing posts with "query" column
# df_fact_check_post_mapping: DataFrame mapping posts to fact checks
# fact_check_ids: List of document IDs corresponding to embeddings in the index

for idx, row in df_posts_split.iterrows():
    qrels = {}
    runs = {}

    # Get the query text
    query = row['query']

    # Compute the embedding for the query
    tokenized_query = tokenizerSMP(
        query,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    with torch.no_grad():
        query_embedding = modelSMP(tokenized_query["input_ids"]).pooler_output.cpu().numpy()

    # Search the FAISS index for the top 10 nearest neighbors
    k = 10  # Number of top results
    distances, indices = index.search(query_embedding, k)

    # Get the ranked fact-check IDs and scores
    ranked_fact_checks = [fact_check_ids[i] for i in indices[0]]
    ranked_fact_checks_scores = distances[0]

    # Get ground truth fact-check IDs for the post
    ground_truth = df_fact_check_post_mapping[
        df_fact_check_post_mapping['post_id'] == idx
    ]['fact_check_id'].tolist()

    # Prepare QRELs and runs for evaluation
    qrels[str(idx)] = {str(fc_id): 1 for fc_id in ground_truth}
    runs[str(idx)] = {str(fc_id): float(score) for fc_id, score in zip(ranked_fact_checks, ranked_fact_checks_scores)}

    # Evaluate using pytrec_eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    results = evaluator.evaluate(runs)

    for query_id in results:
        for metric in metrics:
            average_scores[metric] += results[query_id][metric]

    
for metric in average_scores:
    average_scores[metric] /= len(df_posts_split)

print("Average scores:")
print(average_scores)



