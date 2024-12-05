import argparse
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from dataloader import get_data
from utils import read_json
from metrics import calculate_ndcg
import torch

from transformers import DPRConfig, DPRContextEncoder, DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, pipeline
from torch.utils.data import DataLoader, TensorDataset

import faiss
from tqdm import tqdm
import pytrec_eval
import sys
import os
import csv
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import multiprocessing as mp
from functools import partial
import numpy as np

from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')

 # Function to write rows to a CSV file incrementally
def write_row_to_csv(row, file_path, header=False, i = 0):
    try:
        with open(file_path, mode='a', newline='', encoding='utf-8') as csvfile:
            # Use csv.DictWriter for dictionary-style row writing
            writer = csv.DictWriter(csvfile, fieldnames=row.index if hasattr(row, 'index') else row.keys())
            
            if header:
                writer.writeheader()
            
            # Convert row to dictionary before writing
            writer.writerow(row.to_dict() if hasattr(row, 'to_dict') else dict(row))
            
        # if i is not None:
        #     print(f"Row {i} written successfully.")
        
    except Exception as e:
        print(f"An error occurred while writing row {i}: {e}")

def save_dataframe_to_csv(df, file_path, index=True, encoding='utf-8'):
    """
    Saves a pandas DataFrame to a CSV file, including the index by default.

    Parameters:
    - df (pd.DataFrame): The DataFrame to save.
    - file_path (str): The path where the CSV file will be saved.
    - index (bool): Whether to include the DataFrame index in the CSV file. Default is True.
    - encoding (str): The encoding to use for the CSV file. Default is 'utf-8'.

    Returns:
    - None
    """
    try:
        df.to_csv(file_path, index=index, encoding=encoding)
        print(f"DataFrame successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the DataFrame: {e}")
        
   

def process_chunk(chunk, model_name):
    """Process a chunk of documents"""
    chunk['expanded_doc'] = chunk['doc'].apply(lambda x: expand_document(x, model_name))
    return chunk

def add_expanded_doc_column_parallel(df, model_name="llama2", num_processes=4):
    """Parallel processing of document expansion"""
    # Split DataFrame into chunks
    chunks = np.array_split(df, num_processes)
    
    # Create pool and process chunks
    with mp.Pool(processes=num_processes) as pool:
        # Create partial function with fixed model_name
        process_func = partial(process_chunk, model_name=model_name)
        # Process chunks in parallel
        results = pool.map(process_func, chunks)
    
    # Combine results
    df_expanded = pd.concat(results)
    return df_expanded.sort_index()

# Usage
def add_expanded_doc_column(df, model_name="llama2"):
    # Get number of CPU cores (leave one free for system)
    num_processes = max(1, mp.cpu_count() - 1)
    return add_expanded_doc_column_parallel(df, model_name, num_processes)

# Function to expand document using LLM
def expand_document(doc, model_name="llama2", device = 'cuda'):
    # Initialize the LLM pipeline
    generator = pipeline('text-generation', model=model_name, device = device)
    
    # Create the prompt
    pre_prompt = "Expand the following fact-checked claim and title by generating up to 10 concise, relevant keywords in a single line. These keywords should provide additional context to enhance the retrieval of this document for fact-checking purposes. Avoid repeating any words already present in the claim or title. Output only the keywords, separated by commas, without any additional text:"
    pre_prompt_2 = "just give the keywords, separated by commas, without ANY additional text or notations like 'Keywords:'."
    prompt = f"{pre_prompt}\n\nClaim and Context:{doc}\n Answer Instruction: Do not include the prompt and {pre_prompt_2}"
    # prompts = [f"Expand the following fact-checked claims and title with additional context such that if there is some social media post we can retrieve this document for fact checking only if this document is relevant. Do not repeat the present words in the claims\n\n{doc}\n\nContext:" for doc in batch]
        
    # Generate the expanded context
    ec = generator(prompt, max_new_tokens=100, num_return_sequences=1, pad_token_id=generator.tokenizer.eos_token_id)
    expanded_context_set = set(ec[0]['generated_text'].split("'Keywords:'.")[-1].replace("\n"," ").split(","))
    expanded_context = " ".join(map(str, expanded_context_set))
    
    
    # print(f"Prompt: {prompt}")
    
    # print(f"Expanded context: {expanded_context}")
    
    # Concatenate the original doc with the expanded context
    expanded_doc = f"{doc} {expanded_context}"
    
    return expanded_doc



# Check for Apple M2 GPU and set device
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use Metal Performance Shaders (MPS) for Apple GPUs
    print("Using Apple M2 GPU (MPS)")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

batch_size = 4


df_fact_checks, df_posts, df_fact_check_post_mapping = get_data('./data')
tasks = read_json(f"./data/tasks.json")


TASK = "monolingual" #args.task
LANG = "eng" # args.lang
SPLIT = "train" # args.split

print(f"Task: {TASK}, Language: {LANG}, Split: {SPLIT}")

posts_split = tasks[TASK][LANG][f'posts_{SPLIT}']
print(f"Number of posts in {SPLIT} set:", len(posts_split))

fact_checks = tasks[TASK][LANG]['fact_checks']
print("Number of fact checks:", len(fact_checks))

## filter dataframes
df_posts_split = df_posts[df_posts.index.isin(posts_split)]
assert len(df_posts_split) == len(posts_split)

df_fact_checks = df_fact_checks[df_fact_checks.index.isin(fact_checks)]
assert len(df_fact_checks) == len(fact_checks)




df_posts_split['query_ids'] = df_posts_split.index

df_fact_checks['doc_ids'] = df_fact_checks.index

## Extract the source language 

# concat all OCR text from source language (0th index)
df_posts_split['ocr_all_srclang'] = df_posts_split['ocr'].apply(lambda x: ' '.join([i[0] for i in x]) if x else "")

# extract text from source language (0th index)
df_posts_split['text_srclang'] = df_posts_split['text'].apply(lambda x: x[0] if x else "")

# query: OCR + text
df_posts_split['query'] = df_posts_split['ocr_all_srclang'] + ' ' + df_posts_split['text_srclang']

# extract claim and title from source language (0th index)
df_fact_checks['claim_srclang'] = df_fact_checks['claim'].apply(lambda x: x[0] if x else "")
df_fact_checks['title_srclang'] = df_fact_checks['title'].apply(lambda x: x[0] if x else "")

# doc: claim + title
df_fact_checks['doc'] = df_fact_checks['claim_srclang'] + ' ' + df_fact_checks['title_srclang']




# Example usage
if __name__ == "__main__":
    
    # Assuming df_fact_checks is already loaded and contains the 'doc' column

    # Add the expanded_doc column
    output_file = "p0_llama_3_2_1b__expanded_docs_monolingual_train.csv"
    header_written = False

    # Process each row and write incrementally
    model_name = "meta-llama/Llama-3.2-1B"
    i   = 0
    istart = 0 #17038 # 17038/153743
    # iend = 153743
    p1 = 44379
    p2 = 71720
    p3 = 99061
    p4 = 126402
    iend = 17038
    for _, row in tqdm(df_fact_checks.iterrows(), total=len(df_fact_checks), desc="Processing Rows"):
        i += 1
        if(i<istart or i > iend) :
            print(f"Processed {i} rows.")
            # break
            continue
        row['expanded_doc'] = expand_document(row['doc'], model_name=model_name, device = device)
        write_row_to_csv(row, output_file, header=not header_written, i = i)
        header_written = True  # Ensure the header is written only once
        

    print(f"Expanded documents saved incrementally to {output_file}.")