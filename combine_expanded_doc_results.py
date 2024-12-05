import pandas as pd
import numpy as np
import os
import sys
import csv
from tqdm import tqdm

def combine_csv_files(files, combined_file):
    """
    Combine multiple CSV files into one, ensuring header is written only once.
    
    Parameters:
    - files (list): List of file paths to combine
    - combined_file (str): Path to the output combined file
    """
    # Open the combined file in write mode
    with open(combined_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = None
        
        # Iterate over each file
        for i, file in enumerate(tqdm(files, desc="Combining files")):
            # Read the current file into a DataFrame
            df = pd.read_csv(file)
            
            # Write header only for the first file
            if i == 0:
                df.to_csv(outfile, index=False, header=True)
            else:
                df.to_csv(outfile, index=False, header=False, mode='a')



# Usage
files = [
    "p0_llama_3_2_1b__expanded_docs_monolingual_train.csv",
    # "llama_3_2_1b__expanded_docs_crosslingual_dev.csv",
    "p1_llama_3_2_1b__expanded_docs_monolingual_train.csv",
    "p2_llama_3_2_1b__expanded_docs_monolingual_train.csv",
    "p3_llama_3_2_1b__expanded_docs_monolingual_dev.csv" #,
    # "p4_llama_3_2_1b__expanded_docs_monolingual_train.csv",
    # "p5_llama_3_2_1b__expanded_docs_monolingual_train.csv"
]

combined_file = "final_expanded_docs_monolingual_train.csv"


combine_csv_files(files, combined_file)
