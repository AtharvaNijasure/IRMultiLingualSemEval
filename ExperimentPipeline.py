from create_faiss_index import create_and_store_dense_index_approx
import pandas as pd
from run_query_faiss import query_dense_index
from sentence_transformers import SentenceTransformer
from dataloader import get_data
from tqdm import tqdm
import pytrec_eval
import os
import json


class ExperimentPipeline:
    def __init__(self, TASK, SPLIT, model_name, model_name_short, create_and_store_index = False, device = 'cpu'):
        """
        Initializes the experiment pipeline with task, model, and indexing configurations.
        
        Args:
            TASK (str): The task type, e.g., 'monolingual' or 'crosslingual'.
            SPLIT (str): The dataset split, e.g., 'train', 'dev', or 'test'.
            model_name (str): The name of the full model to use (e.g., 'facebook/dpr-ctx_encoder-single-nq-base').
            model_name_short (str): A short alias for the model name.
            create_and_store_index (bool): Whether to create and save the FAISS index.
            device (torch.device): The device to use (e.g., 'cuda' or 'cpu').
        """
        self.TASK = TASK
        self.SPLIT = SPLIT
        self.model_name = model_name
        self.model_name_short = model_name_short
        self.create_and_store_index = create_and_store_index
        self.device = device

        # Placeholder attributes for data, models, and index
        # Get documents as a list of fact-checks from docs_monolingual.csv
        self.doc_csv_path = f"docs_{self.TASK}_{self.SPLIT}.csv"
        self.query_csv_path = f"queries_{self.TASK}_{self.SPLIT}.csv"
        self.df_documents = pd.read_csv(self.doc_csv_path)
        self.documents = self.df_documents['doc'].astype(str).tolist()

        self.output_directory = f"{self.TASK}_{self.SPLIT}_dense_index_bert_codensor_ivf"
        self.FINAL_OUTPUT_DIR = f'{self.model_name_short}_{self.TASK}_{self.SPLIT}'

        self.device = "cuda"

        # Index path
        self.index_path = f"{self.output_directory}/faiss_index_ivf"
        self.metadata_path = f"{self.output_directory}/metadata.pkl"
            

        
        # Load data for the specified task and split
        self.df_fact_checks, self.df_posts, self.df_fact_check_post_mapping = self.load_data()

        # Prepare and tokenize documents and queries
        self.prepare_documents()

        # Initialize models
        self.initialize_models()

        # Optionally create and store FAISS index
        # if self.create_and_store_index:
        #     self.create_faiss_index(documents, self.model_name, self.output_directory)

    def load_data(self):
        """
        Loads data for the experiment pipeline.

        Returns:
            Tuple containing:
                - df_fact_checks: DataFrame for fact checks.
                - df_posts: DataFrame for posts.
                - df_fact_check_post_mapping: DataFrame mapping posts to fact checks.
        """
        print("Loading data...")
        df_fact_checks, df_posts, df_fact_check_post_mapping = get_data('./data')
        tasks = read_json(f"./data/tasks.json")

        posts_split = tasks[self.TASK][self.SPLIT]['posts_train']
        fact_checks = tasks[self.TASK][self.SPLIT]['fact_checks']

        df_posts_split = df_posts[df_posts.index.isin(posts_split)]
        df_fact_checks = df_fact_checks[df_fact_checks.index.isin(fact_checks)]
        return df_fact_checks, df_posts_split, df_fact_check_post_mapping

    def prepare_documents(self):
        """
        Prepares documents and queries for the experiment pipeline by concatenating
        and tokenizing necessary text fields.
        """
        print("Preparing documents...")
        self.df_posts['ocr_all_srclang'] = self.df_posts['ocr'].apply(
            lambda x: ' '.join([i[0] for i in x]) if x else ""
        )
        self.df_posts['query'] = self.df_posts['ocr_all_srclang'] + ' ' + self.df_posts['text'].apply(
            lambda x: x[0] if x else ""
        )

        self.df_fact_checks['claim_srclang'] = self.df_fact_checks['claim'].apply(
            lambda x: x[0] if x else ""
        )
        self.df_fact_checks['doc'] = self.df_fact_checks['claim_srclang'] + ' ' + self.df_fact_checks['title'].apply(
            lambda x: x[0] if x else ""
        )

    def initialize_models(self):
        """
        Initializes tokenizers and models for encoding documents and queries.
        """
        print("Initializing models...")
        self.model = SentenceTransformer(self.model_name).to(self.device)
        

    def create_faiss_index(self,documents, model_name, output_directory):
        """
        Creates and stores a FAISS index for document embeddings.
        """
        print("Creating FAISS index...")
        create_and_store_dense_index_approx(documents, model_name, output_directory)

        print("FAISS index created and stored.")
