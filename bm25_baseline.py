import argparse
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from dataloader import get_data
from utils import read_json
from metrics import calculate_ndcg

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='monolingual', help='monolingual or crosslingual')
    parser.add_argument('--lang', type=str, default='eng', help='Language')
    parser.add_argument('--split', type=str, default='train', help='train, dev')
    args = parser.parse_args()

    TASK = args.task
    LANG = args.lang
    SPLIT = args.split

    print(f"Task: {TASK}, Language: {LANG}, Split: {SPLIT}")

    df_fact_checks, df_posts, df_fact_check_post_mapping = get_data('../data')
    tasks = read_json(f"../data/tasks.json")

    posts_split = tasks[TASK][LANG][f'posts_{SPLIT}']
    print(f"Number of posts in {SPLIT} set:", len(posts_split))

    fact_checks = tasks[TASK][LANG]['fact_checks']
    print("Number of fact checks:", len(fact_checks))

    ## filter dataframes
    df_posts_split = df_posts[df_posts.index.isin(posts_split)]
    assert len(df_posts_split) == len(posts_split)

    df_fact_checks = df_fact_checks[df_fact_checks.index.isin(fact_checks)]
    assert len(df_fact_checks) == len(fact_checks)

    ## BM25 pre-processing

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


    ## BM25
    corpus = df_fact_checks['doc'].tolist()
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    fact_check_ids = df_fact_checks.index.tolist()

    # Calculate NDCG@10 for each query
    ndcg_scores = []
    k = 10

    for idx, row in df_posts_split.iterrows():
        query = row['query']
        tokenized_query = query.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)

        ranked_fact_checks = [fact_check_ids[i] for i in np.argsort(doc_scores)[::-1]]

        ground_truth = df_fact_check_post_mapping[
            df_fact_check_post_mapping['post_id'] == idx
        ]['fact_check_id'].tolist()
        print("Ground truth:", ground_truth)

        # Calculate NDCG@10
        ndcg = calculate_ndcg(ranked_fact_checks, ground_truth, k)
        ndcg_scores.append(ndcg)
        print("NDCG@10:", ndcg)
        
    # Calculate and print average NDCG@10
    mean_ndcg = np.mean(ndcg_scores)
    print(f"\nAverage NDCG@{k}: {mean_ndcg:.4f}")
