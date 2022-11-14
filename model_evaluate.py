from time import time
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os
from pathlib import Path
import argparse
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=None,
        type=str,
        required=True,
        help="ft dataset 'nfcorpus', 'fiqa', 'scifact'",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="model path",
    )
    parser.add_argument(
        '--scoring',
        type=str,
        default='cos_sim',
        help='scoring function for relevance search',
    )
    parser.add_argument(
        '--test_data_path',
        type=str,
        default='/media/disk1/intern1001',
        help='Training dataset dir',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Model batch size',
    )
    parser
    args=parser.parse_args()
    return args

def main(args):
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                         datefmt='%Y-%m-%d %H:%M:%S',
                         level=logging.INFO,
                         handlers=[LoggingHandler()])
    #### /print debug information to stdout

    #### Download scifact.zip dataset and unzip the dataset
    dataset_dir = Path(args.test_data_path)
    datafile = Path(args.dataset)
    data_path = os.path.join(dataset_dir, datafile)
    #### Provide the data_path where scifact has been downloaded and unzipped
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    #### Load the SBERT model and retrieve using cosine-similarity
    model_path = args.model_path
    model = DRES(models.SentenceBERT(model_path), batch_size=args.batch_size)
    retriever = EvaluateRetrieval(model, score_function=args.scoring) # or "cos_sim" for cosine similarity

    #### Retrieve dense results (format of results is identical to qrels)
    start_time = time()
    results = retriever.retrieve(corpus, queries)
    end_time = time()
    print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
    #### Evaluate your retrieval using NDCG@k, MAP@K ...

    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
    recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="r_cap")
    hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")

if __name__ == "__main__":
    args = get_args()
    main(args)
