'''
This examples show how to train a basic Bi-Encoder for any BEIR dataset without any mined hard negatives or triplets.

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass pairs in the format:
(query, positive_passage). Other positive passages within a single batch becomes negatives given the pos passage.

We do not mine hard negatives or train triplets in this example.

Running this script:
python train_sbert.py
'''

from sentence_transformers import losses, models, SentenceTransformer
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os
import logging
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
        "--train_type",
        type=str,
        default=None,
        required=True,
        help="finetuning method 'few', 'full'",
    )
    parser.add_argument(
        '--epoch',
        default=1,
        type = int,
        help='finetuning epoch',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Trained model dir',
    )
    parser.add_argument(
        '--scoring',
        type=str,
        default='cos_sim',
        help='scoring function for relevance search',        
    )
    parser.add_argument(
        '--training_data_path',
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
    parser.add_argument(
        '--sampling_version',
        type=int,
        default = 1,
        help='Few-shot sampling version: 1-first 10 percent samples, 2-random 10 percent samples, 3-random 50 samples, 4-random 3 qrels',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default = 1001,
        help='seed',
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

    #### Download nfcorpus.zip dataset and unzip the dataset
    #datasets =['scifact']
    # + nfcorpus
    # datasets = ["trec-covid", "nfcorpus",
    #                      "nq", "hotpotqa", "fiqa", "arguana",
    #                      "webis-touche2020", "quora",
    #                      "dbpedia-entity", "scidocs", "fever",
    #                      "climate-fever", "scifact", "germanquad"]
    # url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    # out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    # data_path = util.download_and_unzip(url, out_dir)


    # #### Provide the data_path where nfcorpus has been downloaded and unzipped
    # corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")
    # #### Please Note not all datasets contain a dev split, comment out the line if such the case
    # dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")

    #### Provide any sentence-transformers or HF model
    # model_name = "BeIR/sparta-msmarco-distilbert-base-v1"
    # word_embedding_model = models.Transformer(model_name, max_seq_length=350)
    # pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    # model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    #### Or provide pretrained sentence-transformer model
    model_name="msmarco-distilbert-base-v3"
    model = SentenceTransformer("msmarco-distilbert-base-v3")

    retriever = TrainRetriever(model=model, batch_size=args.batch_size)
        
    dataset_dir= Path(args.training_data_path)
    datafile=Path(args.dataset)
    data_path = os.path.join(dataset_dir, datafile)
    
    #### Provide the data_path where nfcorpus has been downloaded and unzipped
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")
    #### Please Note not all datasets contain a dev split, comment out the line if such the case
    #dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")
    
    #### Prepare training samples
    if args.sampling_version==int(4) and args.train_type=='few':
        train_samples = retriever.load_train_qrel(corpus, queries, qrels, args.seed)
    else:
        random.seed(args.seed)
        train_samples = retriever.load_train(corpus, queries, qrels)
        if args.sampling_version==int(1) and args.train_type=='few':
            train_samples = train_samples[:int(len(train_samples)/10)]
        elif args.sampling_version==int(2) and args.train_type=='few':
            train_samples = random.sample(train_samples, int(len(train_samples)/10))
        elif args.sampling_version==int(3) and args.train_type=='few':
            train_samples = random.sample(train_samples, 64)
    print(len(train_samples))
    train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

    #### Training SBERT with cosine-product
    train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
    #### training SBERT with dot-product
    # train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.dot_score)

    #### Prepare dev evaluator
    #ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)

    #### If no dev set is present from above use dummy evaluator
    # ir_evaluator = retriever.load_dummy_evaluator()

    #### Provide model save path
    output_dir=Path(args.output_dir)
    output_file=Path("{}-{}-{}-v{}-{}ep-{}".format(model_name, args.train_type,args.dataset,args.sampling_version,args.epoch,args.seed))
    model_save_path = Path(output_dir/output_file)
    os.makedirs(model_save_path, exist_ok=True)

    logging.basicConfig(filename='ft_{}.log'.format(args.dataset),
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger=logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.info("Dataset: {}".format(args.dataset))  
    #### Configure Train params
    num_epochs = args.epoch
    evaluation_steps = 5000
    warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)
    
    logger.info("epoch: {}".format(num_epochs))
    logger.info("evaluation_steps: {}".format(evaluation_steps))
    
    retriever.fit(train_objectives=[(train_dataloader, train_loss)], 
                    #evaluator=ir_evaluator, 
                    epochs=num_epochs,
                    output_path=str(model_save_path),
                    warmup_steps=warmup_steps,
                    evaluation_steps=evaluation_steps,
                    use_amp=True)
        
if __name__ == "__main__":
    args = get_args()
    main(args)
