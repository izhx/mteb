"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

import logging
import math
from typing import Dict, List

from mteb import MTEB
from sentence_transformers import SentenceTransformer
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s : %(message)s'
)

logger = logging.getLogger(__name__)

TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
    "SummEval",
]

TASK_LIST = TASK_LIST_CLASSIFICATION + TASK_LIST_CLUSTERING + TASK_LIST_PAIR_CLASSIFICATION + TASK_LIST_RERANKING + TASK_LIST_RETRIEVAL + TASK_LIST_STS


class Wrapper:
    """
    A multi-gpu wrapper class, fit to all tasks, also compatible with BEIR.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        batch_size: int,
        sep: str = " ",
        mp_tensor_to_cuda: bool = True,
    ):
        self.model = model
        self.batch_size = batch_size
        self.sep = sep
        self.pool: dict = None
        self.mp_threshhold = 0
        self.mp_tensor_to_cuda = mp_tensor_to_cuda

    def start(self):
        self.pool = self.model.start_multi_process_pool()

    def stop(self):
        self.model.stop_multi_process_pool(self.pool)

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,  # will overwrite batch_size from evaluators.
        show_progress_bar: bool = False,
        # output_value: str = 'sentence_embedding',
        # convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        # device: str = None,
        # normalize_embeddings: bool = False
        **kwargs
    ):
        if self.pool is not None and len(sentences) > self.mp_threshhold:
            part_size = math.ceil(len(sentences) / len(self.pool["processes"]))
            chunk_size = part_size if part_size < 3200 else 3200  # for retrieval chunk 50000
            embeddings = self.model.encode_multi_process(
                sentences, self.pool, self.batch_size, chunk_size=chunk_size
            )
            if convert_to_tensor:
                embeddings = torch.from_numpy(embeddings)
                if self.mp_tensor_to_cuda and torch.cuda.is_available():
                    embeddings = embeddings.to(torch.device('cuda'))  # default 0-th gpu
            return embeddings

        return self.model.encode(
            sentences, batch_size=self.batch_size, show_progress_bar=False,
            convert_to_tensor=convert_to_tensor, **kwargs
        )

    def encode_queries(self, queries: List[str], **kwargs):
        return self.encode(queries, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs):
        # borrowed from mteb.abstasks.AbsTaskRetrieval.DRESModel
        if type(corpus) is dict:
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                (doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip()
                for doc in corpus
            ]
        return self.encode(sentences, **kwargs)


model_name = "average_word_embeddings_komninos"
model = SentenceTransformer(model_name)

if torch.cuda.device_count() > 1:
    model = Wrapper(model, batch_size=256)
    model.start()  # multi-GPU pool start

for task in TASK_LIST:
    logger.info(f"Running task: {task}")
    eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
    evaluation = MTEB(tasks=[task], task_langs=["en"]) # Remove "en" for running all languages
    evaluation.run(model, output_folder=f"results/{model_name}", eval_splits=eval_splits)

if torch.cuda.device_count() > 1:
    model.stop()
