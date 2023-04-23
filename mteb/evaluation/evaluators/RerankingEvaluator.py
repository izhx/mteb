import logging

import numpy as np
import torch
import tqdm
from sklearn.metrics import average_precision_score

from .Evaluator import Evaluator
from .utils import cos_sim

logger = logging.getLogger(__name__)


class RerankingEvaluator(Evaluator):
    """
    This class evaluates a SentenceTransformer model for the task of re-ranking.
    Given a query and a list of documents, it computes the score [query, doc_i] for all possible
    documents and sorts them in decreasing order. Then, MRR@10 and MAP is compute to measure the quality of the ranking.
    :param samples: Must be a list and each element is of the form:
        - {'query': '', 'positive': [], 'negative': []}. Query is the search query, positive is a list of positive
        (relevant) documents, negative is a list of negative (irrelevant) documents.
        - {'query': [], 'positive': [], 'negative': []}. Where query is a list of strings, which embeddings we average
        to get the query embedding.
    """

    def __init__(
        self,
        samples,
        mrr_at_k: int = 10,
        name: str = "",
        similarity_fct=cos_sim,
        batch_size: int = 128,
        use_chunked_encoding: bool = True,
        chunk_size: int = 1500,
        limit: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if limit:
            samples = samples.train_test_split(limit)["test"]
        self.samples = samples
        self.name = name
        self.mrr_at_k = mrr_at_k
        self.similarity_fct = similarity_fct
        self.batch_size = batch_size
        self.use_chunked_encoding = use_chunked_encoding
        self.chunk_size = chunk_size

        if isinstance(self.samples, dict):
            self.samples = list(self.samples.values())

        ### Remove sample with empty positive / negative set
        self.samples = [
            sample for sample in self.samples if len(sample["positive"]) > 0 and len(sample["negative"]) > 0
        ]

    def __call__(self, model):
        scores = self.compute_metrics(model)
        return scores

    def compute_metrics(self, model):
        return (
            self.compute_metrics_chunked(model)
            if self.use_chunked_encoding
            else self.compute_metrics_individual(model)
        )

    def compute_metrics_chunked(self, model):
        """
        Computes the metrices in a batched way, by batching all queries and
        all documents into chunks.
        """
        all_mrr_scores = []
        all_ap_scores = []

        for n, i in enumerate(range(0, len(self.samples), self.chunk_size)):
            chunked_samples = self.samples[i : i + self.chunk_size]
            logger.info(f"Processing no.{n} chunk with {len(chunked_samples)} examples...")

            if isinstance(chunked_samples[0]["query"], str):
                logger.info(f"Encoding {len(chunked_samples)} queries...")
                all_query_embs = model.encode(
                    [sample["query"] for sample in chunked_samples],
                    convert_to_tensor=True,
                    batch_size=self.batch_size,
                )
            elif isinstance(chunked_samples[0]["query"], list):
                # In case the query is a list of strings, we get the most similar embedding to any of the queries
                all_query_flattened = [q for sample in chunked_samples for q in sample["query"]]
                logger.info(f"Encoding flattened {len(all_query_flattened)} queries...")
                all_query_embs = model.encode(all_query_flattened, convert_to_tensor=True, batch_size=self.batch_size)
            else:
                raise ValueError(
                    f"Query must be a string or a list of strings but is {type(chunked_samples[0]['query'])}"
                )

            all_docs = []
            for sample in chunked_samples:
                all_docs.extend(sample["positive"])
                all_docs.extend(sample["negative"])

            logger.info(f"Encoding {len(all_docs)} candidates...")
            all_docs_embs = model.encode(all_docs, convert_to_tensor=True, batch_size=self.batch_size)

            # Compute scores
            logger.info("Evaluating...")
            query_idx, docs_idx = 0, 0
            for instance in chunked_samples:
                num_subqueries = len(instance["query"]) if isinstance(instance["query"], list) else 1
                query_emb = all_query_embs[query_idx : query_idx + num_subqueries]
                query_idx += num_subqueries

                num_pos = len(instance["positive"])
                num_neg = len(instance["negative"])
                docs_emb = all_docs_embs[docs_idx : docs_idx + num_pos + num_neg]
                docs_idx += num_pos + num_neg

                if num_pos == 0 or num_neg == 0:
                    continue

                is_relevant = [True] * num_pos + [False] * num_neg

                scores = self._compute_metrics_instance(query_emb, docs_emb, is_relevant)
                all_mrr_scores.append(scores["mrr"])
                all_ap_scores.append(scores["ap"])

        mean_ap = np.mean(all_ap_scores)
        mean_mrr = np.mean(all_mrr_scores)

        return {"map": mean_ap, "mrr": mean_mrr}

    def compute_metrics_individual(self, model):
        """
        Embeds every (query, positive, negative) tuple individually.
        Is slower than the batched version, but saves memory as only the
        embeddings for one tuple are needed. Useful when you have
        a really large test set
        """
        all_mrr_scores = []
        all_ap_scores = []

        for instance in tqdm.tqdm(self.samples, desc="Samples"):
            query = instance["query"]
            positive = list(instance["positive"])
            negative = list(instance["negative"])

            if len(positive) == 0 or len(negative) == 0:
                continue

            docs = positive + negative
            is_relevant = [True] * len(positive) + [False] * len(negative)

            query_emb = model.encode([query], convert_to_tensor=True, batch_size=self.batch_size)
            docs_emb = model.encode(docs, convert_to_tensor=True, batch_size=self.batch_size)

            scores = self._compute_metrics_instance(query_emb, docs_emb, is_relevant)
            all_mrr_scores.append(scores["mrr"])
            all_ap_scores.append(scores["ap"])

        mean_ap = np.mean(all_ap_scores)
        mean_mrr = np.mean(all_mrr_scores)

        return {"map": mean_ap, "mrr": mean_mrr}

    def _compute_metrics_instance(self, query_emb, docs_emb, is_relevant):
        """
        Computes metrics for a single instance = (query, positives, negatives)

        Args:
            query_emb (`torch.Tensor` of shape `(num_queries, hidden_size)`): Query embedding
                if `num_queries` > 0: we take the closest document to any of the queries
            docs_emb (`torch.Tensor` of shape `(num_pos+num_neg, hidden_size)`): Candidates documents embeddings
            is_relevant (`List[bool]` of length `num_pos+num_neg`): True if the document is relevant

        Returns:
            scores (`Dict[str, float]`):
                - `mrr`: Mean Reciprocal Rank @ `self.mrr_at_k`
                - `ap`: Average Precision
        """

        pred_scores = self.similarity_fct(query_emb, docs_emb)
        if len(pred_scores.shape) > 1:
            pred_scores = torch.amax(pred_scores, dim=0)

        pred_scores_argsort = torch.argsort(-pred_scores)  # Sort in decreasing order

        mrr = self.mrr_at_k_score(is_relevant, pred_scores_argsort, self.mrr_at_k)
        ap = self.ap_score(is_relevant, pred_scores.cpu().tolist())
        return {"mrr": mrr, "ap": ap}

    @staticmethod
    def mrr_at_k_score(is_relevant, pred_ranking, k):
        """
        Computes MRR@k score

        Args:
            is_relevant (`List[bool]` of length `num_pos+num_neg`): True if the document is relevant
            pred_ranking (`List[int]` of length `num_pos+num_neg`): Indices of the documents sorted in decreasing order
                of the similarity score

        Returns:
            mrr_score (`float`): MRR@k score
        """
        mrr_score = 0
        for rank, index in enumerate(pred_ranking[:k]):
            if is_relevant[index]:
                mrr_score = 1 / (rank + 1)
                break

        return mrr_score

    @staticmethod
    def ap_score(is_relevant, pred_scores):
        """
        Computes AP score

        Args:
            is_relevant (`List[bool]` of length `num_pos+num_neg`): True if the document is relevant
            pred_scores (`List[float]` of length `num_pos+num_neg`): Predicted similarity scores

        Returns:
            ap_score (`float`): AP score
        """
        # preds = np.array(is_relevant)[pred_scores_argsort]
        # precision_at_k = np.mean(preds[:k])
        # ap = np.mean([np.mean(preds[: k + 1]) for k in range(len(preds)) if preds[k]])
        ap = average_precision_score(is_relevant, pred_scores)
        return ap
