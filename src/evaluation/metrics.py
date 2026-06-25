# # يجب أخذها من ملفات علا , وماهو موجود هنا للتجربة فقط 

# import math
# from typing import Dict, Iterable, List, Sequence


# Qrels = Dict[str, Dict[str, int]]
# Run = Dict[str, List[str]]


# def precision_at_k(
#     retrieved_doc_ids: Sequence[str],
#     relevant_doc_ids: Iterable[str],
#     k: int = 10,
# ) -> float:
#     if k <= 0:
#         return 0.0

#     retrieved_at_k = list(retrieved_doc_ids)[:k]
#     if not retrieved_at_k:
#         return 0.0

#     relevant_set = set(str(doc_id) for doc_id in relevant_doc_ids)

#     hits = sum(
#         1
#         for doc_id in retrieved_at_k
#         if str(doc_id) in relevant_set
#     )

#     return hits / k


# def recall_at_k(
#     retrieved_doc_ids: Sequence[str],
#     relevant_doc_ids: Iterable[str],
#     k: int,
# ) -> float:
#     relevant_set = set(str(doc_id) for doc_id in relevant_doc_ids)

#     if not relevant_set:
#         return 0.0

#     retrieved_at_k = list(retrieved_doc_ids)[:k]

#     hits = sum(
#         1
#         for doc_id in retrieved_at_k
#         if str(doc_id) in relevant_set
#     )

#     return hits / len(relevant_set)


# def average_precision(
#     retrieved_doc_ids: Sequence[str],
#     relevant_doc_ids: Iterable[str],
# ) -> float:
#     relevant_set = set(str(doc_id) for doc_id in relevant_doc_ids)

#     if not relevant_set:
#         return 0.0

#     hits = 0
#     precision_sum = 0.0

#     for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
#         if str(doc_id) in relevant_set:
#             hits += 1
#             precision_sum += hits / rank

#     return precision_sum / len(relevant_set)


# def mean_average_precision(
#     qrels: Qrels,
#     run: Run,
# ) -> float:
#     if not qrels:
#         return 0.0

#     ap_values = []

#     for qid, doc_relevance in qrels.items():
#         relevant_doc_ids = [
#             doc_id
#             for doc_id, relevance in doc_relevance.items()
#             if relevance > 0
#         ]

#         retrieved_doc_ids = run.get(qid, [])
#         ap_values.append(
#             average_precision(
#                 retrieved_doc_ids=retrieved_doc_ids,
#                 relevant_doc_ids=relevant_doc_ids,
#             )
#         )

#     if not ap_values:
#         return 0.0

#     return sum(ap_values) / len(ap_values)


# def dcg_at_k(
#     relevance_scores: Sequence[int],
#     k: int,
# ) -> float:
#     if k <= 0:
#         return 0.0

#     scores = list(relevance_scores)[:k]
#     dcg = 0.0

#     for index, relevance in enumerate(scores):
#         rank = index + 1
#         gain = (2 ** relevance) - 1
#         discount = math.log2(rank + 1)
#         dcg += gain / discount

#     return dcg


# def ndcg_at_k(
#     retrieved_doc_ids: Sequence[str],
#     doc_relevance: Dict[str, int],
#     k: int = 10,
# ) -> float:
#     if k <= 0:
#         return 0.0

#     retrieved_scores = [
#         int(doc_relevance.get(str(doc_id), 0))
#         for doc_id in list(retrieved_doc_ids)[:k]
#     ]

#     ideal_scores = sorted(
#         [int(score) for score in doc_relevance.values()],
#         reverse=True,
#     )[:k]

#     actual_dcg = dcg_at_k(retrieved_scores, k)
#     ideal_dcg = dcg_at_k(ideal_scores, k)

#     if ideal_dcg == 0:
#         return 0.0

#     return actual_dcg / ideal_dcg


# def evaluate_query(
#     retrieved_doc_ids: Sequence[str],
#     doc_relevance: Dict[str, int],
#     k: int = 10,
# ) -> dict:
#     relevant_doc_ids = [
#         doc_id
#         for doc_id, relevance in doc_relevance.items()
#         if relevance > 0
#     ]

#     return {
#         "precision_at_k": precision_at_k(
#             retrieved_doc_ids=retrieved_doc_ids,
#             relevant_doc_ids=relevant_doc_ids,
#             k=k,
#         ),
#         "recall_at_k": recall_at_k(
#             retrieved_doc_ids=retrieved_doc_ids,
#             relevant_doc_ids=relevant_doc_ids,
#             k=k,
#         ),
#         "average_precision": average_precision(
#             retrieved_doc_ids=retrieved_doc_ids,
#             relevant_doc_ids=relevant_doc_ids,
#         ),
#         "ndcg_at_k": ndcg_at_k(
#             retrieved_doc_ids=retrieved_doc_ids,
#             doc_relevance=doc_relevance,
#             k=k,
#         ),
#     }


# def evaluate_run(
#     qrels: Qrels,
#     run: Run,
#     k: int = 10,
# ) -> dict:
#     if not qrels:
#         return {
#             "map": 0.0,
#             f"precision_at_{k}": 0.0,
#             f"recall_at_{k}": 0.0,
#             f"ndcg_at_{k}": 0.0,
#             "num_queries": 0,
#         }

#     precision_values = []
#     recall_values = []
#     ap_values = []
#     ndcg_values = []

#     for qid, doc_relevance in qrels.items():
#         retrieved_doc_ids = run.get(qid, [])

#         query_metrics = evaluate_query(
#             retrieved_doc_ids=retrieved_doc_ids,
#             doc_relevance=doc_relevance,
#             k=k,
#         )

#         precision_values.append(query_metrics["precision_at_k"])
#         recall_values.append(query_metrics["recall_at_k"])
#         ap_values.append(query_metrics["average_precision"])
#         ndcg_values.append(query_metrics["ndcg_at_k"])

#     num_queries = len(qrels)

#     return {
#         "map": sum(ap_values) / num_queries,
#         f"precision_at_{k}": sum(precision_values) / num_queries,
#         f"recall_at_{k}": sum(recall_values) / num_queries,
#         f"ndcg_at_{k}": sum(ndcg_values) / num_queries,
#         "num_queries": num_queries,
#     }



import math
from typing import Dict, Iterable, List, Sequence


Qrels = Dict[str, Dict[str, int]]
Run = Dict[str, List[str]]


def relevant_doc_ids(
    doc_relevance: Dict[str, int],
    min_relevance: int = 1,
) -> list[str]:
    """
    Return binary-relevant document ids.

    In the TREC Genomics qrels, relevance grades can be 0, 1, or 2.
    For Precision, Recall, and Average Precision, only relevance > 0
    should be considered relevant.
    """
    return [
        str(doc_id)
        for doc_id, relevance in doc_relevance.items()
        if int(relevance) >= min_relevance
    ]


def precision_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_docs: Iterable[str],
    k: int = 10,
) -> float:
    if k <= 0:
        return 0.0

    retrieved_at_k = [str(doc_id) for doc_id in list(retrieved_doc_ids)[:k]]
    if not retrieved_at_k:
        return 0.0

    relevant_set = {str(doc_id) for doc_id in relevant_docs}
    hits = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)

    # Standard P@K divides by K, not by len(retrieved_at_k).
    return hits / k


def recall_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_docs: Iterable[str],
    k: int,
) -> float:
    if k <= 0:
        return 0.0

    relevant_set = {str(doc_id) for doc_id in relevant_docs}
    if not relevant_set:
        return 0.0

    retrieved_at_k = [str(doc_id) for doc_id in list(retrieved_doc_ids)[:k]]
    hits = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)

    return hits / len(relevant_set)


def average_precision(
    retrieved_doc_ids: Sequence[str],
    relevant_docs: Iterable[str],
) -> float:
    """
    Average Precision over the supplied ranked list.

    If the caller wants MAP@100 or MAP@1000, it should pass only the first
    100 or 1000 retrieved documents to this function.
    """
    relevant_set = {str(doc_id) for doc_id in relevant_docs}
    if not relevant_set:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        doc_id = str(doc_id)
        if doc_id in relevant_set:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / len(relevant_set)


def dcg_at_k(
    relevance_scores: Sequence[int],
    k: int,
) -> float:
    if k <= 0:
        return 0.0

    scores = list(relevance_scores)[:k]
    dcg = 0.0

    for index, relevance in enumerate(scores):
        rank = index + 1
        gain = (2 ** int(relevance)) - 1
        discount = math.log2(rank + 1)
        dcg += gain / discount

    return dcg


def ndcg_at_k(
    retrieved_doc_ids: Sequence[str],
    doc_relevance: Dict[str, int],
    k: int = 10,
) -> float:
    if k <= 0:
        return 0.0

    retrieved_scores = [
        int(doc_relevance.get(str(doc_id), 0))
        for doc_id in list(retrieved_doc_ids)[:k]
    ]

    ideal_scores = sorted(
        [int(score) for score in doc_relevance.values()],
        reverse=True,
    )[:k]

    actual_dcg = dcg_at_k(retrieved_scores, k)
    ideal_dcg = dcg_at_k(ideal_scores, k)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def evaluate_query(
    retrieved_doc_ids: Sequence[str],
    doc_relevance: Dict[str, int],
    precision_k: int = 10,
    recall_k: int = 100,
    ndcg_k: int = 10,
    map_depth: int | None = None,
    min_relevance: int = 1,
) -> dict:
    relevant_docs = relevant_doc_ids(
        doc_relevance=doc_relevance,
        min_relevance=min_relevance,
    )

    retrieved_for_ap = list(retrieved_doc_ids)
    if map_depth is not None:
        retrieved_for_ap = retrieved_for_ap[:map_depth]

    return {
        f"precision_at_{precision_k}": precision_at_k(
            retrieved_doc_ids=retrieved_doc_ids,
            relevant_docs=relevant_docs,
            k=precision_k,
        ),
        f"recall_at_{recall_k}": recall_at_k(
            retrieved_doc_ids=retrieved_doc_ids,
            relevant_docs=relevant_docs,
            k=recall_k,
        ),
        "average_precision": average_precision(
            retrieved_doc_ids=retrieved_for_ap,
            relevant_docs=relevant_docs,
        ),
        f"ndcg_at_{ndcg_k}": ndcg_at_k(
            retrieved_doc_ids=retrieved_doc_ids,
            doc_relevance=doc_relevance,
            k=ndcg_k,
        ),
        "num_relevant": len(relevant_docs),
        "num_retrieved": len(retrieved_doc_ids),
    }


def evaluate_run(
    qrels: Qrels,
    run: Run,
    k: int = 10,
    precision_k: int | None = None,
    recall_k: int | None = None,
    ndcg_k: int | None = None,
    map_depth: int | None = None,
    min_relevance: int = 1,
    include_per_query: bool = False,
) -> dict:
    """
    Evaluate a full ranked run against qrels.

    Backward-compatible behavior:
      evaluate_run(qrels, run, k=10)
    returns precision_at_10, recall_at_10, ndcg_at_10.

    Recommended project behavior:
      precision_k=10, ndcg_k=10, recall_k=run_depth, map_depth=run_depth
    so Precision/nDCG focus on top results, while Recall/MAP use deeper runs.
    """
    precision_k = precision_k if precision_k is not None else k
    recall_k = recall_k if recall_k is not None else k
    ndcg_k = ndcg_k if ndcg_k is not None else k

    if not qrels:
        empty = {
            "map": 0.0,
            f"precision_at_{precision_k}": 0.0,
            f"recall_at_{recall_k}": 0.0,
            f"ndcg_at_{ndcg_k}": 0.0,
            "num_queries": 0,
        }
        if include_per_query:
            empty["per_query"] = []
        return empty

    precision_values = []
    recall_values = []
    ap_values = []
    ndcg_values = []
    per_query_rows = []

    for qid, doc_relevance in qrels.items():
        qid = str(qid)
        retrieved_doc_ids = [str(doc_id) for doc_id in run.get(qid, [])]

        query_metrics = evaluate_query(
            retrieved_doc_ids=retrieved_doc_ids,
            doc_relevance=doc_relevance,
            precision_k=precision_k,
            recall_k=recall_k,
            ndcg_k=ndcg_k,
            map_depth=map_depth,
            min_relevance=min_relevance,
        )

        precision_values.append(query_metrics[f"precision_at_{precision_k}"])
        recall_values.append(query_metrics[f"recall_at_{recall_k}"])
        ap_values.append(query_metrics["average_precision"])
        ndcg_values.append(query_metrics[f"ndcg_at_{ndcg_k}"])

        if include_per_query:
            per_query_rows.append(
                {
                    "query_id": qid,
                    **query_metrics,
                }
            )

    num_queries = len(qrels)
    result = {
        "map": sum(ap_values) / num_queries,
        f"precision_at_{precision_k}": sum(precision_values) / num_queries,
        f"recall_at_{recall_k}": sum(recall_values) / num_queries,
        f"ndcg_at_{ndcg_k}": sum(ndcg_values) / num_queries,
        "num_queries": num_queries,
    }

    if include_per_query:
        result["per_query"] = per_query_rows

    return result
