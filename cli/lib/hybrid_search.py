import os

from inverted_index import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from cli.utils import file, cache_index
from test_gemini import gemini_enhancer
from sentence_transformers import CrossEncoder
import json, time

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(cache_index):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit):
        BM25_results, SM_results = results_getter(self, query, limit, "")
        BM25_normalization = []
        SM_normalization= []
        [BM25_normalization.append(v) for _,v in BM25_results.items()]
        [SM_normalization.append(val["score"]) for val in SM_results]
        BM25_normalized = normalize_command(BM25_normalization)
        SM25_normalized = normalize_command(SM_normalization)
        id_to_doc_scores_map = {}
        index = 0
        for i, _ in BM25_results.items():
            id_to_doc_scores_map[i] = {
                "document": self.semantic_search.document_map[i],
                "BM25": BM25_normalized[index],
                "SM": 0.0,
            }
            index += 1
        index = 0
        for i, d in enumerate(SM_results):
            if d['id'] in id_to_doc_scores_map:
                id_to_doc_scores_map[d['id']]["SM"] =  SM25_normalized[index]
            else:
                id_to_doc_scores_map[d['id']] = {
                    "document": self.semantic_search.document_map[i],
                    "BM25": 0.0,
                    "SM": SM25_normalized[index],
                }
            index += 1
        for k, val in id_to_doc_scores_map.items():
            hybrid_s= hybrid_score(val["BM25"], val["SM"], alpha)           
            id_to_doc_scores_map[k]["hybrid_score"] = hybrid_s
        
        sorted_results = sorted(id_to_doc_scores_map.items(), key = lambda x: x[1]["hybrid_score"], reverse=True)
        return sorted_results[:limit]

    def rrf_search(self, query, k, limit, rerank: str):
        BM25_results, SM_results = results_getter(self, query, limit, rerank)
        BM25_results = sorted(BM25_results.items(), key = lambda x: x[1], reverse=True)
        SM_results.sort(key= lambda x: x['score'], reverse = True)
        rank_map =  {}
        for index, val  in enumerate(BM25_results, 1):
            rank_map[val[0]] = {
                'document': self.semantic_search.document_map[val[0]],
                'BM25_rank': index,
                'SM_rank': 0, 
            }
        for index, val in enumerate(SM_results, 1):
            if val["id"] in rank_map:
                rank_map[val["id"]]["SM_rank"] = index
            else:
                rank_map[val["id"]] = {
                    'document': self.semantic_search.document_map[val["id"]],
                    'BM25_rank': 0,
                    'SM_rank': index,
                }
        for doc, val in rank_map.items():
            if val["BM25_rank"] > 0 and val["SM_rank"] > 0:
                rrf_score =  (1/(k + val["BM25_rank"])) + (1/(k + val["SM_rank"]))
                rank_map[doc]["rrf_score"] = rrf_score
            elif val["BM25_rank"] == 0 and val["SM_rank"] >= 0:
                rrf_score =  1/(k + val["SM_rank"])
                rank_map[doc]["rrf_score"] = rrf_score
            elif val["BM25_rank"] >= 0 and val["SM_rank"] == 0:
                rrf_score =  1/(k + val["BM25_rank"])
                rank_map[doc]["rrf_score"] = rrf_score
        
        results = sorted(rank_map.items(), key= lambda x: x[1]["rrf_score"], reverse=True)

        return results[:limit]

def normalize_command(l: list[float]) -> list[float]:
    if len(l) == 0:
        return []
    if min(l) == max(l):
        print(1.0)
        return []
    min_val = min(l)
    max_val = max(l)
    normalized_score = []
    for score in l:
        new_score= (score - min_val) / (max_val - min_val)
        normalized_score.append(new_score)
    return normalized_score


def hybrid_score(bm25_score, semantic_score, alpha):
    return alpha * bm25_score + (1 - alpha) *semantic_score


def weighted_search_command(query: str, alpha: float, limit: int):
    with open(file, "r") as f:
        movies = json.load(f)
    hybrid_object = HybridSearch(movies["movies"])
    hybrid_object.semantic_search.load_or_create_embeddings(movies["movies"])
    results = hybrid_object.weighted_search(query, alpha, limit)

    for index, r in enumerate(results, 1):
        print(f"{index}. {r[1]['document']['title']}")
        print(f"         Hybrid Score: {r[1]['hybrid_score']:.3f}")
        print(f"         BM25: {r[1]['BM25']:.3f}, Semantic: {r[1]['SM']:.3f}")
        print(f"         {r[1]['document']['description'][:limit]}")

def rrf_search_command(query: str, k: int, limit: int, enhance: str, rerank: str):
    if enhance:
        query = gemini_enhancer(query, enhance, "")
    with open(file, "r") as f:
        movies = json.load(f)
    hybrid_object = HybridSearch(movies["movies"])
    results = hybrid_object.rrf_search(query, k, limit, rerank)
    print(f"Re-ranking top {limit} results using individual method...")
    print(f"Reciprocal Rank Fusion Results for '{query}' (k=60)")
    match rerank:
        case "individual":
            for index, r in enumerate(results, 1):
                new_rank = gemini_enhancer(query, rerank, r[1]["document"])
                r[1]["rerank"] = int(new_rank)
                time.sleep(3)
            results = sorted(results, key= lambda x: x[1]["rerank"], reverse=True)
        case "batch":
            new_ranks = json.loads(gemini_enhancer(query, rerank, results))
            for result in results:
                result[1]["rerank"] = new_ranks.index(result[0]) + 1
            results = sorted(results, key= lambda x: x[1]["rerank"])
        case "cross_encoder":
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
            pairs = []
            for r in results:
                pairs.append([query, f"{r[1]["document"]["title"]} - {r[1]["document"]["description"]}"])
            scores = cross_encoder.predict(pairs)
            for index, r in enumerate(results):
                r[1]["cross_encoder_score"] = scores[index]
            results = sorted(results, key= lambda x: x[1]["cross_encoder_score"], reverse=True)
            
    
    for index , result in enumerate(results[:limit], 1):
        print(f"{index}. {result[1]['document']['title']}")
        print(f"         Cross Encoder Score: {result[1]['cross_encoder_score']:.3f}")
        print(f"         RRF Score: {result[1]['rrf_score']:.3f}")
        print(f"         BM25 rank: {result[1]['BM25_rank']}, Semantic rank: {result[1]['SM_rank']}")
        print(f"         {result[1]['document']['description'][:100]}...")



def results_getter(h: HybridSearch, query: str, limit: int, rerank: str):
    if rerank:
        BM25_results = h._bm25_search(query, limit * 5)
        SM_results = h.semantic_search.search_chunks(query, limit * 5)
    else:
        BM25_results = h._bm25_search(query, limit * 500)
        SM_results = h.semantic_search.search_chunks(query, limit * 500)
    return BM25_results, SM_results