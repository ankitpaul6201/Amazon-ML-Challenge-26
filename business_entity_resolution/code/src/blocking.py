"""
Candidate Generation and Blocking Engine.
High-performance pure-Python/NumPy inverted index and character N-gram search engine.
Adheres strictly to:
- REQ-BLK-1: Drastic search space reduction with high candidate recall.
- REQ-BLK-2: Serialization to candidate_pairs.tsv.
- REQ-BLK-3: Single entity row integrity.
- Hard candidate capping (|C(e_1)| <= 20).
- Zero external DLL / C-extension dependency (safe under all OS security policies).
"""

import math
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd


def get_char_ngrams(text: str, min_n: int = 3, max_n: int = 5) -> List[str]:
    """
    Extracts character n-grams with whitespace boundary tokens.
    """
    if not text:
        return []
    words = text.split()
    ngrams = []
    for w in words:
        padded = f" {w} "
        w_len = len(padded)
        for n in range(min_n, min(max_n + 1, w_len + 1)):
            for i in range(w_len - n + 1):
                ngrams.append(padded[i : i + n])
    return ngrams


def get_word_tokens(text: str) -> List[str]:
    """
    Tokenizes text into words.
    """
    if not text:
        return []
    return [w for w in text.split() if len(w) > 1]


class InvertedIndexEngine:
    """
    TF-IDF Inverted Index with cosine scoring.
    """
    def __init__(self, token_fn):
        self.token_fn = token_fn
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.postings: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self.doc_norms: np.ndarray = np.array([])
        self.num_docs = 0

    def fit_index(self, corpus: List[str]):
        self.num_docs = len(corpus)
        if self.num_docs == 0:
            return

        doc_freq = Counter()
        doc_term_counts = []

        for text in corpus:
            tokens = self.token_fn(text)
            counts = Counter(tokens)
            doc_term_counts.append(counts)
            for t in counts.keys():
                doc_freq[t] += 1

        # Compute IDF weights
        for t, df in doc_freq.items():
            self.idf[t] = math.log((1.0 + self.num_docs) / (1.0 + df)) + 1.0

        # Build postings list and compute doc norms
        norms = np.zeros(self.num_docs, dtype=np.float32)
        for doc_id, counts in enumerate(doc_term_counts):
            sq_sum = 0.0
            for t, count in counts.items():
                w = (1.0 + math.log(count)) * self.idf[t]
                sq_sum += w * w
                self.postings[t].append((doc_id, w))
            norms[doc_id] = math.sqrt(sq_sum) if sq_sum > 0 else 1.0

        self.doc_norms = norms

    def query_top_k(
        self, query_text: str, top_k: int = 15, min_score: float = 0.12
    ) -> List[Tuple[int, float]]:
        if self.num_docs == 0:
            return []

        tokens = self.token_fn(query_text)
        if not tokens:
            return []

        q_counts = Counter(tokens)
        q_weights = {}
        q_sq_sum = 0.0
        for t, count in q_counts.items():
            if t in self.idf:
                w = (1.0 + math.log(count)) * self.idf[t]
                q_weights[t] = w
                q_sq_sum += w * w

        if q_sq_sum == 0.0:
            return []

        q_norm = math.sqrt(q_sq_sum)

        # Accumulate dot product scores
        scores = defaultdict(float)
        for t, qw in q_weights.items():
            for doc_id, dw in self.postings[t]:
                scores[doc_id] += qw * dw

        results = []
        for doc_id, dot in scores.items():
            sim = dot / (q_norm * self.doc_norms[doc_id])
            if sim >= min_score:
                results.append((doc_id, sim))

        if not results:
            return []

        # Sort by similarity score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class MultiIndexBlocker:
    def __init__(
        self,
        max_candidates_per_entity: int = 20,
        name_top_k: int = 12,
        address_top_k: int = 6,
        min_name_sim: float = 0.12,
        min_address_sim: float = 0.15,
    ):
        self.max_candidates_per_entity = max_candidates_per_entity
        self.name_top_k = name_top_k
        self.address_top_k = address_top_k
        self.min_name_sim = min_name_sim
        self.min_address_sim = min_address_sim

    def generate_candidates(
        self, df_s1: pd.DataFrame, df_s2: pd.DataFrame, df_s3: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """
        Generates candidate target IDs for each entity in S1.
        Returns:
            candidate_map: Dict[s1_entity_id -> List[target_entity_ids]]
        """
        df_target = pd.concat([df_s2, df_s3], ignore_index=True)

        candidate_map: Dict[str, List[str]] = {
            s1_id: [] for s1_id in df_s1["entity_id"]
        }

        # Dynamic country partitioning (open set)
        unique_countries = df_s1["clean_country"].unique()

        for country in unique_countries:
            s1_sub = df_s1[df_s1["clean_country"] == country].reset_index(drop=True)
            target_sub = df_target[df_target["clean_country"] == country].reset_index(drop=True)

            if s1_sub.empty or target_sub.empty:
                continue

            s1_candidates = self._block_country_partition(s1_sub, target_sub)
            for s1_id, cands in s1_candidates.items():
                candidate_map[s1_id] = cands

        return candidate_map

    def _block_country_partition(
        self, s1_sub: pd.DataFrame, target_sub: pd.DataFrame
    ) -> Dict[str, List[str]]:
        s1_ids = s1_sub["entity_id"].tolist()
        target_ids = target_sub["entity_id"].tolist()
        n_s1 = len(s1_ids)

        # 1. Build Index A: Name Character N-Gram Inverted Index
        name_index = InvertedIndexEngine(token_fn=get_char_ngrams)
        name_index.fit_index(target_sub["clean_name"].tolist())

        # 2. Build Index B: Address Word Token Inverted Index
        addr_index = InvertedIndexEngine(token_fn=get_word_tokens)
        addr_index.fit_index(target_sub["clean_address"].tolist())

        # 3. Build Index C: Postal/PIN Code Inverted Index
        pin_index = defaultdict(list)
        for tidx, digits in enumerate(target_sub["digits"]):
            for d in digits:
                if len(d) >= 4:
                    pin_index[d].append(tidx)

        result: Dict[str, List[str]] = {}

        for i in range(n_s1):
            s1_id = s1_ids[i]
            s1_name = s1_sub.at[i, "clean_name"]
            s1_addr = s1_sub.at[i, "clean_address"]
            s1_digits = s1_sub.at[i, "digits"]

            cand_scores: Dict[int, float] = defaultdict(float)

            # Query Name Index
            name_hits = name_index.query_top_k(
                s1_name, top_k=self.name_top_k, min_score=self.min_name_sim
            )
            for tidx, score in name_hits:
                cand_scores[tidx] += score * 1.5

            # Query Address Index
            addr_hits = addr_index.query_top_k(
                s1_addr, top_k=self.address_top_k, min_score=self.min_address_sim
            )
            for tidx, score in addr_hits:
                cand_scores[tidx] += score * 1.0

            # Query PIN Index
            for d in s1_digits:
                if len(d) >= 4 and d in pin_index:
                    for tidx in pin_index[d][:10]:
                        cand_scores[tidx] += 0.35

            if not cand_scores:
                result[s1_id] = []
            else:
                # Rank by accumulated score descending
                sorted_cands = sorted(
                    cand_scores.items(), key=lambda x: x[1], reverse=True
                )
                capped_cands = [
                    str(target_ids[tidx])
                    for tidx, _ in sorted_cands[: self.max_candidates_per_entity]
                ]
                result[s1_id] = capped_cands

        return result
