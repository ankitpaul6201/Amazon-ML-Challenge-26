"""
Inference & Prediction Scoring CLI.
Amazon ML Challenge 2026 — Business Entity Resolution.
Generates:
- output/candidate_pairs.tsv
- output/matching_results.tsv
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DEFAULT_TEST_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    MAX_CANDIDATES_PER_ENTITY,
)
from preprocess import load_and_preprocess_tsv
from blocking import MultiIndexBlocker
from feature_extraction import build_pair_feature_matrix
from train import EntityResolutionClassifier


def serialize_candidate_pairs(candidate_map: Dict[str, List[str]], s1_ids_order: List[str], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_ids_order:
            cands = candidate_map.get(s1_id, [])
            seen = set()
            clean_cands = []
            for c in cands:
                if c not in seen and not c.startswith("S1-"):
                    clean_cands.append(c)
                    seen.add(c)
            f.write(f"{s1_id}\t{','.join(clean_cands)}\n")


def serialize_matching_results(
    prediction_map: Dict[str, List[str]],
    s1_ids_order: List[str],
    output_path: str,
    candidate_map: Dict[str, List[str]] = None,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_ids_order:
            matches = prediction_map.get(s1_id, [])
            valid_cands = set(candidate_map.get(s1_id, [])) if candidate_map is not None else None
            seen = set()
            clean_matches = []
            for m in matches:
                if m not in seen and not m.startswith("S1-"):
                    if valid_cands is not None and m not in valid_cands:
                        continue
                    clean_matches.append(m)
                    seen.add(m)
            f.write(f"{s1_id}\t{','.join(clean_matches)}\n")


def run_inference(
    test_dir: str = DEFAULT_TEST_DIR,
    model_path: str = DEFAULT_MODEL_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    threshold: Optional[float] = None,
):
    print("=" * 70)
    print("Running Test Set Inference & Output Serialization")
    print("=" * 70)

    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    df_test_s1 = load_and_preprocess_tsv(s1_path)
    df_test_s2 = load_and_preprocess_tsv(s2_path)
    df_test_s3 = load_and_preprocess_tsv(s3_path)
    df_test_target = pd.concat([df_test_s2, df_test_s3], ignore_index=True)

    s1_order = df_test_s1["entity_id"].tolist()
    print(f"Test S1 entities: {len(s1_order)} | Countries: {df_test_s1['country'].value_counts().to_dict()}")

    # 1. Blocking
    blocker = MultiIndexBlocker(max_candidates_per_entity=MAX_CANDIDATES_PER_ENTITY)
    candidate_map = blocker.generate_candidates(df_test_s1, df_test_s2, df_test_s3)

    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    serialize_candidate_pairs(candidate_map, s1_order, cand_path)
    print(f"Saved: {cand_path}")

    # 2. Classifier Scoring
    clf = EntityResolutionClassifier.load(model_path)
    active_thresh = threshold if threshold is not None else clf.optimal_threshold
    print(f"Scoring candidates with threshold tau={active_thresh:.3f}")

    prediction_map = {s1_id: [] for s1_id in s1_order}
    features_df, pairs = build_pair_feature_matrix(df_test_s1, df_test_target, candidate_map)

    if not features_df.empty and len(pairs) > 0:
        probs = clf.predict_proba(features_df)
        grouped = {}
        for (s1_id, t_id), prob in zip(pairs, probs):
            if prob >= active_thresh:
                grouped.setdefault(s1_id, []).append((t_id, float(prob)))

        for s1_id, matches in grouped.items():
            matches.sort(key=lambda x: x[1], reverse=True)
            seen = set()
            clean = []
            for t_id, _ in matches:
                if t_id not in seen and not t_id.startswith("S1-"):
                    clean.append(t_id)
                    seen.add(t_id)
            prediction_map[s1_id] = clean

    # 3. Serialize matching results
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    serialize_matching_results(prediction_map, s1_order, matching_path, candidate_map)
    print(f"Saved: {matching_path}")

    return cand_path, matching_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run test inference and export TSVs.")
    parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    run_inference(args.test_dir, args.model_path, args.output_dir, args.threshold)
