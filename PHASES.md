# Implementation Phases: Amazon ML Challenge 2026 (Entity Resolution)

## Phase 0: Environment Setup & Exploration
- **Target Completion**: 25th September, 04:00 AM IST
- **Tasks**:
  1. Initialize Git repository with directory structure (`output/`, `code/business_entity_resolution/src/`).
  2. Pin Python dependencies in `requirements.txt` (ensure lightgbm, rapidfuzz, scikit-learn, pyarrow, pandas).
  3. Validate tab-separated parsing using `sep="\t"` across all dataset splits.
  4. Perform EDA on country distributions (US, India in train; prepare open-set handling for France in test).
  5. Build a local stratified validation split replicating the competition evaluation metric ($F_{0.5}$).

## Phase 1: Candidate Generation & Blocking Pipeline
- **Target Completion**: 25th September, 02:00 PM IST
- **Tasks**:
  1. Implement country-level partitioning (ensure non-existent countries do not break indexing).
  2. Implement text cleaning: legal suffix removal (Corp, Inc, Ltd, Pvt), punctuation normalization, and whitespace stripping.
  3. Build multi-pass candidate blocking indexes:
     - Inverted index on token 3-grams of normalized business names.
     - MinHash / LSH (Locality Sensitive Hashing) or TF-IDF Cosine top-$K$ candidates.
     - Postal code / locality token filtering for address candidates.
  4. Optimize candidate set size: maximize pair recall while keeping candidate count per Source 1 entity small (evaluated on reduction ratio).
  5. Export and validate the first `candidate_pairs.tsv` using `utils/validate_submission.py`.

## Phase 2: Feature Engineering & Dataset Assembly
- **Target Completion**: 26th September, 01:00 AM IST
- **Tasks**:
  1. Generate pairwise lexical features for candidate pairs:
     - Name metrics: Levenshtein distance, Token Sort ratio, Jaro-Winkler, Monge-Elkan.
     - Address metrics: Token set ratio, digit/PIN code exact match flag, street-token overlap.
  2. Generate semantic and structural features:
     - TF-IDF character and word n-gram cosine similarities.
     - Source origin indicators (`S2` vs `S3`).
  3. Assemble labeled training feature matrix using `train_ground_truth.tsv` (positives vs negatives selected from blocking candidates).

## Phase 3: Model Training, Tuning & Threshold Calibration
- **Target Completion**: 26th September, 02:00 PM IST
- **Tasks**:
  1. Train LightGBM / CatBoost binary classifier optimizing for logloss / average precision (under MIT/Apache 2.0 license limits).
  2. Perform cross-validation and verify calibration curves.
  3. Optimize entity-level classification threshold specifically for macro $F_{0.5}$.
  4. Tune singleton prediction policy: aggressively prune borderline matches to avoid 0.0 penalty on true singletons.

## Phase 4: Full Pipeline Inference & Format Validation
- **Target Completion**: 27th September, 12:00 PM IST
- **Tasks**:
  1. Run blocking pipeline on `test_source1.tsv`, `test_source2.tsv`, and `test_source3.tsv` to produce final `output/candidate_pairs.tsv`.
  2. Run inference model on all generated candidates to produce probability scores.
  3. Apply optimized precision threshold to emit `output/matching_results.tsv`.
  4. Run `python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`.
  5. Verify that every test entity appears exactly once and that matching IDs form a strict subset of candidate pairs.

## Phase 5: Submission Packaging & Documentation
- **Target Completion**: 27th September, 08:00 PM IST
- **Tasks**:
  1. Complete methodology documentation inside `Documentation_template.md` (detailing blocking, features, and model).
  2. Verify all source code under `code/business_entity_resolution/src/` runs deterministically.
  3. Package final bundle `<team_name>_submission.zip` matching required structure.
  4. Final leaderboard and artifact submission before 11:59 PM IST.
