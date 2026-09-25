# Business Entity Resolution — Reproduction Guide
## Amazon ML Challenge 2026 (Team Cyber X)

This package contains the self-contained, reproducible source code for the Business Entity Resolution solution.

---

## 1. Quickstart

### Environment Setup
Python 3.10+ is supported:
```bash
pip install -r requirements.txt
```

### End-to-End Execution
Run full training, threshold tuning, inference, and serialization:
```bash
python src/train.py --train-dir ../../dataset/train --model-out ../../models/lgb_model.joblib
python src/inference.py --test-dir ../../dataset/test --model-path ../../models/lgb_model.joblib --output-dir ../../output
```

Or run end-to-end via:
```bash
python src/pipeline.py --mode all
```

---

## 2. Directory Layout

```text
src/
├── __init__.py
├── config.py               # Constants, random seeds, and hyperparameters
├── preprocess.py           # Text cleaning, legal suffixes, and open-geography handler
├── blocking.py             # Multi-index inverted index candidate generator
├── feature_extraction.py   # Pairwise 27-dimensional feature engineering engine
├── train.py                # LightGBM GBDT training and early stopping
├── evaluate.py             # Entity-level Macro F_0.5 evaluation logic
└── inference.py            # Serializes matching_results.tsv and candidate_pairs.tsv
```

---

## 3. Submission Verification
```bash
python3 ../../utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test
```
