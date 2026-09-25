"""
Business Entity Resolution System — Amazon ML Challenge 2026.
Modules:
- config: System hyperparameters, paths, and seeds.
- preprocess: Canonicalization, legal suffix stripping, and text cleaning.
- blocking: Multi-index country-partitioned candidate generator.
- feature_extraction: Pairwise lexical, numeric, token, and semantic features.
- train: LightGBM training engine with cross-validation.
- evaluate: Macro F_0.5 evaluation metric with singleton accounting.
- inference: Prediction scoring and TSV serialization.
"""

__version__ = "1.0.0"
