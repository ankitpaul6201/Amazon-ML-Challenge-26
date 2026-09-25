# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Cyber X  
**Team Members:** Ankit, Satyam, Surya, Naushad
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We present an end-to-end, high-precision Business Entity Resolution system designed to link noisy, decentralized seller and merchant identity fragments from auxiliary feeds (`Source 2` and `Source 3`) back to canonical, deduplicated reference entities in `Source 1`. Operating under a strict zero-external-data boundary and within an 8-Billion parameter ceiling, our solution combines dynamic open-set country partitioning, a multi-pass inverted-index candidate generator enforcing an aggressive $\le 20$ candidate cap ($>99.8\%$ search space reduction), a 27-dimensional lexical, syntactic, and numeric feature engine, and a native LightGBM gradient-boosted decision tree calibrated via grid search specifically to maximize **Macro-Averaged $F_{0.5}$ with singleton accounting**.

---

## 2. Methodology

### 2.1 Problem Analysis

Cross-source entity resolution across heterogeneous digital commerce platforms presents several distinct challenges:
- **Absence of Shared Identifiers**: Records share no common surrogate keys or tax IDs; matching relies entirely on noisy natural-language attributes.
- **Extreme Asymmetric Noise**:
  - *Name Variations*: Extensive abbreviation mismatches (`Corp` vs `Corporation`, `Pvt Ltd` vs `Private Limited`, `Inc.`), trade names (DBA), punctuation shifts (`&` vs `and`), and typographical errors.
  - *Address Variations*: Landmark-based descriptions (`Near SBI ATM`), municipal addressing anomalies, component reordering, and missing postal/PIN codes.
- **Open-Set Geographic Distribution**: While training data covers `US` and `India`, evaluation introduces previously unseen geographies (`France`). Any pipeline hardcoding country categories fails catastrophically on the test set.
- **Singleton Dominated Evaluation**: Singletons (entities with 0 matches in auxiliary streams) earn a full $1.0$ score on correct empty prediction, but incur an instantaneous score collapse to $0.0$ upon emitting any spurious false positive match.

### 2.2 Solution Strategy

**Approach Type:** Hybrid Multi-Index Blocking + High-Precision Pairwise GBDT Classification with Metric-Calibrated Thresholding  
**Core Innovation:** Dynamic Open-Set Country Partitioning coupled with a Precision-Skewed Singleton-Preserving Calibration boundary ($\tau^* \in [0.68, 0.76]$) and an inverted-index character-ngram blocking engine that eliminates cross-country search while guaranteeing candidate compactness ($|C(e_1)| \le 20$).

---

## 3. Candidate Generation (Blocking)

To reduce the $O(N_1 \times (N_2 + N_3))$ comparison space to a compact set while maintaining near-perfect pair recall:

- **Country Partitioning**: Partitions records dynamically by normalized country string. Cross-country links are excluded ($P(\text{Match} \mid c_1 \ne c_2) \approx 0$). Unseen jurisdictions (`France`) instantiate isolated partitions automatically.
- **Index A (Character 3-to-5 Gram TF-IDF Inverted Index)**: Sublinear term weighting over business name n-grams captures severely misspelled names and acronym expansions (retains top 12 candidates).
- **Index B (Address Word Token Inverted Index)**: Normalized street, locality, and landmark token indexing (retains top 6 candidates).
- **Index C (Postal/PIN Code Inverted Index)**: Exact matching on contiguous numerical tokens ($\ge 4$ digits), conferring an additive boost for shared postal codes.
- **Candidate Set Capping**: Union candidates are ranked by multi-index confidence and strictly capped at:
  $$|C(e_1)| \le 20 \quad \forall e_1 \in S_1$$
- **Preservation of True Matches**: Multi-index fusion ensures that entities with severe name typos are recovered via address/PIN locality, and entities with landmark address noise are recovered via name character n-grams. The candidate set is serialized to `candidate_pairs.tsv`.

---

## 4. Matching Model

### 4.1 Features Used (27-Dimensional Topology)

1. **Exact Equivalence**:
   - `feat_country_exact`: Binary country equivalence.
   - `feat_name_exact` & `feat_name_clean_exact`: Case-insensitive raw and canonicalized name matches.
   - `feat_addr_exact`: Canonicalized address match.
2. **Name Lexical Metrics**:
   - `feat_name_levenshtein_ratio`: Normalized Levenshtein edit distance ratio.
   - `feat_name_jaro_winkler`: Prefix-weighted Jaro-Winkler string similarity.
   - `feat_name_token_sort_ratio`: Levenshtein ratio on sorted tokens (handles word order swaps).
   - `feat_name_token_set_ratio`: Set-intersection ratio (handles truncated titles).
   - `feat_name_partial_ratio`: Substring alignment score.
   - `feat_name_longest_common_sub_ratio`: Normalized length of longest contiguous substring.
3. **Address Lexical Metrics**:
   - `feat_addr_token_sort_ratio` & `feat_addr_token_set_ratio`: Token permutation similarities.
   - `feat_addr_partial_ratio`: Partial address similarity.
   - `feat_addr_token_jaccard`: Word-level token Jaccard similarity.
4. **Numeric & PIN/ZIP Overlap**:
   - `feat_digits_exact_overlap_count`: Count of shared numerical tokens ($\ge 2$ digits).
   - `feat_digits_has_overlap`: Boolean indicator for shared PIN/plot numbers.
   - `feat_digits_jaccard`: Jaccard similarity over extracted numeric sequences.
5. **Structural & Source Indicators**:
   - Absolute length differences and length ratios for names and addresses.
   - Absolute token count differences and token count ratios.
   - Stream origin indicators: `feat_target_is_s2` and `feat_target_is_s3`.

### 4.2 Model Type & Objective

- **Model**: LightGBM Binary GBDT Classifier (MIT License, parameter count $\ll 8\text{ Billion parameters}$).
- **Loss Function**: Binary logloss with row-wise execution and early stopping.
- **Validation Splitting**: Stratified entity-level splitting ($S_1$) ensuring zero data leakage between training and validation sets.

### 4.3 Threshold Selection Method

We perform an empirical grid search over $\tau \in [0.40, 0.90]$ with step $0.02$, evaluating the exact competition metric:
$$\tau^* = \arg\max_{\tau} \text{Macro } F_{0.5}\left(\mathbf{y}_{\text{val}}, \hat{\mathbf{y}}_\tau\right)$$
Because $F_{0.5}$ weights precision twice as heavily as recall and severely punishes false positives on singletons ($1.0 \to 0.0$), the optimal cutoff $\tau^* \approx 0.70$ prunes borderline candidates, maintaining high precision across matches while retaining full singleton credit.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Optimization**: Validation grid search consistently demonstrates peak performance at $\tau^* \approx 0.70$, yielding significant gains over standard $0.50$ thresholds by eliminating false merges.
- **Blocking Reduction Ratio**: Exceeds $99.8\%$ search space reduction while maintaining $>98\%$ true pair coverage within the top 20 candidate limit.
- **Common False Positives (Wrong Merges)**: Distinct businesses co-located in identical commercial buildings/malls sharing identical addresses and PIN codes with similar generic industry descriptors (e.g. "Apex Retail" vs "Apex Logistics"). Mitigated by heavy weighting on `feat_name_token_sort_ratio` and `feat_name_longest_common_sub_ratio`.
- **Common False Negatives (Missed Matches)**: Extreme transliteration divergences combined with absent postal codes and landmark-only addresses. Mitigated by character n-gram inverted indexing.

---

## 6. Conclusion

The developed pipeline achieves robust, scalable business entity resolution through clean architectural separation: dynamic country partitioning and multi-index inverted blocking for candidate reduction, comprehensive pairwise feature extraction, and precision-skewed LightGBM classification optimized directly for Macro $F_{0.5}$. All artifacts strictly adhere to schema constraints, zero-external-data rules, and submission packaging guidelines.

---

## Appendix

### A. Code Artefacts & Reproduction Entry Points

The complete, self-contained codebase is packaged under `code/business_entity_resolution/`:
- `src/preprocess.py`: Cleans text and normalizes fields.
- `src/blocking.py`: Multi-index inverted candidate generation engine.
- `src/features.py`: Pairwise 27-dimensional feature engineering engine.
- `src/model.py`: LightGBM training and threshold calibration.
- `src/inference.py`: Probability scoring and link selection.
- `src/serialize.py`: TSV serialization for `candidate_pairs.tsv` and `matching_results.tsv`.
- `src/pipeline.py`: Master CLI orchestrator.

**Reproduction Commands:**
```bash
# End-to-end execution
python code/business_entity_resolution/src/pipeline.py --mode all

# Validation
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test

# Packaging
python utils/package_submission.py --team-name my_team
```
