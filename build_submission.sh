#!/usr/bin/env bash
set -e

TEAM_NAME="${1:-cyber_x}"

echo "=== 1. Running official submission validation ==="
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

echo "=== 2. Packaging archive strictly to competition specification ==="
zip -r "${TEAM_NAME}_submission.zip" \
    output/matching_results.tsv \
    output/candidate_pairs.tsv \
    business_entity_resolution/

echo "Successfully built ${TEAM_NAME}_submission.zip"
