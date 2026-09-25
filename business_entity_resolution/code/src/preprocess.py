"""
Preprocessing and Text Canonicalization Engine.
Adheres to:
- REQ-IN-1: Strict TSV parsing with sep="\t".
- REQ-IN-2: Open geography handling (supports US, India, France, and arbitrary strings).
- REQ-IN-3: Text normalization (legal suffixes, address abbreviations, punctuation).
"""

import re
import pandas as pd
from typing import Dict, List, Optional, Tuple


LEGAL_SUFFIX_MAP = {
    r"\bcorp\.?\b": "corporation",
    r"\bco\.?\b": "company",
    r"\binc\.?\b": "incorporated",
    r"\bltd\.?\b": "limited",
    r"\bpvt\.?\b": "private",
    r"\bllc\.?\b": "llc",
    r"\bsarl\.?\b": "sarl",
    r"\bsas\.?\b": "sas",
    r"\bsa\.?\b": "sa",
    r"\bgmbh\.?\b": "gmbh",
    r"\bplc\.?\b": "plc",
    r"\bpvt\s+ltd\.?\b": "private limited",
    r"\bprivate\s+limited\b": "private limited",
}

ADDRESS_ABBREV_MAP = {
    r"\brd\.?\b": "road",
    r"\bst\.?\b": "street",
    r"\bave\.?\b": "avenue",
    r"\bblvd\.?\b": "boulevard",
    r"\bln\.?\b": "lane",
    r"\bfl\.?\b": "floor",
    r"\bste\.?\b": "suite",
    r"\bbldg\.?\b": "building",
    r"\bmarg\b": "road",
    r"\bopp\.?\b": "opposite",
    r"\bnr\.?\b": "near",
    r"\bsec\.?\b": "sector",
    r"\bpl\.?\b": "place",
    r"\bpkwy\.?\b": "parkway",
}


def clean_text_basic(text: Optional[str]) -> str:
    """
    Base string cleaning:
    - Null safe
    - Lowercase
    - Replace '&' with 'and'
    - Normalize punctuation and whitespace
    """
    if text is None or pd.isna(text):
        return ""
    s = str(text).lower()
    s = s.replace("&", " and ")
    s = s.replace("/", " ")
    s = s.replace("-", " ")
    s = s.replace(".", " ")
    s = s.replace(",", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_business_name(name: Optional[str]) -> str:
    """
    Normalizes business name by expanding/stripping legal suffixes.
    """
    s = clean_text_basic(name)
    for pattern, replacement in LEGAL_SUFFIX_MAP.items():
        s = re.sub(pattern, replacement, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_address(address: Optional[str]) -> str:
    """
    Normalizes business address components and abbreviations.
    """
    s = clean_text_basic(address)
    for pattern, replacement in ADDRESS_ABBREV_MAP.items():
        s = re.sub(pattern, replacement, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_address_digits(address: Optional[str]) -> List[str]:
    """
    Extracts numerical sequences (PIN codes, ZIP codes, plot/door numbers).
    """
    if address is None or pd.isna(address):
        return []
    s = str(address)
    # Match sequences of 2 or more digits
    digits = re.findall(r"\b\d{2,8}\b", s)
    return digits


def load_and_preprocess_tsv(file_path: str) -> pd.DataFrame:
    """
    Loads TSV file with strict sep="\t", performs schema checks and adds cleaned columns.
    """
    df = pd.read_csv(file_path, sep="\t", dtype=str, keep_default_na=False)

    required_cols = ["entity_id", "business_name", "business_address", "country"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from {file_path}")

    # Canonicalize text columns
    df["clean_name"] = df["business_name"].apply(normalize_business_name)
    df["clean_address"] = df["business_address"].apply(normalize_address)
    # Open-set country normalization (just strip & lowercase, do NOT filter or map to fixed set)
    df["clean_country"] = df["country"].fillna("").astype(str).str.strip().str.lower()
    df["digits"] = df["business_address"].apply(extract_address_digits)

    return df
