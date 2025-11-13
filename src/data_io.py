# =============================================
# FILE: src/data_io.py
# =============================================
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd


LABEL_CANDIDATES = ["label", "class", "attack", "y"]
ATTACK_CAT_CANDIDATES = ["attack_cat", "attack_type", "category"]


DROP_LIKE = re.compile(
    r"(Flow|flow).*id|^id$|timestamp|time_start|time_end|"
    r"src.*ip|dst.*ip|source.*ip|destination.*ip|"
    r"source.*address|destination.*address|"
    r"label_str",
    flags=re.IGNORECASE,
)


BENIGN_STR = {"BENIGN", "Normal", "Benign", "BENIGN ", ""}




def detect_label_columns(df: pd.DataFrame) -> Tuple[str | None, str | None]:
    """Detect label and attack-category columns robustly.

    We strip whitespace and compare case-insensitively. Also allow substring matches
    (e.g. column name ' Label ' or 'attack_cat'). Returns original column names
    as found in df.columns or (None, None) if not found.
    """
    cols = list(df.columns)
    # create normalized map: normalized -> original
    norm_map = {c.strip().lower(): c for c in cols}

    lbl = None
    atac = None

    # direct normalized match
    for cand in LABEL_CANDIDATES:
        if cand in norm_map:
            lbl = norm_map[cand]
            break

    for cand in ATTACK_CAT_CANDIDATES:
        if cand in norm_map:
            atac = norm_map[cand]
            break

    # fallback: substring search in normalized names
    if lbl is None:
        for n, orig in norm_map.items():
            if any(cand in n for cand in LABEL_CANDIDATES):
                lbl = orig
                break

    if atac is None:
        for n, orig in norm_map.items():
            if any(cand in n for cand in ATTACK_CAT_CANDIDATES):
                atac = orig
                break

    return lbl, atac




def to_binary_labels(df: pd.DataFrame, label_col: str | None, attack_cat_col: str | None) -> pd.Series:
    y = None
    if label_col is not None:
        # common cases: numeric 0/1; or strings BENIGN/ATTACK
        s = df[label_col]
        if s.dtype == "O":
            y = (~s.astype(str).str.strip().isin(BENIGN_STR)).astype(int)
        else:
            # assume 1 = attack, 0 = benign
            y = (s.astype(float) > 0.5).astype(int)
    elif attack_cat_col is not None:
        s = df[attack_cat_col].astype(str).str.strip()
        y = (~s.str.lower().isin({"", "nan", "none", "benign", "normal"})).astype(int)
    else:
        raise ValueError("No label column found.")
    return y




def drop_non_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_keep = [c for c in df.columns if not DROP_LIKE.search(c)]
    return df[cols_keep]




def load_csv(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    # Many CICIDS files use non-UTF8 encodings; read with latin1 and disable low_memory to avoid mixed-type issues
    df = pd.read_csv(path, encoding='latin1', low_memory=False)
    # Normalize potential BOM/misaligned header artifacts (e.g. 'ï»¿id')
    def _clean_header(s: str) -> str:
        s = str(s)
        s = re.sub(r'^[\ufeff\uFEFF]+', '', s)  # Unicode BOM
        s = re.sub(r'^ï»¿+', '', s)              # BOM bytes decoded as latin1
        return s.strip()
    df.rename(columns={c: _clean_header(c) for c in df.columns}, inplace=True)
    # sanitize: replace infinite values with NaN to avoid downstream failures
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    lbl, atac = detect_label_columns(df)
    y = to_binary_labels(df, lbl, atac)
    X = drop_non_feature_columns(df.drop(columns=[c for c in [lbl, atac] if c and c in df.columns], errors="ignore"))
    return X, y




def cicids_day_files(root: Path) -> Dict[str, Path]:
    r = root / "CICIDS2017"
    return {
        "monday": r / "Monday-WorkingHours.pcap_ISCX.csv",
        "tuesday": r / "Tuesday-WorkingHours.pcap_ISCX.csv",
        "wednesday": r / "Wednesday-workingHours.pcap_ISCX.csv",
        "thursday_web": r / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "thursday_inf": r / "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        "friday_port": r / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "friday_ddos": r / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    }




def unsw_files(root: Path) -> Dict[str, Path]:
    r = root / "UNSW_NB15"
    return {
        "train": r / "UNSW_NB15_training-set.csv",
        "test": r / "UNSW_NB15_testing-set.csv",
    }
