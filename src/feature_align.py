from __future__ import annotations
from typing import Iterable, Optional
import pandas as pd
import numpy as np


def _find_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in cols:
            return cols[key]
    # try relaxed contains match
    for key_norm, orig in cols.items():
        for cand in candidates:
            cand_norm = str(cand).strip().lower()
            if cand_norm in key_norm:
                return orig
    return None


def _to_num(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    return x


def to_harmonized(df: pd.DataFrame) -> pd.DataFrame:
    """Build a small, harmonized numeric feature set present (or derivable) in both CICIDS and UNSW.

    Output columns:
      - dur
      - pkts_fwd, pkts_bwd, pkts_total, pkts_ratio
      - bytes_fwd, bytes_bwd, bytes_total, bytes_ratio
      - pkt_size_mean
    """
    out = pd.DataFrame(index=df.index)

    # duration
    c_dur = _find_col(df, ["dur", "flow duration", "duration"])
    if c_dur is not None:
        out["dur"] = _to_num(df[c_dur])
    else:
        out["dur"] = np.nan

    # packets fwd/bwd
    c_pf = _find_col(df, ["spkts", "total fwd packets", "tot fwd pkts", "fwd pkts"])
    c_pb = _find_col(df, ["dpkts", "total backward packets", "tot bwd pkts", "bwd pkts"])
    out["pkts_fwd"] = _to_num(df[c_pf]) if c_pf is not None else np.nan
    out["pkts_bwd"] = _to_num(df[c_pb]) if c_pb is not None else np.nan
    out["pkts_total"] = out[["pkts_fwd", "pkts_bwd"]].sum(axis=1, min_count=1)
    out["pkts_ratio"] = out["pkts_fwd"] / out["pkts_bwd"].replace(0, np.nan)

    # bytes fwd/bwd
    c_bf = _find_col(df, ["sbytes", "total length of fwd packets", "fwd bytes"])
    c_bb = _find_col(df, ["dbytes", "total length of bwd packets", "bwd bytes"])
    out["bytes_fwd"] = _to_num(df[c_bf]) if c_bf is not None else np.nan
    out["bytes_bwd"] = _to_num(df[c_bb]) if c_bb is not None else np.nan
    out["bytes_total"] = out[["bytes_fwd", "bytes_bwd"]].sum(axis=1, min_count=1)
    out["bytes_ratio"] = out["bytes_fwd"] / out["bytes_bwd"].replace(0, np.nan)

    # mean packet size
    with np.errstate(invalid="ignore", divide="ignore"):
        out["pkt_size_mean"] = out["bytes_total"] / out["pkts_total"]

    return out

