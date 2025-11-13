from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import unicodedata, re

from src.plots import feature_importance_heatmap, feature_importance_heatmap_overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fi', required=True, help='Path to table_unsw_feature_importance.csv')
    ap.add_argument('--out', required=True, help='Output root (e.g., out/unsw)')
    ap.add_argument('--top-n', type=int, default=20)
    args = ap.parse_args()

    fi_path = Path(args.fi)
    out_root = Path(args.out)
    tables_dir = out_root / 'tables'
    figs_dir = out_root / 'figures'
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(fi_path)
    # sanitize feature names for display
    def _sanitize(n: str) -> str:
        s = unicodedata.normalize('NFKD', str(n))
        s = ''.join(ch for ch in s if ch.isprintable())
        s = re.sub(r"\s+", "_", s.strip())
        s = re.sub(r"[^0-9A-Za-z_./-]", "_", s)
        s = re.sub(r"_+", "_", s)
        return s
    if 'feature' in df.columns:
        df['feature'] = df['feature'].map(_sanitize)
    # Save top-N per model
    rows = []
    for model, g in df.groupby('model'):
        g_sorted = g.sort_values('importance', ascending=False).head(args.top_n)
        for rank, rec in enumerate(g_sorted.itertuples(index=False), start=1):
            rows.append({
                'model': model,
                'rank': rank,
                'feature': rec.feature,
                'importance': rec.importance,
            })
    pd.DataFrame(rows).to_csv(tables_dir / 'table_unsw_feature_top10.csv', index=False)

    # Heatmap across models (top-N by overall sum)
    feature_importance_heatmap(df, str(figs_dir / 'feature_importance_heatmap.png'), top_n=args.top_n)
    # Aggregated (single-column) heatmap top-20 by sum
    feature_importance_heatmap_overall(df, str(figs_dir / 'feature_importance_heatmap_overall.png'), top_n=20)


if __name__ == '__main__':
    main()
