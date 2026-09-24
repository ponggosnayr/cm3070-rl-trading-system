"""Plot measured CSVs: loss needs step,value_loss; SHAP needs feature,attribution."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate(source, kind, output):
    df = pd.read_csv(source)
    cols = ['step', 'value_loss'] if kind == 'loss' else ['feature', 'attribution']
    if not set(cols).issubset(df.columns) or df.empty or df[cols].isna().any().any():
        raise ValueError(f'Non-empty measurements required: {cols}')
    numeric = cols if kind == 'loss' else ['attribution']
    if not np.isfinite(df[numeric].to_numpy(dtype=float)).all():
        raise ValueError('Measurements must be finite')
    fig, ax = plt.subplots(figsize=(8, 5))
    if kind == 'loss':
        df = df.sort_values('step')
        ax.plot(df.step, df.value_loss)
        ax.set(xlabel='Training step', ylabel='Value loss', title='Measured training loss')
    else:
        df = df.sort_values('attribution')
        ax.barh(df.feature, df.attribution)
        ax.set(xlabel='SHAP contribution to actor logit', title='Measured feature attributions')
    output.mkdir(parents=True, exist_ok=True)
    dest = output / ('value_loss_convergence.png' if kind == 'loss' else 'feature_importance.png')
    fig.tight_layout()
    fig.savefig(dest, dpi=200)
    plt.close(fig)
    dest.with_suffix('.source.json').write_text(json.dumps({
        'source': str(source.resolve()), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'rows': len(df), 'kind': kind}, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loss-csv', type=Path)
    parser.add_argument('--shap-csv', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'images')
    args = parser.parse_args()
    if not args.loss_csv and not args.shap_csv:
        parser.error('Supply measured --loss-csv or --shap-csv; no illustrative values are generated.')
    for kind, source in [('loss', args.loss_csv), ('shap', args.shap_csv)]:
        if source:
            generate(source, kind, args.output)


if __name__ == '__main__':
    main()
