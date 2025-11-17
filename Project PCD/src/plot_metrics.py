"""Plot metrics from results CSV and save simple PSNR plots per noise type."""
import os
import sys
import csv
import math
import matplotlib.pyplot as plt

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False


def plot_metrics(csv_in, out_dir='results/plots'):
    os.makedirs(out_dir, exist_ok=True)

    if HAS_PANDAS:
        df = pd.read_csv(csv_in)
        if 'noise_level' in df.columns:
            df['noise_level'] = df['noise_level'].replace('', 0).astype(float)
        agg = df.groupby(['noise_type','noise_level']).agg({
            'sobel_psnr':'mean','prewitt_psnr':'mean','log_psnr':'mean','canny_psnr':'mean'
        }).reset_index()
    else:
        # minimal CSV parsing fallback
        rows = []
        with open(csv_in, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

        # build aggregation: {(noise_type,noise_level): {col: [vals]}}
        agg_map = {}
        for r in rows:
            ntype = r.get('noise_type','')
            nlevel = r.get('noise_level','') or '0'
            key = (ntype, float(nlevel))
            if key not in agg_map:
                agg_map[key] = {'sobel_psnr':[], 'prewitt_psnr':[], 'log_psnr':[], 'canny_psnr':[]}
            for col in ['sobel_psnr','prewitt_psnr','log_psnr','canny_psnr']:
                try:
                    val = float(r.get(col, '') if r.get(col,'')!='' else 'nan')
                except Exception:
                    val = float('nan')
                if not math.isnan(val):
                    agg_map[key][col].append(val)

        # convert to list of dicts similar to pandas agg
        agg_rows = []
        for (ntype, nlevel), d in agg_map.items():
            row = {'noise_type': ntype, 'noise_level': nlevel}
            for col in ['sobel_psnr','prewitt_psnr','log_psnr','canny_psnr']:
                vals = d[col]
                row[col] = sum(vals)/len(vals) if vals else float('nan')
            agg_rows.append(row)
        # create a simple structure for downstream code
        import collections
        agg = []
        # convert list to a structure similar to pandas DataFrame by grouping
        # We'll use agg variable as list of dicts with keys noise_type, noise_level, and mean columns
        # Downstream code treats 'agg' as iterable of dicts
        class DummyDF(list):
            def unique(self, col):
                return sorted(set([r[col] for r in self]))
            def __init__(self, items):
                super().__init__(items)
            def copy(self):
                return DummyDF(list(self))
        agg = DummyDF(agg_rows)

    # Handle both pandas DataFrame and our DummyDF(list of dicts)
    if HAS_PANDAS:
        noise_types = agg['noise_type'].unique()
    else:
        noise_types = agg.unique('noise_type')

    for ntype in noise_types:
        if HAS_PANDAS:
            sub = agg[agg['noise_type']==ntype].copy()
            sub = sub.sort_values('noise_level')
            x = sub['noise_level'].values
            y_sobel = sub['sobel_psnr'].values
            y_prew = sub['prewitt_psnr'].values
            y_log = sub['log_psnr'].values
            y_can = sub['canny_psnr'].values
        else:
            # filter and sort
            sub_list = [r for r in agg if r['noise_type']==ntype]
            sub_list = sorted(sub_list, key=lambda r: r['noise_level'])
            x = [r['noise_level'] for r in sub_list]
            y_sobel = [r['sobel_psnr'] for r in sub_list]
            y_prew = [r['prewitt_psnr'] for r in sub_list]
            y_log = [r['log_psnr'] for r in sub_list]
            y_can = [r['canny_psnr'] for r in sub_list]

        plt.figure(figsize=(6,4))
        plt.plot(x, y_sobel, marker='o', label='Sobel')
        plt.plot(x, y_prew, marker='o', label='Prewitt')
        plt.plot(x, y_log, marker='o', label='LoG')
        plt.plot(x, y_can, marker='o', label='Canny')
        plt.xlabel('Noise level')
        plt.ylabel('Mean PSNR')
        plt.title(f'PSNR per operator ({ntype})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        outp = os.path.join(out_dir, f'psnr_{ntype}.png')
        plt.savefig(outp)
        plt.close()
        print('Saved', outp)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python src/plot_metrics.py <metrics.csv> [out_dir]')
        sys.exit(1)
    csv_in = sys.argv[1]
    out = sys.argv[2] if len(sys.argv)>2 else os.path.join(os.path.dirname(csv_in), 'plots')
    plot_metrics(csv_in, out)
