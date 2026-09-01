#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Four-method analyzer for original-metric benchmark CSVs.

It summarizes:
  final DISTS / LPIPS / PSNR / SSIM
  BPP / BPP parts
  superposed DISTS / LPIPS / PSNR / SSIM
  final occlusion-map suppression diagnostics and DISTS correlations

Usage:
python 4Method_Analysis_with_BPP_Superposed.py \
  --single results_single_cfte.csv \
  --norefine results_norefine.csv \
  --fixedcmr results_fixedcmr.csv \
  --adaptive results_adaptivecmr.csv
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

METHODS = {
    "single": "Single CFTE",
    "norefine": "NoRefine",
    "fixedcmr": "FixedCMR",
    "adaptive": "AdaptiveCMR",
}

RANK_METRICS = {
    # final decoded prediction vs target; aliases kept for compatibility
    "dists": "lower",
    "lpips": "lower",
    "psnr": "higher",
    "ssim": "higher",
    # explicit final aliases
    "final_dists": "lower",
    "final_lpips": "lower",
    "final_psnr": "higher",
    "final_ssim": "higher",
    # superposed/deformed intermediate vs target
    "superposed_dists": "lower",
    "superposed_lpips": "lower",
    "superposed_psnr": "higher",
    "superposed_ssim": "higher",
    # bitrate
    "bpp": "lower",
}

MEAN_COLUMNS = [
    "dists", "lpips", "psnr", "ssim",
    "final_dists", "final_lpips", "final_psnr", "final_ssim",
    "superposed_dists", "superposed_lpips", "superposed_psnr", "superposed_ssim",
    "bpp", "bpp_total", "bpp_past", "bpp_future",
    "rdloss",
    "adaptive_dists_initial", "adaptive_dists_final", "adaptive_dists_improvement", "adaptive_iterations",
    "occlusion_mean_visibility", "occlusion_mean_suppression",
    "occlusion_strong_suppression_ratio", "occlusion_std",
    "occlusion_feature_attenuation",
]

KEYS = ["sample_id", "video_id", "past_idx", "target_idx", "future_idx", "gap_past", "gap_future"]


def read_result(path):
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    return df


def check_keys(dfs):
    for name, df in dfs.items():
        missing = [k for k in KEYS if k not in df.columns]
        if missing:
            raise ValueError(f"{name} missing key columns: {missing}")


def align_wide(dfs):
    wide = dfs["single"][KEYS].copy()
    all_metric_cols = sorted(set().union(*[set(df.columns) for df in dfs.values()]))
    value_cols = [c for c in all_metric_cols if c not in KEYS]
    for method, df in dfs.items():
        keep = KEYS + [c for c in value_cols if c in df.columns]
        tmp = df[keep].copy()
        tmp = tmp.rename(columns={c: f"{method}_{c}" for c in tmp.columns if c not in KEYS})
        wide = wide.merge(tmp, on=KEYS, how="inner")
    return wide


def mean_summary(wide):
    rows = []
    for method, label in METHODS.items():
        row = {"method": label}
        for col in MEAN_COLUMNS:
            wcol = f"{method}_{col}"
            if wcol in wide.columns:
                row[col] = pd.to_numeric(wide[wcol], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def rank_counts(wide):
    rows = []
    method_keys_all = list(METHODS.keys())
    for metric, direction in RANK_METRICS.items():
        # For final metrics and BPP, all four methods usually exist.
        # For superposed_* metrics, Single CFTE intentionally has no column,
        # so rank only among the available two-frame methods.
        method_keys = [m for m in method_keys_all if f"{m}_{metric}" in wide.columns]
        if len(method_keys) < 2:
            continue

        cols = [f"{m}_{metric}" for m in method_keys]
        values = wide[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if direction == "lower":
            scores = values.copy()
            scores[np.isnan(scores)] = np.inf
        else:
            scores = -values.copy()
            scores[np.isnan(scores)] = np.inf
        ranks = np.zeros_like(scores, dtype=int)
        for i in range(scores.shape[0]):
            order = np.lexsort((np.arange(scores.shape[1]), scores[i]))
            for r, j in enumerate(order, start=1):
                ranks[i, j] = r
        for j, method in enumerate(method_keys):
            row = {
                "metric": metric,
                "direction": direction,
                "method": METHODS[method],
                "num_ranked_methods": len(method_keys),
            }
            for r in range(1, 5):
                row[f"rank_{r}"] = int((ranks[:, j] == r).sum()) if r <= len(method_keys) else 0
            rows.append(row)
    return pd.DataFrame(rows)



def mean_summary_by_gap(dfs):
    rows = []
    for method, label in METHODS.items():
        df = dfs[method].copy()
        gap_cols = [c for c in ["gap_past", "gap_future"] if c in df.columns]
        groups = [((), df)] if not gap_cols else df.groupby(gap_cols, dropna=False)
        for gap_values, group in groups:
            if not isinstance(gap_values, tuple):
                gap_values = (gap_values,)
            row = {"method": label, **dict(zip(gap_cols, gap_values)), "num_samples": len(group)}
            for col in MEAN_COLUMNS:
                if col in group.columns:
                    values = pd.to_numeric(group[col], errors="coerce")
                    row[f"{col}_mean"] = values.mean()
                    row[f"{col}_std"] = values.std()
            rows.append(row)
    return pd.DataFrame(rows)


def occlusion_spearman_summary(dfs):
    rows = []
    diagnostics = [
        "occlusion_mean_suppression",
        "occlusion_feature_attenuation",
    ]
    # Single CFTE has no two-reference intermediate superposed image.
    for method in ["norefine", "fixedcmr", "adaptive"]:
        df = dfs[method].copy()
        if "superposed_dists" not in df.columns:
            continue
        gap_cols = [c for c in ["gap_past", "gap_future"] if c in df.columns]
        groups = [((), df)] if not gap_cols else df.groupby(gap_cols, dropna=False)
        for gap_values, group in groups:
            if not isinstance(gap_values, tuple):
                gap_values = (gap_values,)
            gap_info = dict(zip(gap_cols, gap_values))
            for diagnostic in diagnostics:
                if diagnostic not in group.columns:
                    continue
                pair = group[["superposed_dists", diagnostic]].apply(
                    pd.to_numeric, errors="coerce").dropna()
                if len(pair) >= 2:
                    rho = pair["superposed_dists"].corr(pair[diagnostic], method="spearman")
                else:
                    rho = float("nan")
                rows.append({
                    "method": METHODS[method],
                    **gap_info,
                    "diagnostic": diagnostic,
                    "num_samples": len(pair),
                    "spearman_rho": rho,
                })
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", default="results_single_cfte.csv")
    parser.add_argument("--norefine", default="results_norefine.csv")
    parser.add_argument("--fixedcmr", default="results_fixedcmr.csv")
    parser.add_argument("--adaptive", default="results_adaptivecmr.csv")
    parser.add_argument("--outdir", default="four_method_original_metric_summary")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dfs = {
        "single": read_result(args.single),
        "norefine": read_result(args.norefine),
        "fixedcmr": read_result(args.fixedcmr),
        "adaptive": read_result(args.adaptive),
    }
    check_keys(dfs)
    wide = align_wide(dfs)

    avg = mean_summary(wide)
    avg_by_gap = mean_summary_by_gap(dfs)
    ranks = rank_counts(wide)
    occlusion_corr = occlusion_spearman_summary(dfs)

    wide.to_csv(outdir / "wide_four_methods.csv", index=False)
    avg.to_csv(outdir / "summary_average_with_bpp_superposed.csv", index=False)
    avg_by_gap.to_csv(outdir / "summary_average_by_gap_with_occlusion.csv", index=False)
    ranks.to_csv(outdir / "summary_rank_counts_with_bpp_superposed.csv", index=False)
    occlusion_corr.to_csv(outdir / "occlusion_suppression_spearman.csv", index=False)

    print("Input rows:")
    for k, df in dfs.items():
        print(f"  {METHODS[k]}: {len(df)}")
    print(f"Aligned rows: {len(wide)}")
    print("\n=== Mean summary ===")
    print(avg.to_string(index=False))
    print("\n=== Mean summary by gap, including occlusion diagnostics ===")
    print(avg_by_gap.to_string(index=False))
    print("\n=== Occlusion suppression Spearman correlations ===")
    print(occlusion_corr.to_string(index=False))
    print("\n=== Rank counts ===")
    print(ranks.to_string(index=False))
    print(f"\nSaved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
