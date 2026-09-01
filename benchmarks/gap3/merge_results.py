#!/usr/bin/env python3
"""Merge per-method CFTE benchmark CSV files into long/wide/per-video/overall tables."""
import argparse
from pathlib import Path
import pandas as pd

METRICS = [
    'dists', 'lpips', 'psnr', 'ssim', 'ms_ssim', 'mae', 'mse',
    'bpp', 'bpp_total', 'bpp_past', 'bpp_future', 'rdloss',
    'superposed_dists', 'superposed_lpips', 'superposed_psnr', 'superposed_ssim',
    'occlusion_mean_visibility', 'occlusion_mean_suppression',
    'occlusion_strong_suppression_ratio', 'occlusion_std',
    'occlusion_feature_attenuation',
    'adaptive_dists_initial', 'adaptive_dists_final',
    'adaptive_dists_improvement', 'adaptive_dists_sequence_mean',
    'adaptive_dists_curve_length', 'adaptive_iterations',
    'adaptive_delta_mean',
]
ID_COLS = ['sample_id', 'video_id', 'video_rel_path', 'past_idx', 'target_idx', 'future_idx', 'gap_past', 'gap_future']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True, help='Per-method CSVs')
    parser.add_argument('--output_dir', default='benchmark_results/tables')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for p in args.inputs:
        df = pd.read_csv(p)
        if len(df) > 0:
            frames.append(df)
    if not frames:
        raise SystemExit('No non-empty input CSV files.')

    long_df = pd.concat(frames, ignore_index=True, sort=False)
    long_path = out_dir / 'results_all_long.csv'
    long_df.to_csv(long_path, index=False)

    # Wide table for direct same-sample comparison. Include temporal-gap and
    # frame-index columns in the key so gap-3, gap-5, and gap-7 CSVs may be
    # merged safely even when their sample_id counters restart from 000001.
    available_metrics = [m for m in METRICS if m in long_df.columns]
    comparison_key = [
        c for c in [
            'sample_id', 'video_id', 'video_rel_path',
            'past_idx', 'target_idx', 'future_idx',
            'gap_past', 'gap_future',
        ] if c in long_df.columns
    ]
    base = long_df[comparison_key].drop_duplicates(comparison_key)
    wide = base.copy()
    for metric in available_metrics:
        piv = long_df.pivot_table(
            index=comparison_key,
            columns='model',
            values=metric,
            aggfunc='mean',
        )
        piv.columns = [f'{model}_{metric}' for model in piv.columns]
        piv = piv.reset_index()
        wide = wide.merge(piv, on=comparison_key, how='left')
    wide_path = out_dir / 'results_all_wide.csv'
    wide.to_csv(wide_path, index=False)

    numeric_metrics = [m for m in available_metrics if m in long_df.columns]
    gap_cols = [c for c in ['gap_past', 'gap_future'] if c in long_df.columns]

    per_video_group = ['video_id', 'model'] + gap_cols
    per_video = long_df.groupby(per_video_group, as_index=False, dropna=False).agg(
        num_samples=('sample_id', 'count'),
        **{f'{m}_mean': (m, 'mean') for m in numeric_metrics},
        **{f'{m}_std': (m, 'std') for m in numeric_metrics},
    )
    per_video_path = out_dir / 'results_per_video.csv'
    per_video.to_csv(per_video_path, index=False)

    overall_group = ['model'] + gap_cols
    overall = long_df.groupby(overall_group, as_index=False, dropna=False).agg(
        num_samples=('sample_id', 'count'),
        **{f'{m}_mean': (m, 'mean') for m in numeric_metrics},
        **{f'{m}_std': (m, 'std') for m in numeric_metrics},
    )
    sort_cols = gap_cols + (['dists_mean'] if 'dists_mean' in overall.columns else [])
    if sort_cols:
        overall = overall.sort_values(sort_cols)
    overall_path = out_dir / 'results_overall.csv'
    overall.to_csv(overall_path, index=False)

    # Chapter-4 hypothesis test: within each dual-reference method and gap,
    # evaluate whether a worse intermediate superposition (larger DISTS) is
    # associated with stronger mask suppression.
    corr_path = out_dir / 'occlusion_suppression_correlations.csv'
    corr_rows = []
    diagnostics = [
        'occlusion_mean_suppression',
        'occlusion_feature_attenuation',
    ]
    if 'superposed_dists' in long_df.columns:
        corr_group = ['model'] + gap_cols
        for group_values, group in long_df.groupby(corr_group, dropna=False):
            if not group['superposed_dists'].notna().any():
                continue
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            group_info = dict(zip(corr_group, group_values))
            for diagnostic in diagnostics:
                if diagnostic not in group.columns:
                    continue
                pair = group[['superposed_dists', diagnostic]].apply(
                    pd.to_numeric, errors='coerce').dropna()
                if len(pair) >= 2:
                    rho = pair['superposed_dists'].corr(pair[diagnostic], method='spearman')
                else:
                    rho = float('nan')
                corr_rows.append({
                    **group_info,
                    'diagnostic': diagnostic,
                    'num_samples': len(pair),
                    'spearman_rho': rho,
                })
    pd.DataFrame(corr_rows).to_csv(corr_path, index=False)

    # Reference-frame characteristics by sample, useful for finding hard samples.
    ref_cols = [c for c in long_df.columns if c.startswith(('past_', 'target_', 'future_')) or c in ID_COLS]
    ref_cols = [c for c in ref_cols if c in long_df.columns]
    ref_key = [c for c in comparison_key if c in ref_cols]
    ref_df = long_df[ref_cols].drop_duplicates(ref_key)
    ref_path = out_dir / 'reference_frame_characteristics.csv'
    ref_df.to_csv(ref_path, index=False)

    print('Wrote:')
    for p in [long_path, wide_path, per_video_path, overall_path, corr_path, ref_path]:
        print(' ', p)


if __name__ == '__main__':
    main()
