# Original-metric benchmark v4: occlusion suppression diagnostics

This benchmark version is intentionally compact.

It computes all reported image-quality metrics using the original CFTE project evaluation implementation in:

```text
evaluate/multiMetric.py
```

Metric sources:

- `DISTS` from `evaluate/multiMetric.py`
- `LPIPS` from `evaluate/multiMetric.py` (`LPIPSvgg`)
- `PSNR` from `evaluate/multiMetric.py` (`cacl_psnr`)
- `SSIM` from `evaluate/multiMetric.py` (`cacl_ssim`)

Each result CSV records only the essential values.

Final decoded prediction vs target:

```text
dists, lpips, psnr, ssim
final_dists, final_lpips, final_psnr, final_ssim
```

Two-frame superposed/deformed intermediate image vs target:

```text
superposed_dists, superposed_lpips, superposed_psnr, superposed_ssim
```

Important: `superposed_*` columns are written only for the two-frame methods:

- NoRefine: `out['deformed']`, the raw two-reference superposed deformed image.
- FixedCMR: `out['deformed']`, the CMR-refined superposed deformed image.
- AdaptiveCMR: `out['deformed']`, the adaptive-corrected superposed deformed image.

Single CFTE does not have two-frame superposition, so the single-frame result CSV does not write `superposed_*` columns and does not save `superposed_deformed.png`.

Rate columns:

```text
bpp, bpp_total, bpp_past, bpp_future, rdloss
```

Final occlusion/visibility-mask diagnostics:

```text
occlusion_mean_visibility
occlusion_mean_suppression
occlusion_strong_suppression_ratio
occlusion_std
occlusion_feature_attenuation
```

Definitions follow Chapter 4:

- `occlusion_mean_visibility = mean(O)`
- `occlusion_mean_suppression = 1 - mean(O)`
- `occlusion_strong_suppression_ratio = mean(O < 0.1)`
- `occlusion_std` is the population standard deviation over the final mask
- `occlusion_feature_attenuation = 1 - ||O * F_pre||_1 / (||F_pre||_1 + eps)`

For **Single CFTE**, `O` and `F_pre` are the branch-level final mask and deformed feature. For **NoRefine**, **FixedCMR**, and **AdaptiveCMR**, they are the final post-superposition mask and fused deformed feature. These are decoder-side diagnostics, not additional transmitted BPP.

`merge_results.py` and `4Method_Analysis_with_BPP_Superposed.py` additionally write gap-wise summaries and Spearman correlations of `superposed_dists` with `occlusion_mean_suppression` and `occlusion_feature_attenuation` for the three dual-reference methods.

This version keeps the CSV compact but adds the Chapter-4 final visibility-mask diagnostics above. Internal adaptive iteration curves, MAE/MSE, frame statistics, and past/future-to-target diagnostic metrics remain excluded.


## v4 note

Single CFTE visualization still shows the single-reference deformed image when available. However, superposed_* metric columns are written only for two-frame methods, because Single CFTE has no two-frame superposition.
