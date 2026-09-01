# Adaptive DISTS Curve Logging

This version records the DISTS value after every DISTS-guided adaptive CMR update.

## What is measured

The curve is computed from:

```text
DISTS(superposed_deformed_image, target_image)
```

where:

```text
superposed_deformed_image = weight_past * warp(source_past, M_past)
                         + weight_future * warp(source_future, M_future)
```

This is the advisor-requested DISTS comparison between the superposed deformed image and the target. It is not the final decoder-output DISTS.

## Curve definition

For max_iterations = 20, the model logs:

```text
metric_adaptive_dists_iter_00  # initial CMR1/CMR2 DISTS before adaptive update
metric_adaptive_dists_iter_01  # DISTS after accepted adaptive update 1
...
metric_adaptive_dists_iter_20  # DISTS after accepted adaptive update 20
```

If the adaptive loop stops early, the remaining entries are padded with the final DISTS. Therefore, a flat tail means the loop had already converged or stopped before the maximum number of iterations.

The actual number of accepted updates is still logged as:

```text
metric_adaptive_iterations
```

The effective curve length, including the initial point, is logged as:

```text
metric_adaptive_dists_curve_length
```

Thus:

```text
metric_adaptive_dists_curve_length = metric_adaptive_iterations + 1
```

## Saved files

At the end of every epoch, `logger.py` saves one epoch-averaged curve under:

```text
checkpoint_new/.../adaptive-dists-curves/
```

with two files per epoch:

```text
00000000-adaptive-dists-curve.png
00000000-adaptive-dists-curve.csv
```

The PNG visualizes DISTS versus adaptive iteration. The CSV stores the exact numerical values for later plotting or reporting.

## How to judge whether 20 iterations are necessary

If the curve decreases only for the first few iterations and then becomes flat, the model does not need 20 iterations for that training stage. For example, if the curve stabilizes after iteration 6 and `metric_adaptive_iterations` is usually around 6, then setting `max_iterations` to 8 or 10 is enough.

If the curve keeps decreasing until iteration 20, the current upper bound may still be useful.
