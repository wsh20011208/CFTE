# Preprocessed Dataset

This directory contains the **preprocessed train and test videos used by the thesis experiments**.

The videos are derived from the VoxCeleb dataset and were prepared for the CFTE-based experiments at **256 × 256** resolution.

## Directory structure

```text
data/
├── README.md
├── train/
└── test/
```

- `train/` contains the preprocessed videos used for model training.
- `test/` contains the preprocessed videos used for evaluation.
- The supplementary dense GOP-4 physical-bitstream experiment uses the 444 videos in the test set.

## Relation to the experiments

The same processed dataset layout is used by the four archived implementations:

- Single CFTE
- NoRefine
- FixedCMR
- AdaptiveCMR

The controlled benchmark evaluates temporal gaps of 3, 5, and 7 frames using matched target-frame sets across all four methods.

Reported matched target counts are:

- gap 3: 15,023
- gap 5: 14,191
- gap 7: 13,627
- total: 42,841

## Notes

These files are **preprocessed derivatives**, not the original VoxCeleb distribution.

The repository source code and gap-specific configuration files define the training and evaluation procedures used in the thesis.

Users should review and comply with the original VoxCeleb terms and any applicable restrictions when using or redistributing the underlying data.
