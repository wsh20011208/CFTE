# Supplementary physical-bitstream experiment

This directory contains the scripts used for the dense GOP-4 physical-bitstream experiment reported in the thesis.

## Assumptions

- The archived scripts were originally run with `/home/featurize/work` as the experiment working root.
- In the original experiment environment, processed VoxCeleb test videos were located under `<WORK>/vox/test`.
- The four project directories were located under `<WORK>/CFTE*`.
- Gap-3-trained final checkpoints are used.
- VTM anchors use QP 32 by default.
- Shared anchor VTM bitstreams are generated once and reused by all methods.
- Compact residuals are written as real `EntropyBottleneck` byte streams.
- Single CFTE, NoRefine and FixedCMR do not read the true inter-frame target at the decoder.
- AdaptiveCMR receives the true target as decoder-side DISTS oracle information; those oracle bits are not counted.

The MP4 files produced by the decoder are visualization outputs. Their MP4 container size is not the transmitted bitrate.

For a different local layout, set `CFTE_WORK_ROOT` for the shell wrapper and update the `WORK` constant in the Python scripts where required.
