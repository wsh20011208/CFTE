# CFTE Superposition Modification Notes

This version implements a two-reference superposition training model:

- `source_past`: reference frame before the target/current frame
- `driving`: target/current frame to reconstruct
- `source_future`: reference frame after the target/current frame

Two independent CFTE branches are built:

1. past-CFTE: `source_past -> driving`
2. future-CFTE: `source_future -> driving`

The final reconstructed frame is:

```text
prediction = 0.5 * prediction_past + 0.5 * prediction_future
```

## Main changed files

- `frames_dataset.py`
  - training now samples three temporally ordered frames instead of two
  - returns `source_past`, `driving`, `source_future`
  - keeps `source` as an alias of `source_past` for compatibility
  - disables `time_flip` during training to preserve past/current/future semantics

- `run.py`
  - builds two independent CFTE branches: past and future
  - keeps one discriminator on the final superposed prediction

- `modules/model.py`
  - `GeneratorFullModel` runs both CFTE branches and superposes predictions with fixed 0.5/0.5 weights
  - RD rate term uses the sum of past-branch and future-branch bits
  - perceptual/DISTS/GAN losses are computed on the final superposed output

- `train.py`
  - optimizes both branches together
  - saves both past/future branch checkpoints
  - keeps old alias keys for partial compatibility

- `logger.py`
  - checkpoint loader can initialize both branches from an old single-CFTE checkpoint
  - visualizer shows past reference, future reference, target, branch outputs, and final output

- `config/vox-256.yaml`
  - adds explicit `superposition_weight_past: 0.5` and `superposition_weight_future: 0.5`

## Training command

```bash
python run.py --mode train --config ./config/vox-256.yaml --device_ids 0
```

## Starting from old single-CFTE checkpoint

You can pass the old checkpoint. If the checkpoint only contains the original single-reference keys, both branches are initialized from the same single-CFTE weights:

```bash
python run.py --mode train --config ./config/vox-256.yaml --checkpoint /path/to/old/checkpoint.pth.tar --device_ids 0
```

## Important note

This doubles the generator/KP/compressor branch computation and memory compared with the previous single-reference model. If CUDA memory is insufficient, reduce `train_params.batch_size` in `config/vox-256.yaml`.
