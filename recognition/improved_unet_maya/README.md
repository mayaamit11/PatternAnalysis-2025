# OASIS 2D Segmentation with CAN (Context Aggregation Network)

## Overview
This project segments 2D OASIS brain slices using a **2D CAN** (dilated residual CNN).
Target: **per-class Dice ≥ 0.90** on the test set.

## Files
- `modules.py` — CAN2D model (stem + dilated residual stack + 1×1 head).
- `dataset.py` — robust loader for OASIS PNG slices with `case_* ↔ seg_*` pairing.
- `train.py` — training/validation/testing with Dice+CE, plots & checkpoints.
- `predict.py` — load best checkpoint, compute test Dice, save colorized masks.
- `jobs/` — your SLURM runners (optional).
- `runs/` — outputs: `checkpoints/`, `logs/`, `preds/`.

## Data (Rangpur)
Read-only:
