#!/usr/bin/env bash
set -e

python run.py --mode lgf_gated_spatial --dataset custom   --train-img "$TRAIN_IMG"   --train-ann "$TRAIN_ANN"   --val-img   "$VAL_IMG"   --val-ann   "$VAL_ANN"   --insert-level C3 --epochs 80 --batch-size 4 --accum-steps 4
