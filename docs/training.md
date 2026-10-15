# Training

The script supports four families:
- baseline
- se
- cbam
- lgf_sum, lgf_softmax, lgf_gated, lgf_gated_spatial

Insert level choices: C3, C4, C5. C3 is recommended for small objects.

## Custom COCO-style dataset

```bash
python run.py --mode lgf_gated_spatial --dataset custom   --train-img /path/to/train/images   --train-ann /path/to/train/annotations.json   --val-img   /path/to/val/images   --val-ann   /path/to/val/annotations.json   --insert-level C3 --epochs 80 --batch-size 4 --accum-steps 4
```

Key flags:
- `--epochs`: default 80
- `--batch-size`: default 4
- `--accum-steps`: default 4
- `--base-lr`: default 0.005
- `--lr-milestones`: default 40 60
- `--img-size`: default 640
- `--num-workers`: default 8

## Built-in dataset codes

- `coco_nw`: COCO Non‑weather subset
- `coco_weather`: COCO with synthetic fog, rain, snow
- `acdc`: ACDC driving scenes

These defaults reference local paths. Override with the `--train-*` and `--val-*` flags as shown above.
