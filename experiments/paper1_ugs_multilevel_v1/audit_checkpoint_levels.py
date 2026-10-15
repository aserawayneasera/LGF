#!/usr/bin/env python3
"""Inspect actual backbone feature keys in an existing RetinaNet LGF checkpoint.

Usage:
  python audit_checkpoint_levels.py /path/to/last.pth

Loads checkpoint metadata and state dictionary only; never edits files.
Only load checkpoints from sources you trust (torch deserialization).
"""
import argparse
import re
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError('Checkpoint is not a dict')
    state = next((checkpoint[k] for k in ('model_state_dict', 'ema_state_dict', 'state_dict') if isinstance(checkpoint.get(k), dict)), None)
    if state is None:
        state = checkpoint if all(isinstance(k, str) for k in checkpoint) else None
    if state is None:
        raise ValueError('No model state found')
    keys = sorted({m.group(1) for k in state for m in [re.match(r'^(?:module\.)?backbone\.block_fpn_in\.([^\.]+)\.', k)] if m})
    labels = {'0':'C3 (ResNet layer2, 512 channels)', '1':'C4 (ResNet layer3, 1024 channels)', '2':'C5 (ResNet layer4, 2048 channels)'}
    print('Checkpoint:', args.checkpoint)
    print('Saved --level:', checkpoint.get('level', checkpoint.get('insert_level', 'not specified')))
    print('Actual LGF feature keys:', keys)
    for key in keys:
        print(f'  {key}: {labels.get(key, "unknown stage (inspect architecture)")}')
    if not keys:
        print('No feature blocks found under backbone.block_fpn_in (baseline or possible placement issue).')


if __name__ == '__main__':
    main()
