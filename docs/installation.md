# Installation

## Prerequisites

- Python 3.10 or newer
- CUDA 11.8+ for GPU training

## Steps

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Troubleshooting

- If pycocotools fails on Windows, use `pip install pycocotools-windows`.
- If OpenCV import fails, reinstall with `pip install --force-reinstall opencv-python`.
