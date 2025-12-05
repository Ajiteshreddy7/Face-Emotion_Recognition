Robust Facial Emotion Recognition

Group: Jasmin Bargir and Kiranmayi Modugu

Project title: Robust Facial Emotion Recognition

Objective: Design and implement a robust facial emotion recognition (FER) system that classifies the seven basic emotions (anger, disgust, fear, happiness, sadness, surprise, neutral), emphasizing generalization across demographics using augmentation and transfer learning.

This repository contains a PyTorch-based scaffold for training FER models with:
- Data loaders and preprocessing for FER2013 (CSV) and optional datasets
- Albumentations-based augmentations (geometric, photometric, occlusion)
- Pretrained backbone support (via `timm`) and a lightweight CNN baseline
- Training loop with mixed precision, AdamW, scheduler, and checkpointing

Getting started
1. Create a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Prepare data:
- FER2013 CSV: place `fer2013.csv` under `data/fer2013/fer2013.csv` (script expects CSV with `emotion` and `pixels` columns). See dataset sources and license terms before download.
- Optionally provide other datasets and change paths in `src/train.py` arguments.

3. Train a model (example):

```bash
python -m src.train --data-dir data/fer2013 --dataset fer2013 --model resnet50 --epochs 30 --batch-size 64 --output-dir outputs
```

Project layout
- `requirements.txt` — Python dependencies
- `src/data.py` — datasets and transforms (FER2013 loader + optional MTCNN alignment)
- `src/models.py` — pretrained backbone wrapper and lightweight CNN baseline
- `src/train.py` — training & validation loop, checkpointing, metrics
- `src/utils.py` — metric helpers and utils

Notes
- This scaffold is intended as a starting point — adapt dataset paths, hyperparameters, and evaluation scheme for final experiments and cross-dataset evaluation.
- Use GPU with CUDA and consider mixed precision for faster training.

License & citation
- Check dataset licenses (FER2013, AffectNet, CK+, RAF-DB) before using them in publications.

If you want, I can now run a quick lint or create a sample run command file and small example dataset to validate the pipeline locally.
# Face-Emotion_Recognition