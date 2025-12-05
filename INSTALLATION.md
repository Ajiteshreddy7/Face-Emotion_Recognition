# Installation & Usage Guide

## System Requirements

- **Python**: 3.8+
- **GPU**: NVIDIA GPU with CUDA support (training will be slow on CPU)
- **CUDA**: 11.8+ (optional, PyTorch will install)
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 2GB for dependencies + 1GB for FER2013 CSV

## Installation Steps

### 1. Clone Repository & Navigate

```bash
git clone https://github.com/Ajiteshreddy7/Face-Emotion_Recognition.git
cd Face-Emotion_Recognition
```

### 2. Create Virtual Environment (Recommended)

**On Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**For GPU (CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**For GPU (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch 2.0.0
CUDA Available: True
```

## Usage

### Quick Start (Recommended)

```bash
# 1. Check GPU setup
python run_training.py check

# 2. Train model (30 epochs, ~2-4 hours)
python run_training.py train

# 3. Test with GUI
python run_training.py test-gui
```

### Training

**Full training (30 epochs):**
```bash
python run_training.py train
```

**Quick test (5 epochs on subset):**
```bash
python run_training.py train --csv subset_2000_500_500.csv --epochs 5
```

**Custom hyperparameters:**
```bash
python run_training.py train \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --model efficientnet_b0
```

### GUI Testing

**Default (camera 0):**
```bash
python run_training.py test-gui
```

**Different camera:**
```bash
python run_training.py test-gui --camera 1
```

**Custom model checkpoint:**
```bash
python run_training.py test-gui --checkpoint /path/to/model.pth
```

### Evaluation

**Evaluate on PublicTest:**
```bash
python run_training.py evaluate
```

**Evaluate on PrivateTest:**
```bash
python run_training.py evaluate --split PrivateTest
```

## Data Preparation

### FER2013 Dataset

The project uses FER2013 dataset (public). You should have:

```
data/fer2013/
├── fer2013.csv              # Main dataset (35,887 samples)
└── subset_2000_500_500.csv  # Subset for quick testing (3,000 samples)
```

**CSV Format:**
```
emotion,pixels,Usage
3,143 123 124 ... (2304 values),Training
3,104 123 124 ... (2304 values),Training
...
```

- `emotion`: 0-6 (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- `pixels`: 48×48=2304 space-separated grayscale values (0-255)
- `Usage`: Training, PublicTest, or PrivateTest

### Adding Custom Dataset

Edit `src/data.py` to add your dataset loader.

## Output Structure

After training, outputs are saved to:

```
outputs/gpu_training/
├── best.pth                      # Best model weights
├── checkpoint_epoch_1.pth        # Checkpoint after epoch 1
├── checkpoint_epoch_2.pth        # ...
└── confusion_PublicTest.png      # Confusion matrix (after eval)
```

## Environment Variables

Optional environment variables for fine control:

```bash
# Increase logging detail
PYTHONVERBOSE=1 python run_training.py train

# Disable CUDA (force CPU)
CUDA_VISIBLE_DEVICES="" python run_training.py train

# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python run_training.py train  # GPU 0
CUDA_VISIBLE_DEVICES=1 python run_training.py train  # GPU 1
```

## GPU Selection

**Check available GPUs:**
```bash
nvidia-smi
```

**Output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx                                                          |
+----------------|----------------------+----------------------+
| GPU Name | Driver Version | Memory-Usage | Compute Cap. |
+============================================+
| 0 NVIDIA GeForce RTX 3090 | 525.xx | 0/24220MiB | 8.6 |
+------|--------|--------|---------|
```

**Use specific GPU:**
```bash
CUDA_VISIBLE_DEVICES=0 python run_training.py train
```

## Memory Management

### If You Run Out of Memory

**Option 1: Reduce batch size**
```bash
python run_training.py train --batch-size 16
```

**Option 2: Use gradient accumulation**
```bash
# Edit train.py to add gradient accumulation logic
```

**Option 3: Freeze backbone**
```bash
# Edit train.py to add: --freeze-backbone
```

### Memory Requirements by Batch Size

| Batch Size | GPU VRAM |
|-----------|----------|
| 16 | 4GB |
| 32 | 8GB |
| 64 | 12GB |
| 128 | 20GB |

## Troubleshooting

### GPU Not Detected

```bash
# Check CUDA in PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### CUDA Out of Memory

```bash
# Check memory
nvidia-smi

# Reduce batch size
python run_training.py train --batch-size 16

# Or use CPU (slow)
CUDA_VISIBLE_DEVICES="" python run_training.py train
```

### Camera Not Working

```bash
# List cameras on Linux
ls /dev/video*

# Try different camera
python run_training.py test-gui --camera 1
```

### Import Errors

```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Check specific package
python -c "import timm; print(timm.__version__)"
```

## Performance Tips

### Speed Up Training

1. **Increase num_workers:**
   ```bash
   python run_training.py train --num-workers 8
   ```

2. **Use pin_memory (default):**
   ```bash
   # Already enabled in run_training.py
   ```

3. **Monitor GPU:**
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **Increase batch size (if VRAM allows):**
   ```bash
   python run_training.py train --batch-size 128
   ```

### Reduce Training Time

- Use smaller model: `--model mobilenetv3_small`
- Use subset data: `--csv subset_2000_500_500.csv`
- Reduce epochs: `--epochs 10`

## Advanced Configuration

### Edit `src/train.py` for Advanced Options

```python
# Freeze backbone for transfer learning
python run_training.py train --freeze-backbone

# Unfreeze specific layers
--unfreeze-top-n 2
```

### Custom Augmentations

Edit augmentation pipeline in `src/data.py`:
```python
train_transform = A.Compose([
    A.RandomResizedCrop(...),
    # Add your augmentations here
])
```

## Support & Documentation

- **Quick Start**: `QUICK_START.md`
- **Detailed Guide**: `GPU_TRAINING_GUIDE.md`
- **Setup Summary**: `SETUP_SUMMARY.md`
- **Code**: Check `src/` directory

## Next Steps

1. ✅ Install dependencies
2. ✅ Prepare FER2013 data
3. ✅ Verify GPU setup: `python run_training.py check`
4. ✅ Train model: `python run_training.py train`
5. ✅ Test GUI: `python run_training.py test-gui`
6. ✅ Evaluate: `python run_training.py evaluate`

Good luck! 🚀
