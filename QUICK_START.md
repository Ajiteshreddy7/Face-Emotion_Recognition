# 🚀 Quick Start Guide - GPU Training & GUI Test

## TL;DR - Three Simple Commands

```bash
# 1. Check GPU is working
python run_training.py check

# 2. Train the model (30 epochs, ~2-4 hours on RTX 3090)
python run_training.py train

# 3. Test with webcam GUI
python run_training.py test-gui
```

That's it! 🎉

---

## Detailed Steps

### Step 1: Check Your GPU Setup

```bash
python run_training.py check
```

Expected output:
```
✅ GPU AVAILABLE
   Device Count: 1
   Device Name: NVIDIA GeForce RTX 3090
   CUDA Version: 11.8
   PyTorch Version: 2.0.0

✅ Data found at data/fer2013/fer2013.csv

✅ All checks passed!
```

### Step 2: Train the Model

```bash
python run_training.py train
```

**What happens:**
- Loads FER2013 dataset (35,887 training samples)
- Fine-tunes ResNet50 (pretrained on ImageNet)
- Runs for 30 epochs with batch size 64
- Uses GPU with mixed precision for speed
- Saves best model to `outputs/gpu_training/best.pth`

**Expected training time:**
- RTX 3090: ~2-3 hours
- RTX 4090: ~1.5-2 hours
- A100: ~1 hour
- Consumer GPU: 3-6 hours

**Monitor progress:**
```bash
# In another terminal, watch GPU usage
watch -n 1 nvidia-smi
```

### Step 3: Test with GUI

```bash
python run_training.py test-gui
```

**What happens:**
- Opens webcam
- Detects faces with MTCNN
- Predicts emotion in real-time
- Shows emotion + confidence

**Controls:**
- Press 'Q' to quit

---

## Advanced Options

### Quick Test (Subset Data, 5 Epochs)

```bash
python run_training.py train --csv subset_2000_500_500.csv --epochs 5
```

Takes ~10 minutes. Good for testing the pipeline!

### Evaluate on Test Set

```bash
python run_training.py evaluate
```

Generates confusion matrix and metrics for PublicTest split.

### Custom Camera

```bash
python run_training.py test-gui --camera 1
```

Use `--camera 0`, `1`, `2`, etc. for different cameras.

---

## Troubleshooting

### ❌ GPU not detected

```bash
# Check if PyTorch can see GPU
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### ❌ Out of Memory Error

Reduce batch size:
```bash
python run_training.py train --batch-size 32
```

Or use subset data:
```bash
python run_training.py train --csv subset_2000_500_500.csv
```

### ❌ Slow Training

Check GPU usage:
```bash
nvidia-smi
```

If GPU usage < 50%, increase `--num-workers` (default 4):
```bash
python run_training.py train --num-workers 8
```

### ❌ Camera not opening

List available cameras:
```bash
ls /dev/video*
```

Try different camera index:
```bash
python run_training.py test-gui --camera 1
```

---

## What Gets Saved?

After training, you'll have:

```
outputs/gpu_training/
├── best.pth                    # Best model (use this for GUI)
├── checkpoint_epoch_1.pth      # Checkpoint after epoch 1
├── checkpoint_epoch_2.pth      # ... etc
└── confusion_PublicTest.png    # Confusion matrix (if evaluated)
```

---

## Expected Performance

After 30 epochs of training:

| Metric | Value |
|--------|-------|
| Train Accuracy | ~72% |
| Val Accuracy | ~67% |
| Test Accuracy | ~65% |
| Training Time | 2-4 hours |

Individual emotion accuracy varies:
- Happy/Neutral: 70-80% ✅
- Sad/Angry: 60-70% 📊
- Fear/Disgust: 50-60% 🔧

---

## All Available Commands

```bash
# GPU and data checks
python run_training.py check

# Training
python run_training.py train                           # Full training
python run_training.py train --epochs 5                # Quick test
python run_training.py train --csv subset_2000_500_500.csv  # Subset
python run_training.py train --batch-size 32           # Reduce memory

# GUI Testing
python run_training.py test-gui                        # Default camera
python run_training.py test-gui --camera 1             # Different camera
python run_training.py test-gui --checkpoint path/to/model.pth

# Evaluation
python run_training.py evaluate                        # PublicTest
python run_training.py evaluate --split PrivateTest    # PrivateTest
```

---

## Next: Fine-tuning & Experiments

After the initial training, try:

1. **Different Models:**
   ```bash
   python run_training.py train --model efficientnet_b0
   python run_training.py train --model mobilenetv3_small
   ```

2. **Different Learning Rates:**
   ```bash
   python run_training.py train --lr 1e-4  # Smaller
   python run_training.py train --lr 1e-3  # Larger
   ```

3. **Freeze Backbone (transfer learning):**
   ```bash
   # Edit train.py to add: --freeze-backbone
   ```

---

## FAQ

**Q: Can I use CPU instead?**
A: Yes, but it will be very slow. The commands will work without GPU, but expect 10x slower training.

**Q: Can I stop and resume training?**
A: Currently no, but we can add checkpointing resume logic.

**Q: What's MTCNN?**
A: Face detection model that finds faces in the webcam feed before emotion prediction.

**Q: Can I use a different dataset?**
A: Yes! Modify data loading in `src/data.py`.

---

## Contact & Support

For issues or questions, check:
- `GPU_TRAINING_GUIDE.md` - Detailed guide
- `src/train.py` - Training implementation
- `src/gui.py` - GUI implementation

Good luck! 🚀
