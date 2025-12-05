# GPU Training & GUI Test Guide

This guide will help you train the Face Emotion Recognition model on your local GPU and test it with the GUI.

## Prerequisites

1. **GPU Setup**: Ensure you have CUDA-enabled GPU with PyTorch CUDA support installed
2. **Python Environment**: Python 3.8+
3. **Data**: FER2013 CSV file at `data/fer2013/fer2013.csv`

## Step 1: Verify GPU Availability

Before training, verify your GPU is available:

```bash
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
GPU Available: True
GPU Count: 1
GPU Name: NVIDIA GeForce RTX 3090 (or your GPU model)
```

## Step 2: Install Dependencies

```bash
# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Step 3: Train the Model on GPU

### Option A: Using the Training Script (Recommended)

```bash
bash train_gpu.sh
```

This will:
- Train ResNet50 with ImageNet pretrained weights
- Use FER2013 full dataset (35,887 training samples)
- Run for 30 epochs with batch size 64
- Save checkpoints to `outputs/gpu_training/`
- Automatically enable mixed precision (AMP) on GPU

### Option B: Manual Training with Custom Settings

```bash
python -m src.train \
    --data-dir data/fer2013 \
    --csv fer2013.csv \
    --model resnet50 \
    --epochs 30 \
    --batch-size 64 \
    --image-size 224 \
    --lr 3e-4 \
    --output-dir outputs/gpu_training \
    --device cuda \
    --num-workers 4 \
    --pin-memory \
    --to-rgb
```

### Option C: Quick Test with Smaller Dataset

To test the pipeline faster, use the subset dataset:

```bash
python -m src.train \
    --data-dir data/fer2013 \
    --csv subset_2000_500_500.csv \
    --model resnet50 \
    --epochs 10 \
    --batch-size 32 \
    --image-size 224 \
    --lr 3e-4 \
    --output-dir outputs/quick_test \
    --device cuda \
    --num-workers 4 \
    --pin-memory \
    --to-rgb
```

## Training Output

During training, you'll see:
```
Epoch 1/30
train: 100%|████████| 561/561 [01:23<00:00,  6.73it/s]
val: 100%|████████| 72/72 [00:09<00:00,  7.88it/s]
Train loss 1.8234 acc 0.3812 f1 0.3124
Val   loss 1.6543 acc 0.4256 f1 0.3887

Epoch 2/30
...
```

- **loss**: Cross-entropy loss
- **acc**: Accuracy (0-1)
- **f1**: Macro F1 score (0-1)

## Step 4: Test the Model with GUI

Once training completes, test with your webcam:

### Using the GUI Script

```bash
bash test_gui.sh
```

This will use the best model from training and start the webcam inference.

### Custom Checkpoint

```bash
bash test_gui.sh outputs/gpu_training/best.pth
```

Or directly:

```bash
python -m src.gui \
    --checkpoint outputs/gpu_training/best.pth \
    --model resnet50 \
    --image-size 224 \
    --camera 0 \
    --device cuda \
    --use-mtcnn \
    --to-rgb
```

## GUI Controls

- **Display**: Real-time video with detected faces
- **Emotion Labels**: Shows predicted emotion + confidence score
- **Press 'Q'**: Quit the application

Example output:
```
Happy: 0.92
Sad: 0.05
Neutral: 0.03
```

## Evaluation on Test Set

After training, evaluate on the test split:

```bash
python -m src.eval \
    --checkpoint outputs/gpu_training/best.pth \
    --data-dir data/fer2013 \
    --csv fer2013.csv \
    --model resnet50 \
    --split PublicTest \
    --output-dir outputs/gpu_training \
    --device cuda
```

This generates a confusion matrix visualization.

## Troubleshooting

### GPU Not Detected

```bash
# Check CUDA installation
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory (OOM)

Reduce batch size:
```bash
--batch-size 32  # or 16
```

Or use gradient accumulation in the code.

### Slow Training

- Increase `--num-workers` (e.g., 8) if available
- Use `--pin-memory` (already enabled in scripts)
- Ensure no other processes use GPU: `nvidia-smi`

### Camera Not Found

```bash
# Check available cameras
ls /dev/video*

# Use different camera index
bash test_gui.sh
# Then modify script to use --camera 1 or --camera 2
```

## Expected Performance

On FER2013 with ResNet50 (30 epochs):
- **Training Accuracy**: ~70-75%
- **Validation Accuracy**: ~65-70%
- **Test Accuracy**: ~63-68%

Performance depends on:
- Data augmentation
- Learning rate schedule
- Batch size
- GPU VRAM (affects batch processing)

## Files Generated

After training:
```
outputs/gpu_training/
├── best.pth                    # Best model weights
├── checkpoint_epoch_1.pth      # Checkpoint after epoch 1
├── checkpoint_epoch_2.pth      # ... etc
└── confusion_PublicTest.png    # Confusion matrix (if evaluated)
```

## Next Steps

1. ✅ Train model with full FER2013 data
2. ✅ Test with GUI on webcam
3. 📊 Evaluate on test set (PublicTest/PrivateTest)
4. 🔄 Experiment with:
   - Different architectures (EfficientNet, MobileNet)
   - Hyperparameters (lr, batch_size, epochs)
   - Data augmentation settings
   - Freeze/unfreeze backbone layers

## Quick Commands

```bash
# Train
bash train_gpu.sh

# Test GUI
bash test_gui.sh

# Evaluate
python -m src.eval --checkpoint outputs/gpu_training/best.pth --model resnet50 --device cuda

# Check GPU stats during training
watch -n 1 nvidia-smi
```

Good luck! 🚀
