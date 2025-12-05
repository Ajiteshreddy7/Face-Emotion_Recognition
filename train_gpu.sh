#!/bin/bash
# GPU Training Script for Face Emotion Recognition
# This script trains the ResNet50 model on your local GPU

set -e

# Configuration
DATA_DIR="data/fer2013"
CSV_FILE="fer2013.csv"
MODEL_NAME="resnet50"
EPOCHS=30
BATCH_SIZE=64
IMAGE_SIZE=224
LR=3e-4
OUTPUT_DIR="outputs/gpu_training"
DEVICE="cuda"

echo "=========================================="
echo "Face Emotion Recognition - GPU Training"
echo "=========================================="
echo "Configuration:"
echo "  Data Directory: $DATA_DIR"
echo "  CSV File: $CSV_FILE"
echo "  Model: $MODEL_NAME"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Image Size: $IMAGE_SIZE"
echo "  Learning Rate: $LR"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Device: $DEVICE"
echo "=========================================="

# Check if data exists
if [ ! -f "$DATA_DIR/$CSV_FILE" ]; then
    echo "ERROR: Data file not found at $DATA_DIR/$CSV_FILE"
    exit 1
fi

echo "Starting training..."
python -m src.train \
    --data-dir "$DATA_DIR" \
    --csv "$CSV_FILE" \
    --model "$MODEL_NAME" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --image-size "$IMAGE_SIZE" \
    --lr "$LR" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --num-workers 4 \
    --pin-memory \
    --to-rgb

echo "=========================================="
echo "Training Complete!"
echo "Best model saved to: $OUTPUT_DIR/best.pth"
echo "=========================================="
