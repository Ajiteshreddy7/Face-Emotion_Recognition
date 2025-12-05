#!/bin/bash
# GUI Test Script for Face Emotion Recognition
# This script tests the trained model on your webcam

set -e

# Configuration
CHECKPOINT="${1:-outputs/gpu_training/best.pth}"
MODEL_NAME="resnet50"
IMAGE_SIZE=224
CAMERA=0
DEVICE="cuda"

echo "=========================================="
echo "Face Emotion Recognition - GUI Test"
echo "=========================================="
echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Model: $MODEL_NAME"
echo "  Image Size: $IMAGE_SIZE"
echo "  Camera: /dev/video$CAMERA"
echo "  Device: $DEVICE"
echo "=========================================="

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "Please train the model first using: bash train_gpu.sh"
    exit 1
fi

echo "Starting GUI test..."
echo "Press 'Q' to quit."
echo ""

python -m src.gui \
    --checkpoint "$CHECKPOINT" \
    --model "$MODEL_NAME" \
    --image-size "$IMAGE_SIZE" \
    --camera "$CAMERA" \
    --device "$DEVICE" \
    --use-mtcnn \
    --to-rgb

echo "=========================================="
echo "GUI Test Complete!"
echo "=========================================="
