@echo off
REM GPU Training Script for Face Emotion Recognition (Windows)
REM This script trains the ResNet50 model on your local GPU

setlocal enabledelayedexpansion

REM Configuration
set "DATA_DIR=data\fer2013"
set "CSV_FILE=fer2013.csv"
set "MODEL_NAME=resnet50"
set "EPOCHS=30"
set "BATCH_SIZE=64"
set "IMAGE_SIZE=224"
set "LR=3e-4"
set "OUTPUT_DIR=outputs\gpu_training"
set "DEVICE=cuda"

echo.
echo ==========================================
echo Face Emotion Recognition - GPU Training
echo ==========================================
echo Configuration:
echo   Data Directory: %DATA_DIR%
echo   CSV File: %CSV_FILE%
echo   Model: %MODEL_NAME%
echo   Epochs: %EPOCHS%
echo   Batch Size: %BATCH_SIZE%
echo   Image Size: %IMAGE_SIZE%
echo   Learning Rate: %LR%
echo   Output Directory: %OUTPUT_DIR%
echo   Device: %DEVICE%
echo ==========================================
echo.

REM Check if data exists
if not exist "%DATA_DIR%\%CSV_FILE%" (
    echo ERROR: Data file not found at %DATA_DIR%\%CSV_FILE%
    exit /b 1
)

echo Starting training...
python -m src.train ^
    --data-dir %DATA_DIR% ^
    --csv %CSV_FILE% ^
    --model %MODEL_NAME% ^
    --epochs %EPOCHS% ^
    --batch-size %BATCH_SIZE% ^
    --image-size %IMAGE_SIZE% ^
    --lr %LR% ^
    --output-dir %OUTPUT_DIR% ^
    --device %DEVICE% ^
    --num-workers 4 ^
    --pin-memory ^
    --to-rgb

echo.
echo ==========================================
echo Training Complete!
echo Best model saved to: %OUTPUT_DIR%\best.pth
echo ==========================================
echo.
pause
