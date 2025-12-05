@echo off
REM GUI Test Script for Face Emotion Recognition (Windows)
REM This script tests the trained model on your webcam

setlocal enabledelayedexpansion

REM Configuration
set "CHECKPOINT=outputs\gpu_training\best.pth"
if not "%1"=="" set "CHECKPOINT=%1"
set "MODEL_NAME=resnet50"
set "IMAGE_SIZE=224"
set "CAMERA=0"
set "DEVICE=cuda"

echo.
echo ==========================================
echo Face Emotion Recognition - GUI Test
echo ==========================================
echo Configuration:
echo   Checkpoint: %CHECKPOINT%
echo   Model: %MODEL_NAME%
echo   Image Size: %IMAGE_SIZE%
echo   Camera: %CAMERA%
echo   Device: %DEVICE%
echo ==========================================
echo.

REM Check if checkpoint exists
if not exist "%CHECKPOINT%" (
    echo ERROR: Checkpoint not found at %CHECKPOINT%
    echo Please train the model first using: train_gpu.bat
    pause
    exit /b 1
)

echo Starting GUI test...
echo Press 'Q' to quit.
echo.

python -m src.gui ^
    --checkpoint %CHECKPOINT% ^
    --model %MODEL_NAME% ^
    --image-size %IMAGE_SIZE% ^
    --camera %CAMERA% ^
    --device %DEVICE% ^
    --use-mtcnn ^
    --to-rgb

echo.
echo ==========================================
echo GUI Test Complete!
echo ==========================================
echo.
pause
