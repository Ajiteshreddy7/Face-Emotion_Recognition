# Facial Emotion Recognition System

A deep learning-based system for real-time facial emotion detection using CNNs trained on the FER2013 dataset.

## 📊 Project Overview

This project implements a facial emotion recognition system capable of detecting 7 different emotions:
- **Angry** 😠
- **Disgust** 🤢
- **Fear** 😨
- **Happy** 😊
- **Neutral** 😐
- **Sad** 😢
- **Surprise** 😮

## Demo
![Emotion Recognition Demo](https://raw.githubusercontent.com/Ajiteshreddy7/Face-Emotion_Recognition/main/utils/Demo.gif)

## 🏗️ System Architecture

### Dataset
- **Source**: FER2013 (Facial Expression Recognition 2013)
- **Training samples**: 28,709 images
- **Test samples**: 7,178 images
- **Image format**: 48×48 grayscale
- **Classes**: 7 emotion categories
- **Organization**: `data/emotions/train/[emotion]/` and `data/emotions/test/[emotion]/`

### Models

#### Model 1: Advanced CNN (62% Accuracy)
**Architecture:**
```
Input (48×48×1)
├─ Block 1: Conv32 → Conv32 → Pool → Dropout(0.25)
├─ Block 2: Conv64 → Conv64 → Pool → Dropout(0.25)
├─ Block 3: Conv128 → Conv128 → Pool → Dropout(0.25)
└─ Dense: Flatten → FC512 → Dropout(0.5) → FC256 → Dropout(0.5) → Output(7)
```
- **Parameters**: 685,159
- **Training**: Simple augmentation (rotation, zoom, horizontal flip)
- **File**: `saved_models/emotion_model.h5`

#### Model 2: Improved CNN (68% Accuracy) ⭐
**Architecture:**
```
Input (48×48×1)
├─ Block 1: Conv64 → Conv64 → Pool → Dropout(0.25)
├─ Block 2: Conv128 → Conv128 → Pool → Dropout(0.25)
├─ Block 3: Conv256 → Conv256 → Pool → Dropout(0.3)
├─ Block 4: Conv512 → Pool → Dropout(0.4)
└─ Dense: GlobalAveragePooling → FC512 → Dropout(0.5) → FC256 → Dropout(0.5) → Output(7)
```
- **Parameters**: 2,728,903 (4× larger)
- **Training**: Moderate augmentation + validation split + LR scheduling
- **File**: `saved_models/emotion_model_improved.h5`

### Key Differences: 62% vs 68% Model

| Feature | 62% Model | 68% Model |
|---------|-----------|-----------|
| Depth | 3 blocks | 4 blocks |
| Max filters | 128 | 512 |
| Parameters | 685K | 2.7M |
| Pooling | Flatten | GlobalAveragePooling |
| Validation split | No | Yes (10%) |
| LR scheduling | Fixed | Adaptive |
| Accuracy | 62% | 68% |

## 📁 Project Structure

```
Face-Emotion_Recognition/
├── data/
│   └── emotions/
│       ├── train/          # Training data (28,709 images)
│       │   ├── angry/
│       │   ├── disgust/
│       │   ├── fear/
│       │   ├── happy/
│       │   ├── neutral/
│       │   ├── sad/
│       │   └── surprise/
│       └── test/           # Test data (7,178 images)
│           └── [same structure]
├── models/
│   └── cnn_models.py              # CNN architecture definitions
├── utils/
│   └── data_loader.py             # Dataset loading utilities
├── saved_models/
│   ├── emotion_model.h5                    # 62% accuracy model
│   ├── emotion_model_improved.h5           # 68% accuracy model ⭐
│   ├── emotion_model_62percent_backup.h5   # Backup
│   └── [other experimental models]
├── train_simple.py                # Train 62% model
├── train_improved.py              # Train 68% model
├── predict_realtime.py            # Real-time demo (62%)
├── predict_realtime_improved.py   # Real-time demo (68%) ⭐
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running Real-time Demo

**With 68% improved model (recommended):**
```bash
python predict_realtime_improved.py
```

**With 62% original model:**
```bash
python predict_realtime.py
```

Press **'q'** in the video window to quit.

### Training Models

**Train 62% model (~20 minutes):**
```bash
python train_simple.py
```

**Train 68% improved model (~45 minutes):**
```bash
python train_improved.py --epochs 60
```

## 📊 Data Pipeline

### Loading & Preprocessing
1. Load images from folder structure (train/test)
2. Convert to grayscale (48×48)
3. Normalize pixel values (0-1)
4. Create emotion labels (0-6)

### Data Augmentation
- Rotation: ±20°
- Zoom: ±15%
- Shift: ±15%
- Horizontal flip
- Brightness: 0.7-1.3×

### Training Configuration (68% model)
```python
Batch size: 64
Optimizer: Adam (lr=0.001)
Loss: Sparse Categorical Crossentropy
Max epochs: 60
Early stopping: patience=20
LR reduction: 0.5× on plateau
Validation split: 10%
```

## 🎯 Performance

| Model | Accuracy | Parameters | Size | Time |
|-------|----------|-----------|------|------|
| 62% | 62.5% | 685K | 8.3 MB | 20 min |
| 68% | 67.9% | 2.7M | 10.4 MB | 45 min |

## 🔍 Architecture Details

### Convolutional Blocks
- Progressive channel expansion (64→128→256→512)
- Batch normalization after each conv
- ReLU activation functions
- Max pooling (2×2)
- Increasing dropout (0.25→0.4)

### Pooling Strategy
- **62% model**: Flatten → Dense layers
- **68% model**: GlobalAveragePooling → Dense layers
  - More robust to spatial variations
  - Reduces overfitting

### Regularization
- Batch normalization: Stabilizes training
- Dropout: Prevents overfitting (0.25-0.5)
- Learning rate reduction: Fine-tunes in later epochs

## 📝 Model Comparison

### Why 68% is Better
1. **Deeper architecture**: 4 blocks vs 3
2. **More feature maps**: 512 vs 128 max
3. **Better pooling**: GlobalAveragePooling vs Flatten
4. **Smarter training**: Validation split + LR scheduling
5. **6.4% accuracy improvement**: 62% → 68%

### When to Use Each
- **62% model**: Fast inference, lower memory, quick prototyping
- **68% model**: Better accuracy, production systems, modern GPUs

## ⚙️ Customization

### To improve accuracy:
- Increase network depth
- Add more filters
- Use transfer learning (VGG, ResNet)
- Ensemble multiple models
- Collect more data

### To speed up training:
- Reduce batch size
- Lower image resolution
- Fewer filters/blocks
- GPU acceleration

## 📦 Requirements

- TensorFlow >= 2.10.0
- OpenCV (cv2)
- NumPy
- Scikit-learn

See `requirements.txt` for versions.

## 🎓 References

- **FER2013 Dataset**: https://www.kaggle.com/datasets/msambare/fer2013
- **CNN Fundamentals**: https://arxiv.org/abs/1409.1556
- **Batch Normalization**: https://arxiv.org/abs/1502.03167

## 📄 File Descriptions

### Core Files
- `models/cnn_models.py`: Defines `create_advanced_cnn()` for 62% model
- `utils/data_loader.py`: Dataset loading and preprocessing

### Training Scripts
- `train_simple.py`: Quick training (62%)
- `train_improved.py`: Enhanced training with validation (68%)

### Inference Scripts
- `predict_realtime.py`: Webcam demo using 62% model
- `predict_realtime_improved.py`: Webcam demo using 68% model

## 📄 Project Structure

```
Face-Emotion_Recognition/
├── data/
│   └── emotions/
│       ├── train/          # Training data (28,709 images)
│       │   ├── angry/
│       │   ├── disgust/
│       │   ├── fear/
│       │   ├── happy/
│       │   ├── neutral/
│       │   ├── sad/
│       │   └── surprise/
│       └── test/           # Test data (7,178 images)
│           └── [same structure]
├── models/
│   └── cnn_models.py              # CNN architecture definitions
├── utils/
│   └── data_loader.py             # Dataset loading utilities
├── saved_models/
│   ├── emotion_model.h5                    # 62% accuracy model
│   ├── emotion_model_improved.h5           # 68% accuracy model ⭐
│   ├── emotion_model_62percent_backup.h5   # Backup
│   └── [other experimental models]
├── train_simple.py                # Train 62% model
├── train_improved.py              # Train 68% model
├── predict_realtime.py            # Real-time demo (62%)
├── predict_realtime_improved.py   # Real-time demo (68%) ⭐
├── requirements.txt
└── README.md
```

---

**Best Model**: `emotion_model_improved.h5` (68% accuracy)  
**Quick Start**: `python predict_realtime_improved.py`  
**Last Updated**: December 8, 2025
│           ├── disgust/
│           ├── fear/
│           ├── happy/
│           ├── neutral/
│           ├── sad/
│           └── surprise/
├── models/
│   └── cnn_models.py
├── utils/
│   └── data_loader.py
├── train.py
├── predict_realtime.py
└── saved_models/
```

## 🚀 How to Use

### 1. Add Your Data
Place your emotion images in the appropriate folders:
- Training images: `data/emotions/train/[emotion_name]/`
- Test images: `data/emotions/test/[emotion_name]/`

### 2. Train the Model
```bash
python train.py
```

### 3. Real-time Detection
```bash
python predict_realtime.py
```

## 📊 Features

- ✅ Simple and clean code structure
- ✅ CNN models (simple and advanced)
- ✅ Data augmentation
- ✅ Real-time webcam detection
- ✅ 7 emotion classes: angry, disgust, fear, happy, neutral, sad, surprise

## 📋 Requirements

```bash
pip install tensorflow opencv-python numpy scikit-learn matplotlib
```

That's it! Simple and ready to use. 🎭
