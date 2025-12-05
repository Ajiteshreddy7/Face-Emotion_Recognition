#!/usr/bin/env python
"""
GPU Training & GUI Test Manager
Simplified interface for training on GPU and testing with GUI
"""

import argparse
import subprocess
import sys
import os
import torch


def check_gpu():
    """Check GPU availability"""
    if not torch.cuda.is_available():
        print("❌ GPU NOT AVAILABLE")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        print(f"   CUDA Version: {torch.version.cuda}")
        sys.exit(1)
    
    print("✅ GPU AVAILABLE")
    print(f"   Device Count: {torch.cuda.device_count()}")
    print(f"   Device Name: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   PyTorch Version: {torch.__version__}")
    print()


def check_data():
    """Check if training data exists"""
    csv_path = "data/fer2013/fer2013.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Data not found at {csv_path}")
        sys.exit(1)
    print(f"✅ Data found at {csv_path}")
    print()


def train(args):
    """Run training"""
    print("🚀 Starting Training on GPU...")
    print()
    
    cmd = [
        "python", "-m", "src.train",
        "--data-dir", args.data_dir,
        "--csv", args.csv,
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--image-size", str(args.image_size),
        "--lr", str(args.lr),
        "--output-dir", args.output_dir,
        "--device", "cuda",
        "--num-workers", str(args.num_workers),
    ]
    
    if args.pin_memory:
        cmd.append("--pin-memory")
    if args.to_rgb:
        cmd.append("--to-rgb")
    
    try:
        subprocess.run(cmd, check=True)
        print()
        print("✅ Training Complete!")
        print(f"   Best model: {args.output_dir}/best.pth")
        print()
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        sys.exit(1)


def test_gui(args):
    """Run GUI test"""
    checkpoint = args.checkpoint
    
    if not os.path.exists(checkpoint):
        print(f"❌ Checkpoint not found at {checkpoint}")
        print("   Please train the model first using: python run_training.py train")
        sys.exit(1)
    
    print(f"🎥 Starting GUI Test with checkpoint: {checkpoint}")
    print("   Press 'Q' to quit")
    print()
    
    cmd = [
        "python", "-m", "src.gui",
        "--checkpoint", checkpoint,
        "--model", args.model,
        "--image-size", str(args.image_size),
        "--camera", str(args.camera),
        "--device", "cuda",
        "--use-mtcnn",
        "--to-rgb",
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print()
        print("✅ GUI Test Complete!")
        print()
    except subprocess.CalledProcessError as e:
        print(f"❌ GUI test failed with error: {e}")
        sys.exit(1)


def evaluate(args):
    """Run evaluation"""
    checkpoint = args.checkpoint
    
    if not os.path.exists(checkpoint):
        print(f"❌ Checkpoint not found at {checkpoint}")
        sys.exit(1)
    
    print(f"📊 Starting Evaluation with checkpoint: {checkpoint}")
    print()
    
    cmd = [
        "python", "-m", "src.eval",
        "--checkpoint", checkpoint,
        "--data-dir", args.data_dir,
        "--csv", args.csv,
        "--model", args.model,
        "--split", args.split,
        "--output-dir", args.output_dir,
        "--device", "cuda",
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print()
        print("✅ Evaluation Complete!")
        print(f"   Confusion matrix: {args.output_dir}/confusion_{args.split}.png")
        print()
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed with error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="GPU Training & GUI Test Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check GPU and data
  python run_training.py check
  
  # Train with full FER2013 data
  python run_training.py train
  
  # Train with subset for quick test
  python run_training.py train --csv subset_2000_500_500.csv --epochs 5
  
  # Test GUI with best model
  python run_training.py test-gui
  
  # Evaluate model on test set
  python run_training.py evaluate
        """)
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Check command
    subparsers.add_parser('check', help='Check GPU and data availability')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model on GPU')
    train_parser.add_argument('--data-dir', default='data/fer2013', help='Data directory')
    train_parser.add_argument('--csv', default='fer2013.csv', help='CSV filename')
    train_parser.add_argument('--model', default='resnet50', help='Model name')
    train_parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    train_parser.add_argument('--image-size', type=int, default=224, help='Image size')
    train_parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    train_parser.add_argument('--output-dir', default='outputs/gpu_training', help='Output directory')
    train_parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    train_parser.add_argument('--pin-memory', action='store_true', default=True, help='Pin memory')
    train_parser.add_argument('--to-rgb', action='store_true', default=True, help='Convert to RGB')
    
    # Test GUI command
    test_parser = subparsers.add_parser('test-gui', help='Test model with GUI')
    test_parser.add_argument('--checkpoint', default='outputs/gpu_training/best.pth', help='Model checkpoint')
    test_parser.add_argument('--model', default='resnet50', help='Model name')
    test_parser.add_argument('--image-size', type=int, default=224, help='Image size')
    test_parser.add_argument('--camera', type=int, default=0, help='Camera index')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model on test set')
    eval_parser.add_argument('--checkpoint', default='outputs/gpu_training/best.pth', help='Model checkpoint')
    eval_parser.add_argument('--data-dir', default='data/fer2013', help='Data directory')
    eval_parser.add_argument('--csv', default='fer2013.csv', help='CSV filename')
    eval_parser.add_argument('--model', default='resnet50', help='Model name')
    eval_parser.add_argument('--split', default='PublicTest', choices=['PublicTest', 'PrivateTest'], help='Test split')
    eval_parser.add_argument('--output-dir', default='outputs/gpu_training', help='Output directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    print("=" * 50)
    print("Face Emotion Recognition - GPU Training Manager")
    print("=" * 50)
    print()
    
    if args.command == 'check':
        check_gpu()
        check_data()
        print("✅ All checks passed!")
        print()
    
    elif args.command == 'train':
        check_gpu()
        check_data()
        train(args)
    
    elif args.command == 'test-gui':
        check_gpu()
        test_gui(args)
    
    elif args.command == 'evaluate':
        check_gpu()
        evaluate(args)


if __name__ == '__main__':
    main()
