import argparse
import cv2
import torch
import numpy as np
from PIL import Image

from facenet_pytorch import MTCNN
from .models import get_model
from .data import get_transforms


EMOTION_LABELS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (best.pth)')
    p.add_argument('--model', type=str, default='resnet50')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--to-rgb', action='store_true', default=True)
    p.add_argument('--use-mtcnn', action='store_true', default=True, help='Use MTCNN to detect faces')
    p.add_argument('--threshold', type=float, default=0.5, help='MTCNN detection threshold')
    return p.parse_args()


def preprocess_face(face_bgr, image_size, to_rgb=True):
    # face_bgr: numpy array (H,W,3) in BGR (from OpenCV)
    if to_rgb:
        face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    else:
        face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        face = np.stack([face] * 3, axis=-1)  # make 3-channel for model

    pil = Image.fromarray(face)
    _, val_t = get_transforms(image_size=image_size)
    augmented = val_t(image=np.array(pil))
    tensor = augmented['image'].unsqueeze(0)  # add batch dim
    return tensor


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load model
    model = get_model(name=args.model, num_classes=7, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    # handle both checkpoint formats
    state = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Prepare MTCNN for face detection
    mtcnn = MTCNN(keep_all=True, device=device) if args.use_mtcnn else None

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print('Cannot open camera')
        return

    print('Starting camera. Press Q to quit.')
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()

            face_crops = []
            boxes = None
            if mtcnn is not None:
                # MTCNN expects RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes, probs = mtcnn.detect(rgb)
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = [int(max(0, v)) for v in box]
                        # crop with a small margin
                        h, w = frame.shape[:2]
                        pad = int(0.1 * max(x2 - x1, y2 - y1))
                        x1 = max(0, x1 - pad)
                        y1 = max(0, y1 - pad)
                        x2 = min(w, x2 + pad)
                        y2 = min(h, y2 + pad)
                        crop = frame[y1:y2, x1:x2]
                        face_crops.append((crop, (x1, y1, x2, y2), probs[i] if probs is not None else None))
            else:
                # fallback: whole frame as face
                h, w = frame.shape[:2]
                face_crops.append((frame, (0, 0, w, h), None))

            # Run inference on each crop
            for crop, (x1, y1, x2, y2), score in face_crops:
                try:
                    inp = preprocess_face(crop, image_size=args.image_size, to_rgb=args.to_rgb)
                    inp = inp.to(device)
                    out = model(inp)
                    probs = torch.softmax(out, dim=1)[0].cpu().numpy()
                    pred = int(probs.argmax())
                    conf = float(probs[pred])
                    label = f"{EMOTION_LABELS.get(pred, str(pred))}: {conf:.2f}"
                    # draw box and label
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                except Exception as e:
                    cv2.putText(display, 'Error', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow('FER Live', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
