import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from models.asl_cnn import get_model

# Settings
MODEL_PATH = 'Arabic_letters_in_sign_language/outputs/asl_cnn_best.pth'
DATA_DIR = 'Arabic_letters_in_sign_language/data/asl_alphabet_train'
IMG_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load class names from the dataset folder
from torchvision.datasets import ImageFolder
class_names = ImageFolder(DATA_DIR).classes

# Load model
model = get_model(num_classes=len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
        return class_names[pred.item()]

def main():
    cap = cv2.VideoCapture(0)
    sentence = ""
    last_pred = ""
    pred_count = 0
    PRED_THRESHOLD = 10  # Number of consistent frames before adding to sentence
    print("Press 'c' to clear sentence, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Define region of interest (ROI) for hand
        h, w, _ = frame.shape
        x1, y1, x2, y2 = w//2-100, h//2-100, w//2+100, h//2+100
        roi = frame[y1:y2, x1:x2]
        if roi.shape[0] == 0 or roi.shape[1] == 0:
            continue
        pred = predict(roi)
        # Only add to sentence if prediction is stable
        if pred == last_pred:
            pred_count += 1
        else:
            pred_count = 0
        if pred_count == PRED_THRESHOLD:
            if pred == 'space':
                sentence += ' '
            elif pred == 'delete':
                sentence = sentence[:-1]
            elif pred != 'nothing':
                sentence += pred.upper()
            pred_count = 0
        last_pred = pred
        # Draw ROI and text
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f'Pred: {pred}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(frame, f'Sentence: {sentence}', (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow('ASL Translator', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c'):
            sentence = ""
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 