# Real-Time English Sign Language Translator

This project uses PyTorch and OpenCV to recognize American Sign Language (ASL) hand signs from a live camera feed and translates them into readable text in real time.

## Features
- Real-time hand sign recognition using webcam
- PyTorch-based deep learning model
- User-friendly interface displaying recognized words and sentences

## Setup
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) and place it in `Arabic_letters_in_sign_language/data/`.

## Usage
- To train the model:
  ```bash
  python train.py
  ```
- To run real-time prediction:
  ```bash
  python predict.py
  ```

## Directory Structure
```
Arabic_letters_in_sign_language/
  - data/           # Dataset
  - models/         # Model code
  - outputs/        # Saved models, logs
  - utils/          # Helper scripts
  - train.py        # Training script
  - predict.py      # Real-time inference
  - requirements.txt
  - README.md
```

## License
MIT 