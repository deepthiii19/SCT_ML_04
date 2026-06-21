# Hand Gesture Recognition

A hand gesture recognition system that identifies and classifies hand gestures from image and video data using **MediaPipe** hand landmark detection and a **scikit-learn** classifier — enabling intuitive human-computer interaction and gesture-based control.

> Task 04 — SkillCraft Technology Machine Learning Internship



## Overview

Instead of training a CNN directly on raw images, this project uses **MediaPipe Hands** to extract 21 hand keypoints (x, y, z) per image, normalizes them (translation + scale invariant), and trains a lightweight classifier (MLP / Random Forest) on those landmark vectors. This approach:

- Generalizes well with a few hundred images per gesture class
- Is invariant to background and lighting differences
- Runs in real time on CPU — no GPU required
- Uses the same landmark extractor for both training (static images) and live inference (webcam video)

## Dataset

Trained on the [LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) dataset — 10 gesture classes (palm, fist, thumb, index, ok, etc.), ~20,000 grayscale images captured from 10 subjects.

## Project Structure
SCT_ML_04/

├── leapGestRecog/         

├── data/                   # Extracted landmark CSV (generated

├── models/                 # Trained model + label encoder (generated)

├── screenshots/            # Demo screenshots

├── extract_landmarks.py    # Extracts MediaPipe hand landmarks from dataset images

├── train_model.py          # Trains classifier on extracted landmarks

├── realtime_inference.py   # Live webcam gesture recognition

├── requirements.txt


## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Extract landmarks from the dataset

```bash
python extract_landmarks.py --dataset leapGestRecog --out data/landmarks.csv
```

### 2. Train the classifier

```bash
python train_model.py --csv data/landmarks.csv
```

Trains both an MLP and a Random Forest, prints accuracy reports, and saves the better-performing model to `models/gesture_model.pkl`.

### 3. Run real-time webcam inference

```bash
python realtime_inference.py
```

Opens your webcam, detects your hand, and overlays the predicted gesture, confidence score, and FPS in real time. Press `q` to quit.

## Tech Stack

- **MediaPipe** — hand landmark detection
- **OpenCV** — image/video processing
- **scikit-learn** — MLP / Random Forest classifier
- **pandas / numpy** — data handling

## Results

The model achieves strong accuracy across all 10 gesture classes on the held-out test split (see training output above for exact metrics).
