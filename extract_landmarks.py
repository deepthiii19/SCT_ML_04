"""
extract_landmarks.py

Walks through a dataset folder structured as:

    dataset/
        gesture_1/
            img001.jpg
            img002.jpg
            ...
        gesture_2/
            img001.jpg
            ...
        ...

For every image, runs MediaPipe Hands to extract 21 (x, y, z) landmarks,
normalizes them (relative to the wrist + scale), and writes everything to
a CSV file: data/landmarks.csv

Usage:
    python extract_landmarks.py --dataset path/to/dataset --out data/landmarks.csv
"""

import argparse
import csv
import os

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands


def normalize_landmarks(landmarks):
    """
    landmarks: list of 21 (x, y, z) tuples (already in image-normalized 0-1 coords
    from MediaPipe). We re-normalize relative to the wrist point and scale by the
    max distance, so the features are translation- and scale-invariant.
    """
    pts = np.array(landmarks, dtype=np.float32)  # shape (21, 3)
    wrist = pts[0].copy()
    pts -= wrist  # translate so wrist is origin

    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 1e-6:
        pts /= max_dist  # scale invariant

    return pts.flatten().tolist()  # 63 values


def extract_from_dataset(dataset_dir, out_csv, max_hands=1, min_detection_conf=0.6):
    classes = sorted(
        [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    )
    print(f"Found {len(classes)} classes: {classes}")

    rows = []
    header = [f"f{i}" for i in range(63)] + ["label"]

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=max_hands,
        min_detection_confidence=min_detection_conf,
    ) as hands:
        for label in classes:
            class_dir = os.path.join(dataset_dir, label)
            image_files = [
                f for f in os.listdir(class_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]
            print(f"  Processing '{label}': {len(image_files)} images")

            detected = 0
            for fname in image_files:
                fpath = os.path.join(class_dir, fname)
                img = cv2.imread(fpath)
                if img is None:
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    feats = normalize_landmarks(coords)
                    rows.append(feats + [label])
                    detected += 1

            print(f"    -> hand detected in {detected}/{len(image_files)} images")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
"""
extract_landmarks.py

Walks through a dataset folder structured as:

    dataset/
        gesture_1/
            img001.jpg
            img002.jpg
            ...
        gesture_2/
            img001.jpg
            ...
        ...

For every image, runs MediaPipe Hands to extract 21 (x, y, z) landmarks,
normalizes them (relative to the wrist + scale), and writes everything to
a CSV file: data/landmarks.csv

Usage:
    python extract_landmarks.py --dataset path/to/dataset --out data/landmarks.csv
"""

import argparse
import csv
import os

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands


def normalize_landmarks(landmarks):
    """
    landmarks: list of 21 (x, y, z) tuples (already in image-normalized 0-1 coords
    from MediaPipe). We re-normalize relative to the wrist point and scale by the
    max distance, so the features are translation- and scale-invariant.
    """
    pts = np.array(landmarks, dtype=np.float32)  # shape (21, 3)
    wrist = pts[0].copy()
    pts -= wrist  # translate so wrist is origin

    max_dist = np.max(np.linalg.norm(pts, axis=1))
    if max_dist > 1e-6:
        pts /= max_dist  # scale invariant

    return pts.flatten().tolist()  # 63 values


def clean_label(folder_name):
    """
    Strips a leading numeric prefix like '01_palm' -> 'palm', '10_down' -> 'down'.
    Leaves already-clean names untouched.
    """
    parts = folder_name.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return parts[1]
    return folder_name


def find_gesture_folders(dataset_dir):
    """
    Auto-detects dataset layout and returns a dict: {gesture_label: [list_of_dirs]}

    Supports two layouts:
      1. Flat:    dataset/<gesture>/*.jpg
      2. Nested:  dataset/<subject>/<gesture>/*.jpg   (e.g. LeapGestRecog: 00/01_palm/...)

    For the nested layout, all subject folders contribute to the same gesture label,
    so e.g. 00/01_palm and 01/01_palm both map to label "palm".
    """
    top_level = sorted(
        d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))
    )
    if not top_level:
        return {}

    gesture_dirs = {}

    first_dir = os.path.join(dataset_dir, top_level[0])
    has_images_directly = any(
        f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        for f in os.listdir(first_dir)
    )

    if has_images_directly:
        for d in top_level:
            label = clean_label(d)
            gesture_dirs.setdefault(label, []).append(os.path.join(dataset_dir, d))
    else:
        for subject in top_level:
            subject_path = os.path.join(dataset_dir, subject)
            sub_dirs = sorted(
                d for d in os.listdir(subject_path)
                if os.path.isdir(os.path.join(subject_path, d))
            )
            for d in sub_dirs:
                label = clean_label(d)
                gesture_dirs.setdefault(label, []).append(os.path.join(subject_path, d))

    return gesture_dirs


def extract_from_dataset(dataset_dir, out_csv, max_hands=1, min_detection_conf=0.6):
    gesture_dirs = find_gesture_folders(dataset_dir)
    classes = sorted(gesture_dirs.keys())
    print(f"Found {len(classes)} classes: {classes}")

    rows = []
    header = [f"f{i}" for i in range(63)] + ["label"]

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=max_hands,
        min_detection_confidence=min_detection_conf,
    ) as hands:
        for label in classes:
            all_dirs = gesture_dirs[label]
            image_files = []
            for d in all_dirs:
                for f in os.listdir(d):
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        image_files.append(os.path.join(d, f))

            print(f"  Processing '{label}': {len(image_files)} images")

            detected = 0
            for fpath in image_files:
                img = cv2.imread(fpath)
                if img is None:
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    feats = normalize_landmarks(coords)
                    rows.append(feats + [label])
                    detected += 1

            print(f"    -> hand detected in {detected}/{len(image_files)} images")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} samples to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to dataset root (class subfolders)")
    parser.add_argument("--out", default="data/landmarks.csv", help="Output CSV path")
    args = parser.parse_args()

    extract_from_dataset(args.dataset, args.out)