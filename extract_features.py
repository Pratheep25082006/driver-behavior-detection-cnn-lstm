"""
extract_features.py
--------------------
Run AFTER train_cnn.py.

Loads the trained CNN, cuts it at the 'feature_vector' layer, and runs every
frame through it. Frames are grouped by video (subject + class + angle) and
sorted so the output is a proper time-ordered sequence - exactly what an
LSTM needs in Review 2.

Output: one .npz file per video inside features/, each containing:
    features : shape (num_frames, FEATURE_DIM)
    label    : the class index for that video
    key      : the video identifier string (e.g. "Harini_Sleeping_front")

Also writes features/manifest.csv listing every video key, its label, and
its saved .npz path, so the LSTM training script can load everything easily.
"""

import os
import re
import glob
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import models

DATA_DIR = "Frames_Clean_224"
IMG_SIZE = (224, 224)
CLASSES = ["Distraction", "Drinking", "Normal", "Phone", "Sleeping"]
MODEL_PATH = "cnn_full_model.keras"
OUT_DIR = "features"

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------- Load trained CNN, strip head ----------------------
full_model = models.load_model(MODEL_PATH)
feature_extractor = models.Model(
    inputs=full_model.input,
    outputs=full_model.get_layer("feature_vector").output,
)
feature_extractor.trainable = False


# ----------------------- Group frames into videos --------------------------
def get_video_key(filepath):
    """
    Must match the grouping logic used in train_cnn.py.
    Matches the extraction script's naming convention:
        {video_stem}_{framenumber:06d}.jpg   e.g. front_000001.jpg, 2_000005.jpg
    """
    parts = filepath.replace("\\", "/").split("/")
    subj, cls = parts[-3], parts[-2]
    fname = os.path.splitext(os.path.basename(filepath))[0]
    angle, _, _ = fname.rpartition("_")
    if not angle:
        angle = fname
    return f"{subj}_{cls}_{angle}"


def frame_sort_key(filepath):
    """Extracts the trailing number in the filename so frames sort in time order."""
    nums = re.findall(r"\d+", os.path.basename(filepath))
    return int(nums[-1]) if nums else 0


def load_and_preprocess(filepath):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)  # handles jpg AND png
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img


# Collect all frames, grouped by video key
video_frames = {}  # key -> list of filepaths
video_labels = {}  # key -> class index

all_files = glob.glob(os.path.join(DATA_DIR, "*", "*", "**", "*.*"), recursive=True)
all_files = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

for f in all_files:
    cls_name = f.replace("\\", "/").split("/")[-2]
    if cls_name not in CLASSES:
        continue
    key = get_video_key(f)
    video_frames.setdefault(key, []).append(f)
    video_labels[key] = CLASSES.index(cls_name)

print(f"Found {len(video_frames)} videos across {len(all_files)} frames")

# ----------------------- Extract + save per video ---------------------------
manifest_rows = []

for key, filepaths in video_frames.items():
    out_path = os.path.join(OUT_DIR, f"{key}.npz")

    if os.path.exists(out_path):
        print(f"Skipping {key} (already extracted)")
        manifest_rows.append([key, video_labels[key], len(filepaths), out_path])
        continue

    filepaths = sorted(filepaths, key=frame_sort_key)

    batch = tf.stack([load_and_preprocess(f) for f in filepaths])
    feats = feature_extractor.predict(batch, batch_size=32, verbose=0)  # (num_frames, FEATURE_DIM)

    np.savez_compressed(out_path, features=feats, label=video_labels[key], key=key)

    manifest_rows.append([key, video_labels[key], len(filepaths), out_path])
    print(f"Saved {key}: {feats.shape}")

with open(os.path.join(OUT_DIR, "manifest.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["video_key", "label", "num_frames", "npz_path"])
    writer.writerows(manifest_rows)

print(f"\nDone. {len(manifest_rows)} video feature-sequences saved to '{OUT_DIR}/'")
print("manifest.csv lists every sequence for the LSTM stage in Review 2.")