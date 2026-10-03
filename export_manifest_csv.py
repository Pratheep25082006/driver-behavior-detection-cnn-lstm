"""
export_manifest_csv.py
------------------------
Reads every .npz file in features/ (produced by extract_features.py) and
writes a clean, human-readable CSV where entries are grouped into separate
SECTIONS by class label (Distraction, Drinking, Normal, Phone, Sleeping),
rather than one flat, intermixed list.

Output: features/manifest_by_class.csv

Format:
    === Distraction (27 videos) ===
    video_key,label,label_name,num_frames,npz_path
    Harini_Distraction_front,0,Distraction,187,features/Harini_Distraction_front.npz
    ...

    === Drinking (27 videos) ===
    video_key,label,label_name,num_frames,npz_path
    ...
"""

import os
import glob
import csv
import numpy as np
from collections import defaultdict

FEATURES_DIR = "features"
CLASSES = ["Distraction", "Drinking", "Normal", "Phone", "Sleeping"]

npz_files = glob.glob(os.path.join(FEATURES_DIR, "*.npz"))
print(f"Found {len(npz_files)} .npz files in '{FEATURES_DIR}/'")

# Group entries by class label
grouped = defaultdict(list)  # label_name -> list of (video_key, label, num_frames, npz_path)

for npz_path in sorted(npz_files):
    data = np.load(npz_path, allow_pickle=True)
    key = str(data["key"])
    label = int(data["label"])
    num_frames = data["features"].shape[0]
    label_name = CLASSES[label]

    grouped[label_name].append((key, label, num_frames, npz_path))

# Write the sectioned CSV
out_path = os.path.join(FEATURES_DIR, "manifest_by_class.csv")

with open(out_path, "w", newline="") as f:
    writer = csv.writer(f)

    for cls in CLASSES:
        entries = grouped.get(cls, [])
        f.write(f"=== {cls} ({len(entries)} videos) ===\n")
        writer.writerow(["video_key", "label", "label_name", "num_frames", "npz_path"])
        for key, label, num_frames, npz_path in entries:
            writer.writerow([key, label, cls, num_frames, npz_path])
        f.write("\n")  # blank line between sections

# Also print a quick summary to the console
print("\nSummary:")
total_videos = 0
total_frames = 0
for cls in CLASSES:
    entries = grouped.get(cls, [])
    frames_in_class = sum(e[2] for e in entries)
    print(f"  {cls:<12} -> {len(entries)} videos, {frames_in_class} total frames")
    total_videos += len(entries)
    total_frames += frames_in_class

print(f"\nTotal: {total_videos} videos, {total_frames} frames")
print(f"Saved sectioned CSV to: {out_path}")
