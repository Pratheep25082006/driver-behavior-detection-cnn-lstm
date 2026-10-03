"""
train_lstm.py
---------------
Step 6: trains an LSTM on the CNN-extracted feature sequences.

Reads features/manifest.csv (produced by extract_features.py), loads every
video's (num_frames, 256) feature sequence, pads/truncates to a common
length, and trains an LSTM classifier - the temporal stage of the
CNN -> feature extraction -> LSTM pipeline.

Uses the SAME subject split (same seed, same 6/2/1 split) as train_cnn.py,
so CNN and LSTM results are directly comparable on identical held-out people.
"""

import os
import random
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# ----------------------------- CONFIG ----------------------------------
FEATURES_DIR = "features"
MANIFEST_PATH = os.path.join(FEATURES_DIR, "manifest.csv")
CLASSES = ["Distraction", "Drinking", "Normal", "Phone", "Sleeping"]
FEATURE_DIM = 256
BATCH_SIZE = 16
EPOCHS = 30
SEED = 42

# Must match train_cnn.py exactly, so both models are evaluated on the same people
ALL_SUBJECTS = ["Harini", "HM", "Mahesh", "Mohan", "Praneeth",
                 "Pratheep", "Rakesh", "Siva", "Sriram"]

random.seed(SEED)
shuffled = ALL_SUBJECTS.copy()
random.shuffle(shuffled)

TRAIN_SUBJECTS = shuffled[:6]
VAL_SUBJECTS = shuffled[6:8]
TEST_SUBJECTS = shuffled[8:9]

print("=" * 60)
print("SUBJECT SPLIT (matches train_cnn.py)")
print("=" * 60)
print("Train subjects:", TRAIN_SUBJECTS)
print("Val subjects:  ", VAL_SUBJECTS)
print("Test subjects: ", TEST_SUBJECTS)

# ----------------------------- LOAD MANIFEST ----------------------------
if not os.path.exists(MANIFEST_PATH):
    raise FileNotFoundError(f"'{MANIFEST_PATH}' not found. Run extract_features.py first.")

manifest = pd.read_csv(MANIFEST_PATH)
print(f"\nManifest has {len(manifest)} videos")

EXPECTED_VIDEOS = len(ALL_SUBJECTS) * len(CLASSES) * 3  # 3 angles
if len(manifest) != EXPECTED_VIDEOS:
    print(f"  WARNING: expected {EXPECTED_VIDEOS} videos (9 subjects x 5 classes x 3 angles), "
          f"found {len(manifest)}. Check extraction completed for everyone.")


def get_subject_from_key(video_key):
    # video_key format: Subject_Class_angle (e.g. "Harini_Distraction_front")
    return video_key.split("_")[0]


manifest["subject"] = manifest["video_key"].apply(get_subject_from_key)

train_manifest = manifest[manifest["subject"].isin(TRAIN_SUBJECTS)]
val_manifest = manifest[manifest["subject"].isin(VAL_SUBJECTS)]
test_manifest = manifest[manifest["subject"].isin(TEST_SUBJECTS)]

print(f"\nTrain videos: {len(train_manifest)} | Val videos: {len(val_manifest)} | Test videos: {len(test_manifest)}")

if len(train_manifest) + len(val_manifest) + len(test_manifest) != len(manifest):
    unmatched = manifest[~manifest["subject"].isin(ALL_SUBJECTS)]
    print(f"  WARNING: {len(unmatched)} videos have a subject name not in ALL_SUBJECTS:")
    print(unmatched["video_key"].tolist())

# ----------------------------- SEQUENCE LENGTH AUDIT ---------------------
all_lengths = manifest["num_frames"].values
print("\n" + "=" * 60)
print("SEQUENCE LENGTH STATS (frames per video)")
print("=" * 60)
print(f"  min: {all_lengths.min()}  max: {all_lengths.max()}  "
      f"mean: {all_lengths.mean():.1f}  median: {np.median(all_lengths):.1f}")

MAX_SEQ_LEN = int(np.percentile(all_lengths, 95))
print(f"  Using MAX_SEQ_LEN = {MAX_SEQ_LEN} (95th percentile) "
      f"- longer videos truncated, shorter ones zero-padded")

n_truncated = (all_lengths > MAX_SEQ_LEN).sum()
print(f"  {n_truncated} / {len(all_lengths)} videos will be truncated")


# ----------------------------- LOAD + PAD SEQUENCES ----------------------
def load_sequences(df):
    X = np.zeros((len(df), MAX_SEQ_LEN, FEATURE_DIM), dtype=np.float32)
    y = np.zeros(len(df), dtype=np.int32)

    for i, row in enumerate(df.itertuples()):
        data = np.load(row.npz_path)
        feats = data["features"]  # (num_frames, 256)
        label = int(data["label"])

        n = min(len(feats), MAX_SEQ_LEN)
        X[i, :n, :] = feats[:n]
        y[i] = label

    return X, y


print("\nLoading and padding sequences...")
X_train, y_train = load_sequences(train_manifest)
X_val, y_val = load_sequences(val_manifest)
X_test, y_test = load_sequences(test_manifest)

print(f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

from collections import Counter

def print_distribution(name, labels):
    counts = Counter(labels)
    print(f"\n{name} class distribution:")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {counts.get(i, 0)}")

print_distribution("TRAIN", y_train)
print_distribution("VALIDATION", y_val)
print_distribution("TEST", y_test)


# ----------------------------- BUILD LSTM MODEL ---------------------------
def build_lstm(seq_len, feature_dim, num_classes=5):
    inputs = layers.Input(shape=(seq_len, feature_dim))

    x = layers.Masking(mask_value=0.0)(inputs)  # ignore zero-padded timesteps
    x = layers.LSTM(128, return_sequences=False,
                     dropout=0.3, recurrent_dropout=0.0)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu",
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = models.Model(inputs, outputs, name="lstm_classifier")
    return model


model = build_lstm(MAX_SEQ_LEN, FEATURE_DIM, num_classes=len(CLASSES))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ----------------------------- TRAIN --------------------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_accuracy"),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss"),
    tf.keras.callbacks.ModelCheckpoint("best_lstm.keras", save_best_only=True, monitor="val_accuracy"),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
)

with open("training_history_lstm.json", "w") as f:
    json.dump(history.history, f)
print("Saved training history to training_history_lstm.json")

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

final_metrics = {
    "train_accuracy": float(history.history["accuracy"][-1]),
    "val_accuracy": float(history.history["val_accuracy"][-1]),
    "best_val_accuracy": float(max(history.history["val_accuracy"])),
    "test_accuracy": float(test_acc),
    "test_loss": float(test_loss),
    "max_seq_len": MAX_SEQ_LEN,
    "train_subjects": TRAIN_SUBJECTS,
    "val_subjects": VAL_SUBJECTS,
    "test_subjects": TEST_SUBJECTS,
}
with open("final_metrics_lstm.json", "w") as f:
    json.dump(final_metrics, f, indent=2)
print("Saved final metrics to final_metrics_lstm.json")

model.save("lstm_full_model.keras")
print("Saved trained model to lstm_full_model.keras (and best_lstm.keras checkpoint)")

# Save test set predictions for confusion matrix generation later
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
np.savez("lstm_test_predictions.npz", y_true=y_test, y_pred=y_pred, y_pred_probs=y_pred_probs)
print("Saved test predictions to lstm_test_predictions.npz (for confusion matrix)")