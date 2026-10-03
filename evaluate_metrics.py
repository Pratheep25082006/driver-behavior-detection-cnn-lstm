"""
evaluate_standalone.py
-----------------------
Run this AFTER training, in a fresh session, using the saved
`cnn_full_model.keras`. It rebuilds the val/test datasets exactly the
same way train_cnn.py did (same subject split, same seed, same
preprocessing) and produces:

    1. confusion_matrix.png       - test-set confusion matrix (counts + %)
    2. classification_report.txt  - precision/recall/F1 per class
    3. accuracy_curve.png / loss_curve.png  - ONLY if history.pkl exists
       (see the note at the bottom of train_cnn.py's fit() call to save it)
    4. Printed final train / val / test accuracy

Just run:  python evaluate_standalone.py
"""

import os
import glob
import random
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# ----------------------------- CONFIG (must match train_cnn.py) --------
DATA_DIR = "Frames_Clean_224"
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
CLASSES = ["Distraction", "Drinking", "Normal", "Phone", "Sleeping"]
SEED = 42
MODEL_PATH = "cnn_full_model.keras"   # or "best_cnn.keras" if you want the checkpoint instead

ALL_SUBJECTS = ["Harini", "HM", "Mahesh", "Mohan", "Praneeth",
                 "Pratheep", "Rakesh", "Siva", "Sriram"]

random.seed(SEED)
shuffled = ALL_SUBJECTS.copy()
random.shuffle(shuffled)

TRAIN_SUBJECTS = shuffled[:6]
VAL_SUBJECTS = shuffled[6:8]
TEST_SUBJECTS = shuffled[8:9]

print("Train subjects:", TRAIN_SUBJECTS)
print("Val subjects:  ", VAL_SUBJECTS)
print("Test subjects: ", TEST_SUBJECTS)


# ----------------------- DATA COLLECTION (same as train_cnn.py) --------
def collect_filepaths(subjects, frame_stride=3):
    samples = []
    for subj in subjects:
        for class_idx, cls in enumerate(CLASSES):
            pattern = os.path.join(DATA_DIR, subj, cls, "**", "*.*")
            files = glob.glob(pattern, recursive=True)
            files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            files.sort()
            files = files[::frame_stride]
            for f in files:
                samples.append((f, class_idx))
    return samples


train_samples = collect_filepaths(TRAIN_SUBJECTS, frame_stride=3)
val_samples = collect_filepaths(VAL_SUBJECTS, frame_stride=3)
test_samples = collect_filepaths(TEST_SUBJECTS, frame_stride=3)

print(f"Train frames: {len(train_samples)} | Val frames: {len(val_samples)} | Test frames: {len(test_samples)}")


# ----------------------------- TF DATA PIPELINE (unaugmented) ----------
def load_image(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img, label


def make_dataset(samples):
    filepaths = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# NOTE: this train_ds is UNAUGMENTED (no random flip/rotate/zoom/brightness),
# unlike the one used during training. That's intentional here — we want a
# clean "how well does the model fit its own training data" number, not one
# blurred by random augmentation.
train_ds = make_dataset(train_samples)
val_ds = make_dataset(val_samples)
test_ds = make_dataset(test_samples)


# ----------------------------- LOAD MODEL -------------------------------
print(f"Loading model from {MODEL_PATH} ...")
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()


# --------------------------------------------------------------------
# 1. ACCURACY / LOSS CURVES (only if you saved history.pkl during training)
# --------------------------------------------------------------------
if os.path.exists("history.pkl"):
    with open("history.pkl", "rb") as f:
        hist = pickle.load(f)

    acc, val_acc = hist["accuracy"], hist["val_accuracy"]
    loss, val_loss = hist["loss"], hist["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, acc, label="Train Accuracy", marker="o")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", marker="o")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("accuracy_curve.png", dpi=150); plt.close()
    print("Saved accuracy_curve.png")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, loss, label="Train Loss", marker="o")
    plt.plot(epochs_range, val_loss, label="Validation Loss", marker="o")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150); plt.close()
    print("Saved loss_curve.png")
else:
    print("\n[!] No history.pkl found - skipping accuracy/loss curves.")
    print("    To get these next time, add this right after model.fit() in train_cnn.py:")
    print('        import pickle')
    print('        with open("history.pkl", "wb") as f:')
    print('            pickle.dump(history.history, f)\n')


# --------------------------------------------------------------------
# 2. CONFUSION MATRIX ON THE TEST SET
# --------------------------------------------------------------------
y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
y_prob = model.predict(test_ds)
y_pred = np.argmax(y_prob, axis=1)

cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

im0 = axes[0].imshow(cm, cmap="Blues")
axes[0].set_title("Confusion Matrix (Counts)")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
axes[0].set_xticks(range(len(CLASSES))); axes[0].set_yticks(range(len(CLASSES)))
axes[0].set_xticklabels(CLASSES, rotation=45, ha="right")
axes[0].set_yticklabels(CLASSES)
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        axes[0].text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
axes[1].set_title("Confusion Matrix (Row-normalized %)")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
axes[1].set_xticks(range(len(CLASSES))); axes[1].set_yticks(range(len(CLASSES)))
axes[1].set_xticklabels(CLASSES, rotation=45, ha="right")
axes[1].set_yticklabels(CLASSES)
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        axes[1].text(j, i, f"{cm_norm[i, j]*100:.1f}%", ha="center", va="center",
                     color="white" if cm_norm[i, j] > 0.5 else "black")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Saved confusion_matrix.png")


# --------------------------------------------------------------------
# 3. CLASSIFICATION REPORT
# --------------------------------------------------------------------
report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
print("\nClassification Report (Test Set):\n")
print(report)
with open("classification_report.txt", "w") as f:
    f.write(report)
print("Saved classification_report.txt")


# --------------------------------------------------------------------
# 4. FINAL ACCURACY SUMMARY (train / val / test)
# --------------------------------------------------------------------
final_train_acc = model.evaluate(train_ds, verbose=0)[1]
final_val_acc = model.evaluate(val_ds, verbose=0)[1]
final_test_acc = model.evaluate(test_ds, verbose=0)[1]

print("\n===== FINAL ACCURACY SUMMARY =====")
print(f"Train Accuracy:      {final_train_acc:.4f}")
print(f"Validation Accuracy: {final_val_acc:.4f}")
print(f"Test Accuracy:       {final_test_acc:.4f}")
print("===================================")
