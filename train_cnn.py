"""
train_cnn.py
------------
Trains a CNN from scratch on the driver drowsiness/distraction dataset.

Expected folder structure:
    Frames/
        <Subject>/              e.g. Harini, HM, Mahesh, Mohan, Praneeth, ...
            <Class>/             Distraction, Drinking, Normal, Phone, Sleeping
                *.jpg / *.png    frame images (any naming, angle can be in filename
                                 or in an extra subfolder - see get_video_key() below)

Subjects are split at the SUBJECT level (not frame level) into train/val/test
so the same person's face never appears in more than one split. This avoids
identity leakage inflating your accuracy.
"""

import os
import glob
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ----------------------------- CONFIG ----------------------------------
DATA_DIR = "Frames_Clean_224"       # black-padding-free, already-resized 224x224 frames
IMG_SIZE = (224, 224)               # matches the pre-resized frames
BATCH_SIZE = 64
EPOCHS = 30
FEATURE_DIM = 256                   # size of the feature vector for the LSTM stage
CLASSES = ["Distraction", "Drinking", "Normal", "Phone", "Sleeping"]
SEED = 42

# Subject-wise split. Edit these lists to match your 9 subject folder names.
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


# ----------------------- DATA COLLECTION --------------------------------
def collect_filepaths(subjects, frame_stride=3):
    """
    Return list of (filepath, class_index) for the given subjects.
    frame_stride: keep every Nth frame per video to reduce near-duplicate
    frames (consecutive frames from the same video are nearly identical at
    our ~5-6fps extraction rate, and too many near-duplicates encourages
    the CNN to memorize specific people rather than learn general cues).
    """
    samples = []
    for subj in subjects:
        for class_idx, cls in enumerate(CLASSES):
            pattern = os.path.join(DATA_DIR, subj, cls, "**", "*.*")
            files = glob.glob(pattern, recursive=True)
            files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            files.sort()  # group same-video frames together before striding
            files = files[::frame_stride]
            for f in files:
                samples.append((f, class_idx))
    return samples


def get_video_key(filepath):
    """
    Groups a frame with the video it came from: subject + class + angle.
    Used later during feature extraction to build ordered sequences.

    Matches the extraction script's naming convention:
        {video_stem}_{framenumber:06d}.jpg   e.g. front_000001.jpg, 2_000005.jpg

    Angle is whatever the original video file was named (front/left/right,
    or 2/3, etc.) - taken as everything before the last underscore-number.
    """
    parts = filepath.replace("\\", "/").split("/")
    subj, cls = parts[-3], parts[-2]
    fname = os.path.splitext(os.path.basename(filepath))[0]
    angle, _, _ = fname.rpartition("_")  # strip the trailing _000123 frame number
    if not angle:
        angle = fname
    return f"{subj}_{cls}_{angle}"


train_samples = collect_filepaths(TRAIN_SUBJECTS, frame_stride=3)
val_samples = collect_filepaths(VAL_SUBJECTS, frame_stride=3)
test_samples = collect_filepaths(TEST_SUBJECTS, frame_stride=3)

print(f"Train frames: {len(train_samples)} | Val frames: {len(val_samples)} | Test frames: {len(test_samples)}")

from collections import Counter

def print_distribution(name, samples):
    counts = Counter(label for _, label in samples)
    print(f"\n{name} distribution:")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {counts.get(i, 0)}")

print_distribution("TRAIN", train_samples)
print_distribution("VALIDATION", val_samples)
print_distribution("TEST", test_samples)


# ----------------------------- TF DATA PIPELINE --------------------------
def load_image(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)  # handles jpg AND png
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img, label


_augment_layer = tf.keras.Sequential([
    layers.RandomRotation(0.03, fill_mode="reflect"),       # ~±11 degrees max
    layers.RandomZoom(0.10, fill_mode="reflect"),
    layers.RandomTranslation(0.05, 0.05, fill_mode="reflect"),
])


def random_rotate_zoom_translate(img):
    return _augment_layer(img, training=True)


def make_dataset(samples, training=False):
    filepaths = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.cache()  # cache decoded images in RAM after first epoch - big speedup
        ds = ds.shuffle(2000, seed=SEED)
        # stronger augmentation to reduce overfitting to specific people/setups
        ds = ds.map(lambda x, y: (tf.image.random_flip_left_right(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda x, y: (tf.image.random_brightness(x, 0.2), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda x, y: (tf.image.random_contrast(x, 0.8, 1.2), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda x, y: (random_rotate_zoom_translate(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda x, y: (tf.clip_by_value(x, 0.0, 1.0), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.cache()
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(train_samples, training=True)
val_ds = make_dataset(val_samples)
test_ds = make_dataset(test_samples)


# ----------------------------- CNN MODEL ---------------------------------
def build_cnn(input_shape=(224, 224, 3), num_classes=5, feature_dim=256):
    reg = tf.keras.regularizers.l2(1e-4)
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.GlobalAveragePooling2D()(x)

    # This Dense layer IS the feature vector used later for the LSTM stage
    features = layers.Dense(feature_dim, activation="relu", kernel_regularizer=reg, name="feature_vector")(x)
    x = layers.Dropout(0.5)(features)

    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    classifier_model = models.Model(inputs, outputs, name="cnn_classifier")
    return classifier_model


model = build_cnn(input_shape=IMG_SIZE + (3,), num_classes=len(CLASSES), feature_dim=FEATURE_DIM)
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
    tf.keras.callbacks.ModelCheckpoint("best_cnn.keras", save_best_only=True, monitor="val_accuracy"),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.4f}")

model.save("cnn_full_model.keras")
print("Saved trained model to cnn_full_model.keras (and best_cnn.keras checkpoint)")