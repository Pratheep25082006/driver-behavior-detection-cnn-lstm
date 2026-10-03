"""
load_weights_only.py
----------------------
Your saved model's config.json was written by a NEWER Keras version than
the one currently installed (GlorotUniform now serializes extra fields
'input_axes'/'output_axes' that your installed Keras doesn't know about
yet). That breaks tf.keras.models.load_model() entirely, because it has
to reconstruct every layer from that JSON config.

Workaround: rebuild the exact same architecture from code (same as
build_cnn() in train_cnn.py) and load ONLY the weights file - weights
loading doesn't touch the initializer config at all, so the version
mismatch is a non-issue.

This produces a model object identical to the one you trained. Use it
in place of tf.keras.models.load_model() in evaluate_standalone.py.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
CLASSES = ["Distraction", "Drinking", "Normal", "Phone", "Sleeping"]
FEATURE_DIM = 256

WEIGHTS_PATH = "cnn_full_model.keras/model.weights.h5"  # inside the unzipped folder


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

    features = layers.Dense(feature_dim, activation="relu", kernel_regularizer=reg, name="feature_vector")(x)
    x = layers.Dropout(0.5)(features)

    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    return models.Model(inputs, outputs, name="cnn_classifier")


def load_trained_model():
    model = build_cnn(input_shape=IMG_SIZE + (3,), num_classes=len(CLASSES), feature_dim=FEATURE_DIM)
    model.load_weights(WEIGHTS_PATH)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    model = load_trained_model()
    model.summary()
    print("\nWeights loaded successfully - model is ready to use.")

    # Optional: save a clean, reloadable copy for next time in a format
    # this Keras version can both write AND read back without issues.
    model.save("cnn_full_model_clean.keras")
    print("Also saved a fresh, self-consistent copy: cnn_full_model_clean.keras")