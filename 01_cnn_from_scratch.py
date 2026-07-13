"""
============================================================
 1 · CNN from Scratch -- Cats vs Dogs
============================================================
A convolutional neural network built and trained FROM ZERO on
a full cats-vs-dogs dataset.  This is the baseline you compare
every transfer-learning model against.

Steps: load data -> visualise -> build CNN -> train -> plot
       curves -> confusion matrix.
============================================================
"""

# ==========================================================
# Step 1 -- Imports
# ==========================================================
import os, sys, time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")                       # headless-safe
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)

print("TensorFlow version:", tf.__version__)
print("GPU detected      :", tf.config.list_physical_devices("GPU"))


# ==========================================================
# Terminal progress-bar callback
# ==========================================================
class TerminalProgressBar(keras.callbacks.Callback):
    """Pretty terminal progress bar with live metrics."""

    BAR_LEN = 30

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.epoch_start = time.time()
        self.num_batches = self.params.get("steps", None)

    def on_train_batch_end(self, batch, logs=None):
        if self.num_batches is None:
            return
        done   = batch + 1
        frac   = done / self.num_batches
        filled = int(self.BAR_LEN * frac)
        bar    = "#" * filled + "-" * (self.BAR_LEN - filled)
        loss   = logs.get("loss", 0)
        acc    = logs.get("accuracy", 0)
        sys.stdout.write(
            f"\r  Epoch {self.epoch+1:>2d} |{bar}| "
            f"{done}/{self.num_batches} "
            f"-- loss: {loss:.4f} -- acc: {acc:.4f}"
        )
        sys.stdout.flush()

    def on_epoch_end(self, epoch, logs=None):
        elapsed  = time.time() - self.epoch_start
        val_loss = logs.get("val_loss", 0)
        val_acc  = logs.get("val_accuracy", 0)
        sys.stdout.write(
            f"\r  Epoch {epoch+1:>2d} |{'#'*self.BAR_LEN}| "
            f"done in {elapsed:.1f}s "
            f"-- val_loss: {val_loss:.4f} -- val_acc: {val_acc:.4f}\n"
        )
        sys.stdout.flush()


# ==========================================================
# Step 2 -- Set up dataset paths
# ==========================================================
# Point to the local full dog-vs-cat dataset
# (already extracted under ./data/dog-cat-full-dataset/)
base_dir  = os.path.join("data", "dog-cat-full-dataset", "data")
train_dir = os.path.join(base_dir, "train")   # ~10 000 cats + ~10 000 dogs
val_dir   = os.path.join(base_dir, "test")     #  2 500 cats +  2 500 dogs

for split in [train_dir, val_dir]:
    for cls in sorted(os.listdir(split)):
        n = len(os.listdir(os.path.join(split, cls)))
        print(f"{split}/{cls}: {n} images")


# ==========================================================
# Step 3 -- Load images into tf.data datasets
# ==========================================================
# Sub-folder names become the labels automatically.
IMG_SIZE   = (160, 160)
BATCH_SIZE = 32

train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=123,
)

# shuffle=False on validation keeps the order fixed so the confusion
# matrix labels line up with the predictions later.
val_ds = keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)


# ==========================================================
# Step 4 -- Look at the data  (save a sample grid)
# ==========================================================
plt.figure(figsize=(9, 9))
for images, labels in train_ds.take(1):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.suptitle("Sample training images")
plt.tight_layout()
plt.savefig("01_sample_images.png", dpi=120)
plt.close()
print("Saved sample grid -> 01_sample_images.png")


# ==========================================================
# Step 5 -- Speed-up pipeline (cache + prefetch)
# ==========================================================
# For a from-scratch CNN we normalise inside the model (a Rescaling
# layer), so here we only cache + prefetch for speed.
AUTOTUNE   = tf.data.AUTOTUNE
train_prep = train_ds.cache().prefetch(AUTOTUNE)
val_prep   = val_ds.cache().prefetch(AUTOTUNE)


# ==========================================================
# Step 6 -- Build the CNN
# ==========================================================
# A small CNN built from scratch: 4 conv/pool blocks + a dense head.
# Rescaling(1/255) normalises pixels; data augmentation fights overfitting.
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")

model = keras.Sequential([
    keras.Input(shape=IMG_SIZE + (3,)),
    data_augmentation,
    layers.Rescaling(1.0 / 255),

    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
], name="cnn_from_scratch")

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()


# ==========================================================
# Step 7 -- Train
# ==========================================================
EPOCHS  = 15
print(f"\n{'='*60}")
print(f"  Training CNN from scratch  --  {EPOCHS} epochs")
print(f"{'='*60}")
history = model.fit(
    train_prep,
    validation_data=val_prep,
    epochs=EPOCHS,
    verbose=0,                        # silence default bar
    callbacks=[TerminalProgressBar()],
)


# ==========================================================
# Step 8 -- Accuracy & loss curves
# ==========================================================
acc      = history.history["accuracy"]
val_acc  = history.history["val_accuracy"]
loss     = history.history["loss"]
val_loss = history.history["val_loss"]
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc,     marker="o", label="Train")
plt.plot(epochs_range, val_acc, marker="o", label="Validation")
plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend(); plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss,     marker="o", label="Train")
plt.plot(epochs_range, val_loss, marker="o", label="Validation")
plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("01_training_curves.png", dpi=120)
plt.close()
print("Saved training curves -> 01_training_curves.png")


# ==========================================================
# Step 9 -- Confusion matrix & report
# ==========================================================
val_loss_v, val_acc_v = model.evaluate(val_prep, verbose=0)
print(f"Validation accuracy: {val_acc_v:.4f}")

# True labels (val_ds was NOT shuffled, so the order is stable)
y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)

# Predicted labels
y_prob = model.predict(val_prep)
y_pred = np.argmax(y_prob, axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.grid(False)
plt.savefig("01_confusion_matrix.png", dpi=120)
plt.close()
print("Saved confusion matrix -> 01_confusion_matrix.png")

print("\nClassification report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))


# ==========================================================
# Step 10 -- See predictions  (save a prediction grid)
# ==========================================================
plt.figure(figsize=(12, 12))
for images, labels in val_ds.take(1):
    probs = model.predict(images, verbose=0)
    preds = np.argmax(probs, axis=1)
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        true_lbl = class_names[labels[i]]
        pred_lbl = class_names[preds[i]]
        color = "green" if true_lbl == pred_lbl else "red"
        plt.title(f"pred: {pred_lbl}\ntrue: {true_lbl}", color=color, fontsize=10)
        plt.axis("off")
plt.tight_layout()
plt.savefig("01_predictions.png", dpi=120)
plt.close()
print("Saved prediction grid -> 01_predictions.png")

print("\n" + "="*60)
print("  Done -- CNN from Scratch")
print("="*60)
