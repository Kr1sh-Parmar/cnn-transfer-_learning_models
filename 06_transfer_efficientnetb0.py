"""
============================================================
 6 · Transfer Learning -- EfficientNetB0
============================================================
EfficientNetB0 uses compound scaling (depth, width and resolution
scaled together) for great accuracy per parameter.
Note: its preprocess_input is a pass-through -- the model rescales
internally, so it expects raw 0-255 pixels.

Steps: load data -> visualise -> preprocess -> load pretrained base
       -> train head -> plot curves -> confusion matrix.
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
matplotlib.use("Agg")
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
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32

train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=123,
)

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
# Step 4 -- Look at the data
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
plt.savefig("06_sample_images.png", dpi=120)
plt.close()
print("Saved sample grid -> 06_sample_images.png")


# ==========================================================
# Step 5 -- Apply EfficientNetB0's own preprocessing
# ==========================================================
# EfficientNet's preprocess_input is a no-op (pass-through).
# The model handles rescaling internally, so we feed raw 0-255.
from tensorflow.keras.applications.efficientnet import preprocess_input

AUTOTUNE = tf.data.AUTOTUNE

def prepare(ds):
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    return ds.cache().prefetch(AUTOTUNE)

train_prep = prepare(train_ds)
val_prep   = prepare(val_ds)


# ==========================================================
# Step 6 -- Build the model on the EfficientNetB0 base
# ==========================================================
base_model = keras.applications.EfficientNetB0(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)

base_model.trainable = False
print("Base model layers:", len(base_model.layers), "| trainable:", base_model.trainable)

model = keras.Sequential([
    keras.Input(shape=IMG_SIZE + (3,)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation="softmax"),
], name="efficientnetb0_transfer")

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()


# ==========================================================
# Step 7 -- Train the classification head
# ==========================================================
EPOCHS = 10
print(f"\n{'='*60}")
print(f"  Training EfficientNetB0 transfer  --  {EPOCHS} epochs")
print(f"{'='*60}")
history = model.fit(
    train_prep,
    validation_data=val_prep,
    epochs=EPOCHS,
    verbose=0,
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
plt.savefig("06_training_curves.png", dpi=120)
plt.close()
print("Saved training curves -> 06_training_curves.png")


# ==========================================================
# Step 9 -- Confusion matrix & report
# ==========================================================
val_loss_v, val_acc_v = model.evaluate(val_prep, verbose=0)
print(f"Validation accuracy: {val_acc_v:.4f}")

y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0)

y_prob = model.predict(val_prep)
y_pred = np.argmax(y_prob, axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.grid(False)
plt.savefig("06_confusion_matrix.png", dpi=120)
plt.close()
print("Saved confusion matrix -> 06_confusion_matrix.png")

print("\nClassification report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))


# ==========================================================
# Step 10 -- See predictions
# ==========================================================
plt.figure(figsize=(12, 12))
for images, labels in val_ds.take(1):
    probs = model.predict(preprocess_input(tf.identity(images)), verbose=0)
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
plt.savefig("06_predictions.png", dpi=120)
plt.close()
print("Saved prediction grid -> 06_predictions.png")

print("\n" + "="*60)
print("  Done -- EfficientNetB0 Transfer Learning")
print("="*60)
