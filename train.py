from pathlib import Path
import json
import tensorflow as tf
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "classification"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
)
valid_ds = tf.keras.utils.image_dataset_from_directory(
    DATA / "valid",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = train_ds.class_names
(ROOT / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomZoom(0.10),
], name="augmentation")

base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)
base.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODELS / "best_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True,
    ),
]

print("\nStage 1: training classifier head...")
h1 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=12,
    callbacks=callbacks,
)

# Fine-tune only the top portion of MobileNetV2.
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\nStage 2: fine-tuning top MobileNetV2 layers...")
h2 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=10,
    callbacks=callbacks,
)

model = tf.keras.models.load_model(MODELS / "best_model.keras")
test_loss, test_acc = model.evaluate(test_ds, verbose=1)

print(f"\nUnseen test accuracy: {test_acc:.4f}")
print(f"Unseen test loss:     {test_loss:.4f}")

acc = h1.history["accuracy"] + h2.history["accuracy"]
val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
loss = h1.history["loss"] + h2.history["loss"]
val_loss = h1.history["val_loss"] + h2.history["val_loss"]

plt.figure(figsize=(9, 5))
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS / "training_accuracy.png", dpi=200)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS / "training_loss.png", dpi=200)
plt.close()

print("Training complete.")
