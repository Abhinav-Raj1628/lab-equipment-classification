from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "classification"
MODEL_PATH = ROOT / "models" / "best_model.keras"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

with open(ROOT / "class_names.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA / "test",
    image_size=(224, 224),
    batch_size=32,
    shuffle=False,
)

model = tf.keras.models.load_model(MODEL_PATH)

y_true = np.concatenate([y.numpy() for _, y in test_ds])
probs = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probs, axis=1)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0,
)
print(report)
(RESULTS / "classification_report.txt").write_text(report, encoding="utf-8")

cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(14, 14))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
plt.title("Unseen Test Set Confusion Matrix")
plt.tight_layout()
plt.savefig(RESULTS / "confusion_matrix.png", dpi=200)
plt.close()

print(f"Saved evaluation files to: {RESULTS}")
