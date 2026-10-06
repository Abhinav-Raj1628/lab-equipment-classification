from pathlib import Path
import json
import sys
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.keras"

if len(sys.argv) != 2:
    print("Usage: python predict.py path/to/image.jpg")
    raise SystemExit(1)

image_path = Path(sys.argv[1])
if not image_path.exists():
    raise FileNotFoundError(image_path)

with open(ROOT / "class_names.json", "r", encoding="utf-8") as f:
    class_names = json.load(f)

model = tf.keras.models.load_model(MODEL_PATH)

img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
arr = tf.keras.utils.img_to_array(img)
arr = tf.expand_dims(arr, 0)

probs = model.predict(arr, verbose=0)[0]
idx = int(np.argmax(probs))

print(f"Predicted class: {class_names[idx]}")
print(f"Confidence: {probs[idx] * 100:.2f}%")
