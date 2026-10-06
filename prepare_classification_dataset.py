from pathlib import Path
import shutil
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "dataset" / "raw"
OUT = ROOT / "dataset" / "classification"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def find_dataset_root():
    # Look for a directory containing data.yaml
    matches = list(RAW.rglob("data.yaml")) + list(RAW.rglob("data.yml"))
    if not matches:
        raise FileNotFoundError(
            "Could not find data.yaml/data.yml. Check dataset/raw after extraction."
        )
    return matches[0].parent

def load_names(root):
    yaml_files = list(root.glob("data.yaml")) + list(root.glob("data.yml"))
    with open(yaml_files[0], "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    if not names:
        raise ValueError("No class names found in data.yaml.")
    return names

def locate_split(root, split):
    # Common YOLO layout: train/images, train/labels
    candidates = [root / split, root / split.lower()]
    for p in candidates:
        if (p / "images").exists() and (p / "labels").exists():
            return p
    # Recursive fallback
    for p in root.rglob("images"):
        if p.parent.name.lower() == split.lower() and (p.parent / "labels").exists():
            return p.parent
    raise FileNotFoundError(f"Could not locate {split}/images and {split}/labels")

def crop_from_yolo(img, line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None, None
    cls, xc, yc, w, h = map(float, parts[:5])
    cls = int(cls)

    W, H = img.size
    x1 = max(0, int((xc - w / 2) * W))
    y1 = max(0, int((yc - h / 2) * H))
    x2 = min(W, int((xc + w / 2) * W))
    y2 = min(H, int((yc + h / 2) * H))

    if x2 <= x1 or y2 <= y1:
        return None, None

    return cls, img.crop((x1, y1, x2, y2))

def process_split(root, split, names):
    split_dir = locate_split(root, split)
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    count = 0
    for img_path in images_dir.rglob("*"):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        lines = label_path.read_text(encoding="utf-8").splitlines()

        for obj_idx, line in enumerate(lines):
            cls, crop = crop_from_yolo(img, line)
            if crop is None or cls >= len(names):
                continue

            class_name = str(names[cls]).replace("/", "-")
            class_dir = OUT / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            out_name = f"{img_path.stem}_obj{obj_idx}.jpg"
            crop.save(class_dir / out_name, quality=95)
            count += 1

    print(f"{split}: created {count} classification crops")

def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    root = find_dataset_root()
    names = load_names(root)

    print("Classes:")
    for i, name in enumerate(names):
        print(f"  {i}: {name}")

    for split in ["train", "valid", "test"]:
        process_split(root, split, names)

    print(f"\nClassification dataset created at: {OUT}")

if __name__ == "__main__":
    main()
