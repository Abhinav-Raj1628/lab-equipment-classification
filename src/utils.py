from pathlib import Path

def count_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for p in Path(folder).rglob("*") if p.suffix.lower() in exts)
