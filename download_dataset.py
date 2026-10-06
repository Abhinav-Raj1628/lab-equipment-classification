from pathlib import Path
import zipfile
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "dataset" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

URL = "https://figshare.com/ndownloader/files/44682395"
ZIP_PATH = RAW / "chemistry_lab_equipment_dataset.zip"

def download():
    print("Downloading the public chemistry-lab equipment dataset...")
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(ZIP_PATH, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc="dataset"
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    print(f"Saved: {ZIP_PATH}")

def extract():
    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(RAW)
    print(f"Extracted into: {RAW}")

if __name__ == "__main__":
    if not ZIP_PATH.exists():
        download()
    else:
        print("ZIP already exists; skipping download.")
    extract()
    print("Dataset preparation download step complete.")
