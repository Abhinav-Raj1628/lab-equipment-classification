# Laboratory Equipment Image Classification using Transfer Learning

## Project objective
Build an image-classification system that recognizes laboratory equipment from images using transfer learning with a pre-trained CNN (MobileNetV2).

The workflow is:

1. Download a public chemistry-lab equipment dataset.
2. Convert the supplied YOLO object-detection annotations into cropped image-classification samples.
3. Train a MobileNetV2 classifier.
4. Fine-tune the upper part of the CNN.
5. Evaluate on unseen test images.
6. Generate a confusion matrix and classification report.
7. Predict the class of a new image.

## Dataset
This project is designed for the public **Real-world chemistry lab image dataset for equipment recognition across 25 apparatus categories**.

Dataset information:
- 4,599 images
- 25 apparatus categories
- train / validation / test split
- 640x640 images
- YOLO bounding-box annotations

Download/source:
https://github.com/SakhawatHossain/chem-lab-equipment-recognition

The GitHub README also provides the Figshare dataset download (about 280 MB):
https://figshare.com/ndownloader/files/44682395

The dataset is described in Scientific Data:
https://www.nature.com/articles/s41597-025-05952-3

## Repository layout

```text
customer-equipment-classifier/
├── README.md
├── requirements.txt
├── .gitignore
├── download_dataset.py
├── prepare_classification_dataset.py
├── train.py
├── predict.py
├── evaluate.py
├── dataset/
│   ├── raw/                 # downloaded dataset goes here
│   └── classification/      # generated train/valid/test class folders
├── src/
│   └── utils.py
├── notebooks/
│   └── equipment_classification.ipynb
├── models/
├── results/
└── screenshots/
```

## Step 1 - Create the project

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd customer-equipment-classifier
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

Linux/macOS:
```bash
source venv/bin/activate
```

## Step 2 - Install packages

```bash
pip install -r requirements.txt
```

## Step 3 - Download the dataset

```bash
python download_dataset.py
```

This saves the dataset ZIP under `dataset/raw/` and extracts it there.

If the automatic download is blocked on your network, manually download the ZIP from the Figshare link above and place it in `dataset/raw/`, then extract it.

## Step 4 - Prepare classification photosets

The original dataset is object detection data: each image has bounding boxes. Since the assignment asks for **image classification**, this script crops every annotated equipment object and creates:

```text
dataset/classification/
├── train/
│   ├── Beaker/
│   ├── Pipette/
│   ├── ...
├── valid/
│   ├── Beaker/
│   ├── Pipette/
│   └── ...
└── test/
    ├── Beaker/
    ├── Pipette/
    └── ...
```

Run:

```bash
python prepare_classification_dataset.py
```

## Step 5 - Train and fine-tune MobileNetV2

```bash
python train.py
```

The script first freezes the pre-trained MobileNetV2 base and trains the new classification head. It then unfreezes the last part of MobileNetV2 and fine-tunes it with a small learning rate.

Outputs:
- `models/best_model.keras`
- `results/training_history.png`
- `results/classification_report.txt`
- `results/confusion_matrix.png`

## Step 6 - Test an unseen image

Put an image into:

```text
screenshots/test_equipment.jpg
```

Then:

```bash
python predict.py screenshots/test_equipment.jpg
```

Example output:

```text
Predicted class: Pipette
Confidence: 94.37%
```

## Step 7 - Evaluate the complete unseen test set

```bash
python evaluate.py
```

This prints accuracy, precision, recall and F1-score and creates a confusion matrix.

## What to show in your laboratory/project demonstration

Show these five things:

1. Dataset folder and sample images.
2. MobileNetV2 transfer-learning architecture.
3. Training/validation accuracy graph.
4. Confusion matrix and test accuracy.
5. A completely unseen image with the predicted equipment class and confidence.

## Important GitHub rule

Do **not** commit:
- `venv/`
- the 280+ MB raw dataset ZIP
- generated model weights if they are large
- Python cache files

The `.gitignore` already excludes these.

## Suggested project title

**Laboratory Equipment Image Classification using Transfer Learning with MobileNetV2**
