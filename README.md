spheroid classifier.

Spheroid Validator (ResNet-50)

Binary image classifier to label spheroid images as valid or invalid.
Includes a ready-to-use inference script for folder-based prediction and CSV export.

Features

ResNet-50 backbone with custom classification head

Image normalization aligned with ImageNet statistics

Batch inference over a folder (./Test) with per-image confidence

CSV output (predictions.csv) and console preview

Flexible class name discovery from Dataset/train/ or via --classes

Repository Structure
.
├─ infer.py                 # Inference script (folder -> CSV)
├─ resnet50_final_model.pth # Example weights (rename/replace as needed)
├─ Dataset/
│  └─ train/
│     ├─ invalid/           # (optional) used to auto-detect class names
│     └─ valid/
└─ Test/                    # Place test images here for inference

Requirements

Python 3.9–3.12 (recommended)

PyTorch, torchvision, Pillow, numpy

pip install torch torchvision pillow numpy

Quick Start (Inference)

Put the images to classify in ./Test/.

Ensure a weights file exists (e.g., resnet50_final_model.pth).

Run:

# CPU
python infer.py --weights resnet50_final_model.pth --device cpu

# Auto-select device (GPU if available)
python infer.py --weights resnet50_final_model.pth


If Dataset/train/ is not available to infer class names, pass them explicitly:

python infer.py --weights resnet50_final_model.pth --classes "invalid,valid"

Output

A file predictions.csv with columns:

filename — image file name

predicted_class — valid or invalid

confidence — softmax probability of the predicted class

Example:

filename,predicted_class,confidence
img_001.png,valid,0.9821
img_002.png,invalid,0.7416

Model Details

Backbone: torchvision.models.resnet50

Head: Dropout(0.5) + Linear(num_features -> 2)

Preprocessing:

Resize to 224×224

Normalize mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]

Reproducibility & Class Mapping

Default class order (if not inferred): ["invalid", "valid"].

When available, subfolder names under Dataset/train/ define the class order (alphabetical).

For strict control, always pass --classes "invalid,valid".

Training (Optional Outline)

Training code is not included here. A typical setup uses:

ImageFolder with Dataset/train and Dataset/val

Weighted cross-entropy if classes are imbalanced

Early stopping based on validation loss

Metric tracking: accuracy, precision, recall, F1

Troubleshooting

No images found: verify files in ./Test and supported extensions (.jpg .jpeg .png .bmp .tif .tiff).

Shape/size issues: script resizes to 224×224; custom preprocessing must match training.

Wrong labels: pass explicit class order via --classes.

CUDA errors: test with --device cpu to isolate GPU/driver issues.

Citation

If this classifier supports a publication or report, cite the repository and associated paper/manuscript as appropriate.

License

Add a license file (e.g., MIT) if distribution is intended.
