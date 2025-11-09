# infer.py
import argparse
import os
import glob
import csv
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

IM_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def load_class_names(dataset_train_dir: str | None, classes_arg: str | None):
    # 1) Try to read from Dataset/train subfolders (the ImageFolder convention)
    if dataset_train_dir and os.path.isdir(dataset_train_dir):
        sub = [d for d in sorted(os.listdir(dataset_train_dir))
               if os.path.isdir(os.path.join(dataset_train_dir, d))]
        if sub:
            print(f"Class names from '{dataset_train_dir}': {sub}")
            return sub
    # 2) Fallback: parse from --classes "invalid,valid"
    if classes_arg:
        classes = [c.strip() for c in classes_arg.split(",") if c.strip()]
        if len(classes) >= 2:
            print(f"Class names from --classes: {classes}")
            return classes
    # 3) Final fallback: assume binary (invalid/valid)
    print("Could not infer classes; defaulting to ['invalid', 'valid']. Use --classes to override.")
    return ["invalid", "valid"]

def build_model(num_classes: int):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

def load_weights(model, weights_path, device):
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    print(f"Loaded weights: {weights_path}")
    return model

def predict_folder(model, transform, image_dir, class_names, device, csv_path):
    files = []
    for ext in IM_EXTS:
        files.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
    files = sorted(files)

    if not files:
        print(f"No images found in '{image_dir}'. Supported: {', '.join(IM_EXTS)}")
        return

    softmax = torch.nn.Softmax(dim=1)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    preds_out = []

    with torch.no_grad(), open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted_class", "confidence"])

        for fp in files:
            try:
                img = Image.open(fp).convert("RGB")
            except Exception as e:
                print(f"Skipping '{fp}': {e}")
                continue

            x = transform(img).unsqueeze(0).to(device)
            logits = model(x)
            probs = softmax(logits).cpu().squeeze(0).tolist()
            pred_idx = int(torch.argmax(logits, dim=1).item())
            pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
            conf = probs[pred_idx] if pred_idx < len(probs) else float("nan")

            writer.writerow([os.path.basename(fp), pred_name, f"{conf:.4f}"])
            preds_out.append((fp, pred_name, conf))

    print(f"\nSaved predictions to: {csv_path}")
    # Quick preview
    for fp, pred_name, conf in preds_out[:10]:
        print(f"{os.path.basename(fp)} -> {pred_name} ({conf:.3f})")
    if len(preds_out) > 10:
        print(f"... and {len(preds_out) - 10} more.")

def main():
    parser = argparse.ArgumentParser(description="Infer classes for images in ./Test")
    parser.add_argument("--weights", type=str, default="resnet50_final_model.pth",
                        help="Path to model weights (.pth).")
    parser.add_argument("--test_dir", type=str, default="Test",
                        help="Folder with test images.")
    parser.add_argument("--dataset_train_dir", type=str, default="Dataset/train",
                        help="Training folder to infer class names (subfolders).")
    parser.add_argument("--classes", type=str, default=None,
                        help="Comma-separated class names, e.g. 'invalid,valid'.")
    parser.add_argument("--device", type=str, default="auto",
                        help="'cpu', 'cuda', or 'auto'.")
    parser.add_argument("--out_csv", type=str, default="predictions.csv",
                        help="Output CSV path.")
    args = parser.parse_args()

    # Device
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class_names = load_class_names(args.dataset_train_dir, args.classes)
    model = build_model(len(class_names))
    model = load_weights(model, args.weights, device)
    transform = get_transform()

    predict_folder(model, transform, args.test_dir, class_names, device, args.out_csv)

if __name__ == "__main__":
    main()
