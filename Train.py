import csv
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import precision_score, recall_score, f1_score
import datetime


def main():
    # -------------------- CONFIG --------------------
    batch_size = 32
    max_epochs = 500
    learning_rate = 1e-4
    weight_decay = 1e-4
    data_dir = "Dataset"

    # -------------------- DEVICE --------------------
    if not torch.cuda.is_available():
        raise SystemError("❌ No GPU found! Please ensure CUDA is available.")
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")

    # -------------------- DATA TRANSFORMS --------------------
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # -------------------- DATASET --------------------
    image_datasets = {
        'train': datasets.ImageFolder(data_dir + '/train', data_transforms['train']),
        'val': datasets.ImageFolder(data_dir + '/val', data_transforms['val'])
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(f"Classes: {class_names}")
    print(f"Training samples: {dataset_sizes['train']}, Validation samples: {dataset_sizes['val']}")

    # -------------------- MODEL --------------------
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, len(class_names))
    )
    model = model.to(device)
    print("Model on device:", next(model.parameters()).device)

    # -------------------- LOSS / OPTIMIZER / SCHEDULER --------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # -------------------- CSV LOGGING --------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file_path = f"resnet50_training_{timestamp}.csv"
    with open(csv_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Epoch", "Train Loss", "Val Loss", "Train Acc", "Val Acc",
            "Precision", "Recall", "F1", "Learning Rate", "Epoch Time (s)", "Best Val Acc"
        ])

    # -------------------- TRAINING LOOP --------------------
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    since = time.time()

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")
        print('-' * 20)
        epoch_start = time.time()

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0
            all_preds, all_labels = [], []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'val':
                precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
                recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
                f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            else:
                precision = recall = f1 = 0.0

            print(f"{phase} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            # Save best model when validation accuracy improves
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), f"best_model_epoch_{epoch+1}.pth")
                print(f"✅ New best model saved (Acc: {best_acc:.4f})")

        # Save per-epoch metrics
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        with open(csv_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, epoch_loss, epoch_loss, epoch_acc.item(), epoch_acc.item(),
                precision, recall, f1, current_lr, epoch_time, best_acc.item()
            ])

    # -------------------- WRAP UP --------------------
    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'resnet50_final_model.pth')
    print(f"✅ Training done. Metrics saved to {csv_file_path}")


if __name__ == "__main__":
    main()
