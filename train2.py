"""
Voxera — train2.py (Unified Preprocessing & Training)
LOCAL WINDOWS VERSION
Features: No Rotation, Temporal Cropping, TQDM ETA Bars, 35-Class Auto-Detect
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg") # Prevents Windows plotting crashes
import matplotlib.pyplot as plt
import cv2
import glob
import random

# ─────────────────────────────────────────────────────────────
# 1. CONFIGURATION (Local Windows Paths)
# ─────────────────────────────────────────────────────────────
# Reverted back to your D: Drive folders!
RAW_VIDEO_DIR    = r"D:\voxera_data\finnal_data\Master_Training_Data" 
FINAL_IMAGE_DIR  = r"D:\voxera_data\finnal_data\finnalimage"      
OUTPUT_DIR       = r"D:\voxera_data\2_Model_Training\Model_2_Test"
MODEL_PATH       = os.path.join(OUTPUT_DIR, "best_test_model.pth")
CURVES_PATH      = os.path.join(OUTPUT_DIR, "test_training_curves.png")

# Extraction Settings 
FRAMES_PER_VIDEO = 30         
TRIM_SECONDS     = 3          # Cut 3 seconds from start and end to avoid empty hands
AUGMENTATIONS    = 3   
ALLOW_FLIP       = False      

# Training Settings
IMAGE_SIZE   = 224      
BATCH_SIZE   = 32       
EPOCHS       = 25       
LR           = 1e-3
VAL_SPLIT    = 0.20
PATIENCE     = 7        
SEED         = 42
NUM_WORKERS  = 0        # Keep at 0 for Windows to prevent multiprocessing freezing

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FINAL_IMAGE_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*60}")
print(f"  VOXERA CNN — Local Training Initialized (No Rotation)")
print(f"{'='*60}")
print(f"  Device : {device}")
if device.type == "cuda":
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────────────────────
# 2. DATA PREPROCESSING (Video to Image Pipeline)
# ─────────────────────────────────────────────────────────────
def add_noise(image):
    row, col, ch = image.shape
    var = random.uniform(10, 50)
    sigma = var ** 0.5
    gauss = np.random.normal(0, sigma, (row, col, ch))
    return np.clip(image + gauss, 0, 255).astype(np.uint8)

def change_brightness(image):
    value = random.randint(-50, 50)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    if value > 0:
        v = cv2.add(v, value)
    else:
        v = cv2.subtract(v, -value)
        
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

def random_shift(image):
    # Pure translation (shifting x and y) with NO rotation
    rows, cols = image.shape[:2]
    tx = random.randint(-int(cols*0.15), int(cols*0.15))
    ty = random.randint(-int(rows*0.15), int(rows*0.15))
    
    # Create the translation matrix
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, M, (cols, rows), borderValue=(128, 128, 128))

def prepare_dataset():
    print(f"\n[1/6] Checking Dataset Preprocessing...")
    subfolders = [f.path for f in os.scandir(RAW_VIDEO_DIR) if f.is_dir()]
    
    needs_extraction = False
    for folder in subfolders:
        class_name = os.path.basename(folder)
        final_class_dir = os.path.join(FINAL_IMAGE_DIR, class_name)
        if not os.path.exists(final_class_dir) or len(os.listdir(final_class_dir)) == 0:
            needs_extraction = True
            break
            
    if not needs_extraction:
        print("  -> Images already exist in Final Image folder. Skipping extraction.")
        return

    print(f"  -> Extracting with TEMPORAL CROPPING (removing empty hands)...")
    
    for folder in tqdm(subfolders, desc="Overall Progress", colour='yellow'):
        class_name = os.path.basename(folder)
        final_class_dir = os.path.join(FINAL_IMAGE_DIR, class_name)
        os.makedirs(final_class_dir, exist_ok=True)
        
        videos = glob.glob(os.path.join(folder, "*.[mM][pP]4")) + glob.glob(os.path.join(folder, "*.[mM][oO][vV]"))
        
        if not videos:
            continue
            
        for vid_path in tqdm(videos, desc=f"Processing '{class_name}'", leave=False, colour='magenta'):
            vid_name = os.path.splitext(os.path.basename(vid_path))[0]
            cam = cv2.VideoCapture(vid_path)
            
            total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cam.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0 or fps == 0: 
                cam.release()
                continue
            
            # --- SMART TEMPORAL CROPPING LOGIC ---
            frames_to_trim = int(fps * TRIM_SECONDS)
            
            if total_frames <= (frames_to_trim * 2):
                start_frame = int(total_frames * 0.15)
                end_frame = int(total_frames * 0.85)
            else:
                start_frame = frames_to_trim
                end_frame = total_frames - frames_to_trim
                
            usable_frames = end_frame - start_frame
            interval = max(1, usable_frames // FRAMES_PER_VIDEO)
            
            cam.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frame_idx = start_frame
            extracted_count = 0
            
            while frame_idx <= end_frame:
                success, frame = cam.read()
                if not success: break
                    
                if (frame_idx - start_frame) % interval == 0 and extracted_count < FRAMES_PER_VIDEO:
                    cv2.imwrite(os.path.join(final_class_dir, f"{class_name}_{vid_name}_f{frame_idx}_clean.jpg"), frame)
                    
                    for i in range(AUGMENTATIONS):
                        aug_img = frame.copy()
                        if random.random() > 0.5: aug_img = change_brightness(aug_img)
                        if random.random() > 0.5: aug_img = add_noise(aug_img)
                        # Applied pure shift instead of shift+rotate
                        aug_img = random_shift(aug_img) 
                        if ALLOW_FLIP and random.random() > 0.5: aug_img = cv2.flip(aug_img, 1)
                        cv2.imwrite(os.path.join(final_class_dir, f"{class_name}_{vid_name}_f{frame_idx}_aug{i}.jpg"), aug_img)
                        
                    extracted_count += 1
                frame_idx += 1
            cam.release()
    print("  -> Extraction Complete!")

prepare_dataset()

# ─────────────────────────────────────────────────────────────
# 3. PYTORCH DATA LOADING & MODEL BUILDING
# ─────────────────────────────────────────────────────────────
print(f"\n[2/6] Loading dataset from: {FINAL_IMAGE_DIR}")

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Ensured PyTorch transforms also have 0 degrees of rotation
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), 
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

full_dataset = datasets.ImageFolder(root=FINAL_IMAGE_DIR)
classes      = full_dataset.classes
NUM_CLASSES  = len(classes)
all_labels   = [label for _, label in full_dataset.samples]

print(f"  Total images : {len(full_dataset)}")
print(f"  Classes      : {NUM_CLASSES} -> Automatically detected classes!")

splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=SEED)
train_idx, val_idx = next(splitter.split(np.zeros(len(all_labels)), all_labels))

class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset, self.transform = subset, transform
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img) if self.transform else img, label

base_dataset = datasets.ImageFolder(root=FINAL_IMAGE_DIR, transform=transforms.Lambda(lambda x: x))
train_subset = TransformSubset(Subset(base_dataset, train_idx), train_transforms)
val_subset   = TransformSubset(Subset(base_dataset, val_idx), val_transforms)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"))
val_loader   = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"))

print(f"\n[3/6] Building ResNet18 Model...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in model.parameters(): param.requires_grad = False
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, NUM_CLASSES),
)
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5, min_lr=1e-6)

# ─────────────────────────────────────────────────────────────
# 4. TRAINING LOOP WITH TQDM BARS
# ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{epochs} [Train]", leave=False, colour='green')
    
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{correct/total:.4f}")
    return running_loss / total, correct / total

def validate(model, loader, criterion, device, epoch, epochs):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d}/{epochs} [Val]  ", leave=False, colour='blue')
    
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{correct/total:.4f}")
    return running_loss / total, correct / total

def unfreeze_backbone(model, optimizer, new_lr=1e-4):
    print(f"\n  >> Unfreezing backbone for fine-tuning (lr={new_lr})")
    for param in model.parameters(): param.requires_grad = True
    optimizer.add_param_group({"params": [p for name, p in model.named_parameters() if "fc" not in name], "lr": new_lr})

print(f"\n[4/6] Training for {EPOCHS} epochs...")
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc, epochs_no_improve, UNFREEZE_EPOCH = 0.0, 0, 5

for epoch in range(1, EPOCHS + 1):
    if epoch == UNFREEZE_EPOCH + 1: unfreeze_backbone(model, optimizer, new_lr=1e-4)

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, EPOCHS)
    val_loss, val_acc     = validate(model, val_loader, criterion, device, epoch, EPOCHS)
    scheduler.step(val_acc)
    
    for k, v in zip(history.keys(), [train_loss, train_acc, val_loss, val_acc]): history[k].append(v)

    if val_acc > best_val_acc:
        best_val_acc, best_epoch, epochs_no_improve = val_acc, epoch, 0
        torch.save({"epoch": epoch, "model_state": model.state_dict(), "classes": classes, "num_classes": NUM_CLASSES}, MODEL_PATH)
        tag = "  ** BEST **"
    else:
        epochs_no_improve += 1
        tag = ""

    print(f"Epoch {epoch:2d}/{EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Loss: {val_loss:.4f}{tag}")
    if epochs_no_improve >= PATIENCE:
        print(f"\n  EarlyStopping triggered.")
        break

print(f"\n[5/6] Saving training curves...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history["train_acc"], label="Train acc"); axes[0].plot(history["val_acc"], label="Val acc"); axes[0].legend()
axes[1].plot(history["train_loss"], label="Train loss"); axes[1].plot(history["val_loss"], label="Val loss"); axes[1].legend()
plt.savefig(CURVES_PATH); plt.close()

print(f"\n[6/6] Done! Best Val Acc: {best_val_acc:.4f} (epoch {best_epoch}).")