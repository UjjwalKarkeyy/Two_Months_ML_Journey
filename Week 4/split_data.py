import os
import random
import shutil
from pathlib import Path

# --- CONFIG ---
# Path to the folder containing all images (original + augmented)
SRC_FULL = Path('./data/PlantVillage/train_before')

# Path to folder containing only original (non-augmented) images
SRC_ORIG = Path('./data/PlantVillage/valid_before')

# Destination folder for splits
DST = Path('./data/PlantVillage/split_data')

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# For reproducibility
random.seed(42)

# --- CREATE OUTPUT FOLDERS ---
for split in ['train', 'valid', 'test']:
    (DST / split).mkdir(parents=True, exist_ok=True)

# --- TRAIN + VALID SPLIT FROM SRC_FULL ---
for cls in SRC_FULL.iterdir():
    if not cls.is_dir():
        continue

    print(f"Processing class: {cls.name}")

    # Create class folders
    (DST/'train'/cls.name).mkdir(parents=True, exist_ok=True)
    (DST/'valid'/cls.name).mkdir(parents=True, exist_ok=True)
    (DST/'test'/cls.name).mkdir(parents=True, exist_ok=True)

    # Get all image files in class folder
    all_imgs = [f for f in cls.glob('*') if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    random.shuffle(all_imgs)

    n = len(all_imgs)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    # Copy training images
    for f in all_imgs[:train_end]:
        shutil.copy2(f, DST/'train'/cls.name/f.name)

    # Copy validation images
    for f in all_imgs[train_end:val_end]:
        shutil.copy2(f, DST/'valid'/cls.name/f.name)

# --- TEST SPLIT FROM SRC_ORIG (originals only) ---
for cls in SRC_ORIG.iterdir():
    if not cls.is_dir():
        continue

    print(f"Processing test class: {cls.name}")

    (DST/'test'/cls.name).mkdir(parents=True, exist_ok=True)

    all_imgs = [f for f in cls.glob('*') if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    random.shuffle(all_imgs)

    test_count = int(len(all_imgs) * TEST_RATIO)

    for f in all_imgs[:test_count]:
        shutil.copy2(f, DST/'test'/cls.name/f.name)

print("\n✅ Dataset split complete!")
print(f"Train, validation, and test sets created in: {DST}")
