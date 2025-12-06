import os
import shutil
import random
from tqdm import tqdm

# ---- Correct TB dataset root ----
TB_ROOT = "/Users/pradumnpandey/Downloads/Dataset of Tuberculosis Chest X-rays Images"
TB_SOURCE = os.path.join(TB_ROOT, "TB Chest X-rays")

# ---- Your project dataset ----
BASE_DIR = "chest_xray"

# Use EXISTING TB folders (do NOT create them)
train_tb = os.path.join(BASE_DIR, "train", "TB")
val_tb = os.path.join(BASE_DIR, "valid", "TB")
test_tb = os.path.join(BASE_DIR, "test", "TB")

# Debug: Show folders (to verify names)
print("Using TB folders:")
print(train_tb)
print(val_tb)
print(test_tb)

# Load TB images
images = [
    img for img in os.listdir(TB_SOURCE)
    if img.lower().endswith((".jpg", ".jpeg", ".png"))
]

random.shuffle(images)

total = len(images)
train_end = int(total * 0.7)
val_end = train_end + int(total * 0.15)

train_imgs = images[:train_end]
val_imgs = images[train_end:val_end]
test_imgs = images[val_end:]

def copy_images(img_list, dest):
    for img in tqdm(img_list, desc=f"Copying → {dest}"):
        src = os.path.join(TB_SOURCE, img)
        dst = os.path.join(dest, img)

        # Safety: Only copy if file doesn’t exist
        if not os.path.exists(dst):
            shutil.copy(src, dst)

copy_images(train_imgs, train_tb)
copy_images(val_imgs, val_tb)
copy_images(test_imgs, test_tb)

print("\nTB dataset copied successfully! 🎉")
