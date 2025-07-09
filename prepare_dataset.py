import os
import shutil
import random
from sklearn.model_selection import train_test_split

def prepare_dataset(raw_dir, output_dir, test_size=0.15, val_size=0.15):
    """
    Organizes raw dataset into train/val/test directories
    """
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'train', 'NORMAL'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train', 'PNEUMONIA'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'NORMAL'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', 'PNEUMONIA'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'NORMAL'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', 'PNEUMONIA'), exist_ok=True)

    # Get all image paths
    normal_images = []
    pneumonia_images = []
    
    for root, _, files in os.walk(raw_dir):
        for file in files:
            if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                if 'NORMAL' in root:
                    normal_images.append(os.path.join(root, file))
                elif 'PNEUMONIA' in root:
                    pneumonia_images.append(os.path.join(root, file))

    print(f"Found {len(normal_images)} normal images")
    print(f"Found {len(pneumonia_images)} pneumonia images")

    # Split normal images
    normal_train, normal_test = train_test_split(
        normal_images, test_size=test_size, random_state=42
    )
    normal_train, normal_val = train_test_split(
        normal_train, test_size=val_size/(1-test_size), random_state=42
    )

    # Split pneumonia images
    pneumonia_train, pneumonia_test = train_test_split(
        pneumonia_images, test_size=test_size, random_state=42
    )
    pneumonia_train, pneumonia_val = train_test_split(
        pneumonia_train, test_size=val_size/(1-test_size), random_state=42
    )

    # Copy files to new structure
    def copy_files(files, category, dataset_type):
        for file in files:
            filename = os.path.basename(file)
            dest = os.path.join(output_dir, dataset_type, category, filename)
            shutil.copy2(file, dest)
    
    # Copy normal images
    copy_files(normal_train, 'NORMAL', 'train')
    copy_files(normal_val, 'NORMAL', 'val')
    copy_files(normal_test, 'NORMAL', 'test')
    
    # Copy pneumonia images
    copy_files(pneumonia_train, 'PNEUMONIA', 'train')
    copy_files(pneumonia_val, 'PNEUMONIA', 'val')
    copy_files(pneumonia_test, 'PNEUMONIA', 'test')
    
    print("Dataset preparation complete!")
    print(f"Train: {len(normal_train)} normal, {len(pneumonia_train)} pneumonia")
    print(f"Val: {len(normal_val)} normal, {len(pneumonia_val)} pneumonia")
    print(f"Test: {len(normal_test)} normal, {len(pneumonia_test)} pneumonia")

if __name__ == "__main__":
    # Update these paths based on your environment
    raw_dataset_dir = "path/to/your/raw/dataset"
    organized_dataset_dir = "../chest_xray"
    
    prepare_dataset(raw_dataset_dir, organized_dataset_dir)