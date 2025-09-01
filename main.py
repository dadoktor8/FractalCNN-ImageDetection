import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import os
import glob
from PIL import Image
import time

from AIDetectionDataset import CustomAIDetectionDataset
import FractalTrainer
from Fractal_CNN import FractalCNN

# Dataset paths
TRAIN_PATH = r"D:\Projects\Generalizable AI-Generated Image Detection Based on Fractal Self-Similarity in the Spectrum\FractalCNN-ImageDetection\archive\Dataset\Train"
TEST_PATH = r"D:\Projects\Generalizable AI-Generated Image Detection Based on Fractal Self-Similarity in the Spectrum\FractalCNN-ImageDetection\archive\Dataset\Test"
VALIDATION_PATH = r"D:\Projects\Generalizable AI-Generated Image Detection Based on Fractal Self-Similarity in the Spectrum\FractalCNN-ImageDetection\archive\Dataset\Validation"

# Dataset class (fixed naming)
def load_dataset_from_path(dataset_path, max_images_per_class=None):
    """Load dataset from a specific path"""
    print(f"📂 Loading from: {dataset_path}")
    
    real_path = os.path.join(dataset_path, "Real")
    fake_path = os.path.join(dataset_path, "Fake") 
    
    if not os.path.exists(real_path) or not os.path.exists(fake_path):
        print(f"❌ Real or Fake folder not found in {dataset_path}")
        return None, None
        
    real_images = []
    fake_images = []
    
    # Load real images
    real_files = glob.glob(os.path.join(real_path, "*"))
    if max_images_per_class:
        real_files = real_files[:max_images_per_class]
    
    print(f"   Loading {len(real_files)} real images...")
    for i, img_path in enumerate(real_files):
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            if max(img.size) > 512:
                img = img.resize((256, 256), Image.Resampling.LANCZOS)
            
            img_array = np.array(img)
            real_images.append(img_array)
            
            if (i + 1) % 1000 == 0:
                print(f"      Real: {i+1}/{len(real_files)}")
                
        except Exception as e:
            continue
    
    # Load fake images
    fake_files = glob.glob(os.path.join(fake_path, "*"))
    if max_images_per_class:
        fake_files = fake_files[:max_images_per_class]
    
    print(f"   Loading {len(fake_files)} fake images...")
    for i, img_path in enumerate(fake_files):
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            if max(img.size) > 512:
                img = img.resize((256, 256), Image.Resampling.LANCZOS)
            
            img_array = np.array(img)
            fake_images.append(img_array)
            
            if (i + 1) % 1000 == 0:
                print(f"      Fake: {i+1}/{len(fake_files)}")
                
        except Exception as e:
            continue
    
    print(f"   ✅ Loaded {len(real_images)} real, {len(fake_images)} fake")
    return real_images, fake_images

def load_dataset_from_path(dataset_path, max_images_per_class=None):
    """Load dataset from a specific path"""
    print(f"📂 Loading from: {dataset_path}")
    
    real_path = os.path.join(dataset_path, "Real")
    fake_path = os.path.join(dataset_path, "Fake") 
    
    if not os.path.exists(real_path) or not os.path.exists(fake_path):
        print(f"❌ Real or Fake folder not found in {dataset_path}")
        return None, None
        
    real_images = []
    fake_images = []
    
    # Load real images
    real_files = glob.glob(os.path.join(real_path, "*"))
    if max_images_per_class:
        real_files = real_files[:max_images_per_class]
    
    print(f"   Loading {len(real_files)} real images...")
    for i, img_path in enumerate(real_files):
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            if max(img.size) > 512:
                img = img.resize((256, 256), Image.Resampling.LANCZOS)
            
            img_array = np.array(img)
            real_images.append(img_array)
            
            if (i + 1) % 1000 == 0:
                print(f"      Real: {i+1}/{len(real_files)}")
                
        except Exception as e:
            continue
    
    # Load fake images
    fake_files = glob.glob(os.path.join(fake_path, "*"))
    if max_images_per_class:
        fake_files = fake_files[:max_images_per_class]
    
    print(f"   Loading {len(fake_files)} fake images...")
    for i, img_path in enumerate(fake_files):
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            if max(img.size) > 512:
                img = img.resize((256, 256), Image.Resampling.LANCZOS)
            
            img_array = np.array(img)
            fake_images.append(img_array)
            
            if (i + 1) % 1000 == 0:
                print(f"      Fake: {i+1}/{len(fake_files)}")
                
        except Exception as e:
            continue
    
    print(f"   ✅ Loaded {len(real_images)} real, {len(fake_images)} fake")
    return real_images, fake_images

def check_all_datasets():
    """Check all three dataset folders"""
    print("📋 Checking All Dataset Folders")
    print("=" * 40)
    
    datasets = {
        "Train": TRAIN_PATH,
        "Test": TEST_PATH, 
        "Validation": VALIDATION_PATH
    }
    
    all_good = True
    for name, path in datasets.items():
        print(f"\n📂 {name}: {path}")
        
        if not os.path.exists(path):
            print(f"   ❌ Path doesn't exist!")
            all_good = False
            continue
            
        real_path = os.path.join(path, "Real")
        fake_path = os.path.join(path, "Fake")
        
        if os.path.exists(real_path) and os.path.exists(fake_path):
            real_count = len(glob.glob(os.path.join(real_path, "*")))
            fake_count = len(glob.glob(os.path.join(fake_path, "*")))
            print(f"   ✅ Real: {real_count}, Fake: {fake_count}")
        else:
            print(f"   ❌ Missing Real/Fake folders")
            all_good = False
    
    return all_good

def check_train_dataset():
    """Check just the train dataset"""
    print("📋 Checking Train Dataset")
    print("=" * 30)
    
    print(f"📂 Train: {TRAIN_PATH}")
    
    if not os.path.exists(TRAIN_PATH):
        print(f"   ❌ Path doesn't exist!")
        return False
        
    real_path = os.path.join(TRAIN_PATH, "Real")
    fake_path = os.path.join(TRAIN_PATH, "Fake")
    
    if os.path.exists(real_path) and os.path.exists(fake_path):
        real_count = len(glob.glob(os.path.join(real_path, "*")))
        fake_count = len(glob.glob(os.path.join(fake_path, "*")))
        print(f"   ✅ Real: {real_count}, Fake: {fake_count}")
        print(f"   📊 Total available: {real_count + fake_count}")
        return True
    else:
        print(f"   ❌ Missing Real/Fake folders")
        return False

def get_file_lists_and_split(dataset_path, max_total_images=3000):
    """Get file paths and split them BEFORE loading images"""
    print(f"📂 Analyzing dataset structure: {dataset_path}")
    print(f"🎯 Limiting to {max_total_images} total images")
    
    real_path = os.path.join(dataset_path, "Real")
    fake_path = os.path.join(dataset_path, "Fake")
    
    # Get all file paths (not loading images yet!)
    real_files = glob.glob(os.path.join(real_path, "*"))
    fake_files = glob.glob(os.path.join(fake_path, "*"))
    
    print(f"   Found {len(real_files)} real files")
    print(f"   Found {len(fake_files)} fake files")
    
    # Calculate how many per class for the limit
    max_per_class = max_total_images // 2  # 1500 per class for 3000 total
    
    # Balance the dataset and apply limit
    min_samples = min(len(real_files), len(fake_files), max_per_class)
    real_files = real_files[:min_samples]
    fake_files = fake_files[:min_samples]
    
    total_limited = min_samples * 2
    print(f"   Limited to {min_samples} per class ({total_limited} total)")
    print(f"   📊 Using {total_limited}/{len(glob.glob(os.path.join(real_path, '*'))) + len(glob.glob(os.path.join(fake_path, '*')))} available images")
    
    # Split file paths (70/15/15)
    from sklearn.model_selection import train_test_split
    
    real_train_files, real_temp_files = train_test_split(real_files, test_size=0.3, random_state=42)
    fake_train_files, fake_temp_files = train_test_split(fake_files, test_size=0.3, random_state=42)
    
    real_val_files, real_test_files = train_test_split(real_temp_files, test_size=0.5, random_state=42)
    fake_val_files, fake_test_files = train_test_split(fake_temp_files, test_size=0.5, random_state=42)
    
    splits = {
        'train': {'real': real_train_files, 'fake': fake_train_files},
        'val': {'real': real_val_files, 'fake': fake_val_files},
        'test': {'real': real_test_files, 'fake': fake_test_files}
    }
    
    print(f"\n📈 File splits (from {total_limited} images):")
    for split_name, split_data in splits.items():
        real_count = len(split_data['real'])
        fake_count = len(split_data['fake'])
        print(f"   {split_name.upper()}: {real_count} real + {fake_count} fake = {real_count + fake_count}")
    
    return splits

def load_images_from_file_list(file_list, split_name):
    """Load images from a list of file paths (memory efficient)"""
    print(f"🔄 Loading {split_name} images: {len(file_list)} files...")
    
    images = []
    failed_count = 0
    
    for i, img_path in enumerate(file_list):
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize for memory efficiency
            
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
            
            img_array = np.array(img)
            images.append(img_array)
            
            # Progress every 1000 images
            if (i + 1) % 1000 == 0:
                print(f"      {split_name}: {i+1}/{len(file_list)} ({((i+1)/len(file_list)*100):.1f}%)")
                
        except Exception as e:
            failed_count += 1
            continue
    
    if failed_count > 0:
        print(f"      ⚠️ Skipped {failed_count} corrupted files")
    
    print(f"   ✅ Loaded {len(images)} {split_name} images")
    return images

def train_fractal_cnn():
    """Main training function - using ONLY Train folder, split for everything"""
    print("🚀 FRACTAL-CNN TRAINING (Using Train folder for everything)")
    print("=" * 50)
    
    # Check train dataset
    if not check_train_dataset():
        print("❌ Dataset check failed!")
        return
    
    print("\n🔄 Step 1: Analyzing and splitting file paths...")
    # Get file paths and split them (no loading yet)
    file_splits = get_file_lists_and_split(TRAIN_PATH, max_total_images=3000)
    
    if not file_splits:
        print("❌ Failed to split files!")
        return
    
    print("\n❓ This will use your Train folder for train/val/test. Continue? (y/n): ", end="")
    response = input().strip().lower()
    
    if response != 'y':
        print("Training cancelled.")
        return
    
    print("\n🔄 Step 2: Loading training images...")
    # Load training images
    real_train = load_images_from_file_list(file_splits['train']['real'], 'real_train')
    fake_train = load_images_from_file_list(file_splits['train']['fake'], 'fake_train')
    
    print("\n🔄 Step 3: Loading validation images...")
    # Load validation images (from same Train folder)
    real_val = load_images_from_file_list(file_splits['val']['real'], 'real_val')
    fake_val = load_images_from_file_list(file_splits['val']['fake'], 'fake_val')
    
    print("\n🔄 Step 4: Loading test images...")
    # Load test images (from same Train folder)
    real_test = load_images_from_file_list(file_splits['test']['real'], 'real_test')
    fake_test = load_images_from_file_list(file_splits['test']['fake'], 'fake_test')
    
    print(f"\n📊 Final Dataset Summary (all from Train folder):")
    print(f"   Train: {len(real_train)} real + {len(fake_train)} fake = {len(real_train) + len(fake_train)}")
    print(f"   Val: {len(real_val)} real + {len(fake_val)} fake = {len(real_val) + len(fake_val)}")
    print(f"   Test: {len(real_test)} real + {len(fake_test)} fake = {len(real_test) + len(fake_test)}")
    
    total_images = len(real_train) + len(fake_train) + len(real_val) + len(fake_val) + len(real_test) + len(fake_test)
    print(f"   🎯 TOTAL: {total_images} images from Train folder!")
    
    # Create datasets
    print(f"\n🔄 Step 5: Creating PyTorch datasets...")
    train_dataset = CustomAIDetectionDataset(real_train, fake_train)
    val_dataset = CustomAIDetectionDataset(real_val, fake_val)
    test_dataset = CustomAIDetectionDataset(real_test, fake_test)
    
    # Create data loaders
    batch_size = 16 if torch.cuda.is_available() else 8
    num_workers = 4 if torch.cuda.is_available() else 2
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    print(f"\n🔧 Training Configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Estimate time based on dataset size
    if total_images < 10000:
        time_est = "2-6 hours"
        epochs = 25
        lr = 0.0001
    elif total_images < 50000:
        time_est = "6-15 hours"
        epochs = 35
        lr = 0.00005
    else:
        time_est = "15-48 hours"
        epochs = 50
        lr = 0.00005
    
    print(f"   Estimated time: {time_est}")
    print(f"   Epochs: {epochs}")
    
    # Initialize model
    print(f"\n🧠 Initializing FractalCNN...")
    model = FractalCNN(num_fractal_levels=3, hidden_channels=32, num_classes=2)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = FractalTrainer.FractalTrainer(model, device=device, lr=lr)
    
    print(f"   Device: {device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Learning rate: {lr}")
    
    # Train
    print(f"\n🎯 Starting training on {total_images} images...")
    print(f"   💡 All data from Train folder, split 70/15/15")
    
    start_time = time.time()
    
    try:
        best_accuracy = trainer.train(train_loader, val_loader, num_epochs=epochs)
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user!")
        best_accuracy = 0.0
    
    training_time = time.time() - start_time
    
    # Test if training completed
    if best_accuracy > 0:
        print(f"\n🧪 Final testing on {len(real_test) + len(fake_test)} test images...")
        test_loss, test_accuracy = trainer.validate(test_loader)
    else:
        test_accuracy = 0.0
    
    # Results
    print(f"\n🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"   Training Time: {training_time/3600:.2f} hours")
    print(f"   Dataset: Train folder only ({total_images} images)")
    print(f"   Split: 70% train, 15% val, 15% test")
    
    if best_accuracy > 0:
        print(f"   Best Val Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Save model
        model_name = f"fractal_cnn_{total_images}imgs_{best_accuracy:.4f}.pth"
        torch.save(model.state_dict(), model_name)
        print(f"   Model saved: {model_name}")
        
        # Performance assessment
        if test_accuracy > 0.85:
            print(f"   🔥 AMAZING! Research-quality results!")
        elif test_accuracy > 0.75:
            print(f"   🎯 EXCELLENT! Major improvement from 62.5% baseline!")
        elif test_accuracy > 0.65:
            print(f"   📈 Good! Better than your 62.5% baseline!")
        else:
            print(f"   📊 Results obtained - room for improvement!")
        
        try:
            trainer.plot_training_history()
        except:
            print("   (Plotting skipped)")
    
    print(f"\n✅ DONE! Used only Train folder for everything! 🎯")
    return model, trainer, test_accuracy if best_accuracy > 0 else None

if __name__ == "__main__":
    print("🔥 FractalCNN Training Script")
    print("Make sure FractalCNN and FractalTrainer classes are imported!")
    print()
    
    # Run training
    train_fractal_cnn()