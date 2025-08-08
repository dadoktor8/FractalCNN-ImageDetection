import os
import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader

from AIDetectionDataset import AIDetectionDataset
from FractalTrainer import FractalTrainer
from Fractal_CNN import FractalCNN

def load_local_dataset(dataset_path, max_images=100):
    """Simple function to load images from your local dataset"""
    real_path = os.path.join(dataset_path, "Real")
    fake_path = os.path.join(dataset_path, "Fake")
    
    print(f"Loading from: {dataset_path}")
    
    # Load real images
    real_images = []
    real_files = glob.glob(os.path.join(real_path, "*"))[:max_images]
    for img_path in real_files:
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            real_images.append(np.array(img))
        except:
            continue
    
    # Load fake images  
    fake_images = []
    fake_files = glob.glob(os.path.join(fake_path, "*"))[:max_images]
    for img_path in fake_files:
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            fake_images.append(np.array(img))
        except:
            continue
            
    print(f"Loaded {len(real_images)} real and {len(fake_images)} fake images")
    return real_images, fake_images

def test_complete_model():
    """Test the complete Fractal CNN"""
    print("Testing Complete Fractal CNN Implementation")
    print("=" * 50)
    
    # YOUR DATASET PATH - CHANGE THIS
    dataset_path = r"D:\Projects\Generalizable AI-Generated Image Detection Based on Fractal Self-Similarity in the Spectrum\archive\Dataset\Train"
    
    # Load real dataset instead of dummy data
    try:
        real_images, fake_images = load_local_dataset(dataset_path, max_images=50)
        if len(real_images) == 0 or len(fake_images) == 0:
            print("❌ No images found, using dummy data instead")
            # Fallback to dummy data
            dummy_real = [np.random.rand(224, 224, 3) * 255 for _ in range(20)]
            dummy_fake = [np.random.rand(224, 224, 3) * 255 for _ in range(20)]
            dataset = AIDetectionDataset(dummy_real, dummy_fake)
        else:
            # Use real dataset
            dataset = AIDetectionDataset(real_images, fake_images)
            print("✅ Using real dataset")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Create test data for model testing
    batch_size = 4
    test_images = torch.randn(batch_size, 1, 224, 224)
    test_labels = torch.randint(0, 2, (batch_size,))
    
    # Initialize model
    model = FractalCNN(num_fractal_levels=3, hidden_channels=32)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(test_images)
        print(f"Input shape: {test_images.shape}")
        print(f"Output shape: {outputs.shape}")
        print(f"Output probabilities: {torch.softmax(outputs, dim=1)}")
    
    # Test training step with real data
    trainer = FractalTrainer(model, device='cpu')
    
    print("\nTesting training step with real data...")
    model.train()
    for images, labels in dataloader:
        outputs = model(images)
        loss = trainer.criterion(outputs, labels)
        print(f"Training loss: {loss.item():.4f}")
        print(f"Real data batch shape: {images.shape}")
        break
    
    print("\n✅ Complete model test passed!")
    return model, trainer

def train_full_model():
    """Train the complete Fractal CNN on your dataset"""
    print("\n🚀 Starting Full Training!")
    print("=" * 50)
    
    # YOUR DATASET PATH
    dataset_path = r"D:\Projects\Generalizable AI-Generated Image Detection Based on Fractal Self-Similarity in the Spectrum\archive\Dataset\Train"
    
    # Load more data for training (increase from 50 to 400 per class)
    real_images, fake_images = load_local_dataset(dataset_path, max_images=400)
    
    if len(real_images) == 0 or len(fake_images) == 0:
        print("❌ No images found!")
        return
    
    # Create train/validation split (80/20)
    from sklearn.model_selection import train_test_split
    
    real_train, real_val = train_test_split(real_images, test_size=0.2, random_state=42)
    fake_train, fake_val = train_test_split(fake_images, test_size=0.2, random_state=42)
    
    print(f"Training: {len(real_train)} real + {len(fake_train)} fake")
    print(f"Validation: {len(real_val)} real + {len(fake_val)} fake")
    
    # Create datasets
    train_dataset = AIDetectionDataset(real_train, fake_train)
    val_dataset = AIDetectionDataset(real_val, fake_val)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Initialize model and trainer
    model = FractalCNN(num_fractal_levels=3, hidden_channels=32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = FractalTrainer(model, device=device, lr=0.001)
    
    print(f"Using device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train the model
    print("\n🎯 Training started...")
    best_accuracy = trainer.train(train_loader, val_loader, num_epochs=20)
    
    # Plot results
    print("\n📊 Plotting training history...")
    trainer.plot_training_history()
    
    print(f"\n✅ Training completed! Best accuracy: {best_accuracy:.4f}")
    return model, trainer

if __name__ == "__main__":
    # First test the model
    model, trainer = test_complete_model()
    
    # Then train it fully
    trained_model, trained_trainer = train_full_model()