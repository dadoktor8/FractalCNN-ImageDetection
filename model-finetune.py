import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from Fractal_CNN import FractalCNN
from AIDetectionDataset import CustomAIDetectionDataset
import FractalTrainer
from main import get_file_lists_and_split, load_images_from_file_list

def continue_training(pretrained_model_path, new_data_path, num_epochs=20):
    model = FractalCNN(num_fractal_levels=3, hidden_channels=32, num_classes=2)
    model.load_state_dict(torch.load(pretrained_model_path))
    file_splits = get_file_lists_and_split(new_data_path, max_total_images=1200)
    real_train = load_images_from_file_list(file_splits['train']['real'], 'real_train')
    fake_train = load_images_from_file_list(file_splits['train']['fake'], 'fake_train')
    real_val = load_images_from_file_list(file_splits['val']['real'], 'real_val')
    fake_val = load_images_from_file_list(file_splits['val']['fake'], 'fake_val')
    
    train_dataset = CustomAIDetectionDataset(real_train, fake_train)
    val_dataset = CustomAIDetectionDataset(real_val,fake_val)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    print(f"✅ New data loaded: {len(train_dataset)} train, {len(val_dataset)} val")
    
    device = 'cuda'
    trainer = FractalTrainer.FractalTrainer(model, device=device, lr=0.0001)
    
    print(f"Strating Fine-Tuning with {num_epochs} epochs")
    best_accuracy = trainer.train(train_loader,val_loader,num_epochs=num_epochs)
    improved_model_name = f"fractal_cnn_finetuned_{best_accuracy:.4f}.pth"
    torch.save(model.state_dict(), improved_model_name)
    
    print(f"\n FINE-TUNING COMPLETE!")
    print(f" Improved model saved: {improved_model_name}")
    print(f" Best accuracy: {best_accuracy:.4f}")
    
    return model, trainer

if __name__ == "__main__":
    
    # Your existing model
    PRETRAINED_MODEL = "fractal_cnn_3000imgs_0.7689.pth"
    
    # Path to NEW data (same structure: Real/Fake folders)
    NEW_DATA_PATH = r"D:\Projects\Generalizable AI-Generated Image Detection Based on Fractal Self-Similarity in the Spectrum\archive\data\train"
    
    # Continue training
    improved_model, trainer = continue_training(
        PRETRAINED_MODEL, 
        NEW_DATA_PATH, 
        num_epochs=20
    )