import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
from torch.utils.data import Dataset, DataLoader
import cv2 
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import os
from PIL import Image

class FractalTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_index,(images,labels) in enumerate(train_loader):
            images,labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs,labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_index % 10 == 0:
                print(f'Batch {batch_index}/{len(train_loader)}, Loss: {loss.item():.4f}')
                
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self,val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images,labels in val_loader:
                images,labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs,labels)
                
                total_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu(). numpy())
                all_labels.extend(labels.cpu().numpy())
                
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels,all_preds)
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss,accuracy
    
    def train(self, train_loader, val_loader, num_epochs=20):
        print("Training Begins.....")
        
        
        best_val_acc = 0
        patience = 5
        patience_counter = 0
        
        for epochs in range(num_epochs):
            print(f"\nEpoch {epochs+1}/{num_epochs}")
            print("_" * 30)
            
            
            train_loss = self.train_epoch(train_loader)
            val_loss,val_acc = self.validate(val_loader)
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                torch.save(self.model.state_dict(), 'best_fractal_model_adaptive_pooling.pth')
                print(f"New best validation accuracy: {best_val_acc:.4f}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early Stopping After {epochs + 1} epochs")
                break 
        print(f"\nTraining Completed. Best Validation Accuracy: {best_val_acc:.4f}")
        return best_val_acc
    
    def plot_training_history(self):
        epochs  = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(12,4))
        
        plt.subplot(1,2,1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1,2,2)
        plt.plot(epochs, self.val_accuracies, 'g-', label='Validation Accuracy')
        plt.title('Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()