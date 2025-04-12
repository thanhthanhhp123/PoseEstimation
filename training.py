import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from utils import *
from net import *

# Configure environment
os.environ.update({
    'TF_CPP_MIN_LOG_LEVEL': '3',
    'MEDIAPIPE_DISABLE_GPU': '1',
    'OMP_NUM_THREADS': '2',
    'MKL_NUM_THREADS': '2',
    'CUDA_LAUNCH_BLOCKING': '1'
})

class FocalLoss(nn.Module):
    def __init__(self, alpha=2, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.setup_seeds()
        self.setup_data()
        self.setup_model()
        self.setup_training()
        
    def setup_seeds(self):
        torch.manual_seed(self.config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        
    def setup_data(self):
        # Create full dataset
        train_dataset = KeypointDataset(
            self.config['data_dir'],
            self.config['label_map'],
            split='train',
            train_ratio=0.8,
        )
        val_dataset = KeypointDataset(
            self.config['data_dir'],
            self.config['label_map'],
            split='val',
            train_ratio=0.8,
        )
        test_dataset = KeypointDataset(
            self.config['data_dir'],
            self.config['label_map'],
            split='test',
            train_ratio=0.8,
        )
        self.train_dataset, self.val_dataset, self.test_dataset = train_dataset, val_dataset, test_dataset
        
    
        
        print(f"Dataset sizes - Train: {train_dataset.__len__()}, Val: {val_dataset.__len__()}, Test: {test_dataset.__len__()}")
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2
        )
        
    def setup_model(self):
        # Initialize model with reduced complexity
        self.model = GCN(
            in_features=2,
            hidden_features=256,  # Reduced from 64
            out_features=2,
            device=self.device
        ).to(self.device)
        
        # Create and move adjacency matrix to device
        self.adj_matrix = create_optimized_adjacency_matrix().to(self.device)
        
        # Calculate class weights
        labels = [label for _, label in self.train_dataset]
        class_counts = torch.bincount(torch.tensor(labels))
        self.class_weights = len(labels) / (2 * class_counts.float())
        self.class_weights = self.class_weights.to(self.device)
        
    def setup_training(self):
        # Initialize focal loss with class weights
        self.criterion = FocalLoss(alpha=2, gamma=2)
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        predictions = []
        targets = []
        
        for batch_idx, (keypoints, labels) in enumerate(self.train_loader):
            keypoints = keypoints.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(keypoints, self.adj_matrix)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
            if batch_idx % 10 == 0:
                print(f"Batch [{batch_idx}/{len(self.train_loader)}], Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = np.mean(np.array(predictions) == np.array(targets))
        return epoch_loss, epoch_acc
    
    def evaluate(self, data_loader):
        self.model.eval()
        running_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_idx, (keypoints, labels) in enumerate(data_loader):
                keypoints = keypoints.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(keypoints, self.adj_matrix)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(data_loader)
        epoch_acc = np.mean(np.array(predictions) == np.array(targets))
        return epoch_loss, epoch_acc, predictions, targets

    def train(self):
        best_val_acc = 0
        patience_counter = 0
        training_start = time.time()
        
        for epoch in range(self.config['num_epochs']):
            epoch_start = time.time()
            
            # Training phase
            train_loss, train_acc = self.train_epoch()
            
            # Validation phase
            val_loss, val_acc, _, _ = self.evaluate(self.val_loader)
            
            # Update learning rate
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print epoch results
            print(f"\nEpoch [{epoch+1}/{self.config['num_epochs']}] - {time.time()-epoch_start:.2f}s")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_checkpoint(epoch, val_acc)
                print(f"Saved best model with val_acc: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print("Early stopping triggered!")
                    break
        
        # Final evaluation
        print("\nTraining completed!")
        print(f"Total training time: {(time.time()-training_start)/60:.2f} minutes")
        
        # Load best model and evaluate
        self.load_best_model()
        self.final_evaluation()
    
    def save_checkpoint(self, epoch, val_acc):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc
        }
        torch.save(checkpoint, "best_model.pth")
        print("Checkpoint saved!")
    
    def load_best_model(self):
        checkpoint = torch.load("best_model.pth", weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Best model loaded!")
    
    def final_evaluation(self):
        test_loss, test_acc, predictions, targets = self.evaluate(self.test_loader)
        
        # Print classification report
        print("\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        print(classification_report(targets, predictions, target_names=self.config['class_names']))
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        print("Confusion Matrix:")
        print(cm)   
    

if __name__ == "__main__":
    config = {
        'data_dir': "keypoints_dataset",
        'label_map': {"true": 0, "false": 1},
        'class_names': ['True Pose', 'False Pose'],
        'batch_size': 256,  # Reduced batch size
        'learning_rate': 0.0005,  # Reduced learning rate
        'weight_decay': 0.02,
        'num_epochs': 150,
        'patience': 15,
        'grad_clip': 0.5,
        'seed': 42
    }
    
    trainer = Trainer(config)
    trainer.train()