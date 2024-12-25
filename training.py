import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import time

from utils import *
from net import *

import warnings 
warnings.filterwarnings("ignore")


def train():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    adj_matrix = create_adjacency_matrix()
    keypoints_dir = "keypoints_dataset_new"
    label_map = {"true": 0, "false": 1}
    dataset = KeypointDataset(keypoints_dir, label_map)
    # dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=2)
    # test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=2)

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    


    model = GCN(in_features=3, hidden_features=64, out_features=2, device=device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Starting training...")
    num_epochs = 50
    start = time.time()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for keypoints, labels in train_dataloader:

            keypoints = keypoints.to(device) 
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(keypoints, adj_matrix)
            loss = criterion(logits, labels)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        torch.save(model.state_dict(), f"models/new_model_{epoch}.pth")

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {correct / total:.4f}")
    end = time.time()
    print(f"Training took {end - start:.2f} seconds")
    torch.save(model.state_dict(), "models/new_model.pth")

if __name__ == "__main__":
    train()