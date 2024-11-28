import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import *
from net import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


adj_matrix = create_adjacency_matrix()
keypoints_dir = "keypoints_dataset"
label_map = {"true": 0, "false": 1}
dataset = KeypointDataset(keypoints_dir, label_map)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


model = GCN(in_features=3, hidden_features=64, out_features=2, device=device)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20 
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for keypoints, label in dataloader:

        keypoints = keypoints.squeeze(0).to(device) 
        label = label.to(device)

        optimizer.zero_grad()

        logits = model(keypoints, adj_matrix)
        loss = criterion(logits.unsqueeze(0), label)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

        pred = logits.argmax(dim=0).item()
        correct += (pred == label.item())
        total += 1

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {correct / total:.4f}")

torch.save(model.state_dict(), "model.pth")