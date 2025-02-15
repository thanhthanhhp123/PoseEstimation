import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Example for filtering UserWarnings

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, device):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        nn.init.xavier_uniform_(self.weight)
        self.device = device
        self.dropout = nn.Dropout(0.3)  # Dropout for regularization
        self.batch_norm = nn.BatchNorm1d(out_features)  # Batch Normalization
        self.layer_norm = nn.LayerNorm(out_features)  # Layer Normalization

    def forward(self, x, adjacency_matrix):
        degree_matrix = torch.sum(adjacency_matrix, dim=-1)  # Shape: (B, N)
        degree_inv_sqrt = torch.pow(degree_matrix, -0.5).unsqueeze(-1)  # Shape: (B, N, 1)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0  # Handle division by zero
        normalized_adj = adjacency_matrix * degree_inv_sqrt * degree_inv_sqrt.transpose(-1, -2)

        # Graph convolution
        normalized_adj = normalized_adj.to(self.device)
        out = normalized_adj @ x @ self.weight
        
        if out.shape[0] == 1:
            out = out.squeeze(0)

        # Apply batch normalization, layer normalization, and dropout
        try:
            B, N, F = out.shape
        except ValueError:
            B = 1
            N, F = out.shape
        out = self.batch_norm(out.view(-1, F)).view(B, N, F)  # BatchNorm for graph data
        out = self.layer_norm(out)  # LayerNorm for graph data
        out = self.dropout(out)
        return out


class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, device):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_features, hidden_features, device)
        self.gcn2 = GCNLayer(hidden_features, hidden_features, device)
        self.gcn3 = GCNLayer(hidden_features, hidden_features, device)  # Additional GCN layer
        self.gcn4 = GCNLayer(hidden_features, hidden_features, device)  # New GCN layer
        self.attention = nn.MultiheadAttention(hidden_features, num_heads=4, batch_first=True)  # Attention mechanism
        self.fc1 = nn.Linear(hidden_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)  # New fully connected layer
        self.fc3 = nn.Linear(hidden_features, out_features)

    def forward(self, x, adjacency_matrix):
        x = F.relu(self.gcn1(x, adjacency_matrix))
        x = F.relu(self.gcn2(x, adjacency_matrix))
        x = F.relu(self.gcn3(x, adjacency_matrix))
        x = F.relu(self.gcn4(x, adjacency_matrix))  # Forward pass through new GCN layer

        # Apply attention mechanism
        x, _ = self.attention(x, x, x)

        # Global pooling (average pooling over nodes for each graph)
        x = x.mean(dim=1)

        # Fully connected layers with nonlinearity
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))  # Forward pass through new fully connected layer
        x = self.fc3(x)
        return x