import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, device):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        nn.init.xavier_uniform_(self.weight)
        self.device = device

    def forward(self, x, adjacency_matrix):
        """
        Args:
            x (torch.Tensor): Node features of shape (B, N, F_in), where B is batch size.
            adjacency_matrix (torch.Tensor): Adjacency matrix of shape (B, N, N).
        
        Returns:
            torch.Tensor: Output features of shape (B, N, F_out).
        """
        # Normalize adjacency matrix
        degree_matrix = torch.sum(adjacency_matrix, dim=-1)  # Shape: (B, N)
        degree_inv_sqrt = torch.pow(degree_matrix, -0.5).unsqueeze(-1)  # Shape: (B, N, 1)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0  # Handle division by zero
        normalized_adj = adjacency_matrix * degree_inv_sqrt * degree_inv_sqrt.transpose(-1, -2)

        # Graph convolution
        normalized_adj = normalized_adj.to(self.device)
        out = normalized_adj @ x @ self.weight
        return out

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, device):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_features, hidden_features, device)
        self.gcn2 = GCNLayer(hidden_features, hidden_features, device)
        self.fc = nn.Linear(hidden_features, out_features)

    def forward(self, x, adjacency_matrix):
        """
        Args:
            x (torch.Tensor): Node features of shape (B, N, F_in).
            adjacency_matrix (torch.Tensor): Adjacency matrix of shape (B, N, N).
        
        Returns:
            torch.Tensor: Class logits of shape (B, C), where C is the number of classes.
        """
        x = F.relu(self.gcn1(x, adjacency_matrix))
        x = F.relu(self.gcn2(x, adjacency_matrix))
        x = x.mean(dim=1)  # Global pooling (average over nodes for each graph)
        x = self.fc(x)
        return x
