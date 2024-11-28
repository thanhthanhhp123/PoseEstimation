import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, device):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.01)
        self.device = device
        self.to(self.device)
        nn.init.xavier_uniform_(self.weight)
        

    def forward(self, x, adjacency_matrix):
        """
        Args:
            x (torch.Tensor): Node features of shape (N, F_in), where N is the number of nodes.
            adjacency_matrix (torch.Tensor): Adjacency matrix of shape (N, N).
        
        Returns:
            torch.Tensor: Output features of shape (N, F_out).
        """
        # Normalize adjacency matrix
        degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))
        degree_inv_sqrt = torch.pow(degree_matrix, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0  # Handle division by zero
        normalized_adj = degree_inv_sqrt @ adjacency_matrix @ degree_inv_sqrt

        normalized_adj = normalized_adj.to(self.device)
        # Graph convolution
        out = normalized_adj @ x @ self.weight
        return out

class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, device):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_features, hidden_features, device)
        self.gcn2 = GCNLayer(hidden_features, hidden_features, device)
        self.fc = nn.Linear(hidden_features, out_features)
        self.device = device
        self.to(self.device)

    def forward(self, x, adjacency_matrix):
        """
        Args:
            x (torch.Tensor): Node features of shape (N, F_in).
            adjacency_matrix (torch.Tensor): Adjacency matrix of shape (N, N).
        
        Returns:
            torch.Tensor: Class logits of shape (1, C), where C is the number of classes.
        """
        x = F.relu(self.gcn1(x, adjacency_matrix))
        x = F.relu(self.gcn2(x, adjacency_matrix))
        x = x.mean(dim=0)  # Global pooling (average over all nodes)
        x = self.fc(x)
        return x
