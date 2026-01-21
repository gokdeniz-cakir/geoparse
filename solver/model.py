
import torch
import torch.nn as nn

class NeuralSolver(nn.Module):
    """
    Neural Solver with Problem Type Embedding.
    Input: 
      - x: 8 dim coordinates
      - type_idx: Problem type index (0-19)
    Output: 1 dim scalar
    """
    def __init__(self, input_dim=8, num_types=25, embedding_dim=16, hidden_dim=128, output_dim=1):
        super().__init__()
        
        self.type_embedding = nn.Embedding(num_types, embedding_dim)
        
        # Input to MLP is coords + embedding
        mlp_input_dim = input_dim + embedding_dim
        
        self.net = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, inputs):
        # inputs is a tuple/list: (x, type_idx)
        x, type_idx = inputs
        
        # Embed type
        emb = self.type_embedding(type_idx) # [B, 16]
        
        # Concatenate: [B, 8] + [B, 16] -> [B, 24]
        combined = torch.cat([x, emb], dim=1)
        
        return self.net(combined)
