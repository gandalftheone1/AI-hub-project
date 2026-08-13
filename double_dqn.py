import torch
import torch.nn as nn

class DuelingDQN(nn.Module):
    def __init__(self,state_dim: int,action_dim: int):
        super(DuelingDQN,self).__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU()
        )
        
        self.value_stream = nn.Linear(128,1)
        
        self.advantage_stream = nn.Linear(128,action_dim)
        
    def forward(self,state: torch.Tensor)-> torch.Tensor:
        features = self.feature_layer(state)
        
        val = self.value_stream(features)
        adv = self.advantage_stream(features)
        
        return val + (adv - adv.mean(dim=-1, keepdim = True))
        