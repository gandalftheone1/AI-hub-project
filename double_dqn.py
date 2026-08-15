import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

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

class ReplayBuffer:
    def __init__(self,capacity: int = 10000):
        self.buffer = deque(maxlen = capacity)
        
    def push(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))
        
    def sample(self,batch_size: int):
        state,action,reward,next_state,done = zip(*random.sample(self.buffer,batch_size))

        def to_float_tensor(data_list):
            cleaned = [s.numpy().flatten() if isinstance(s,torch.Tensor) else np.array(s).flatten() for s in data_list]
            return torch.tensor(np.array(cleaned),dtype = torch.float32)
        
        return(
        to_float_tensor(state),
        torch.tensor(action,dtype = torch.float32),
        torch.tensor(reward,dtype = torch.float32),
        to_float_tensor(next_state),
        torch.tensor(done,dtype = torch.bool) 
    )
    
    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    def __init__(self,state_dim: int,action_dim: int,lr: float = 1e-3,gamma: float = 0.99,buffer_capacity: int = 10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        self.policy_net = DuelingDQN(state_dim,action_dim)
        self.target_net = DuelingDQN(state_dim,action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(),lr = lr)
        self.memory = ReplayBuffer(buffer_capacity)
        
    def select_action(self,state: np.ndarray,epsilon: float)->int:
        
        if (random.random() < epsilon):
            return (random.randint(0,self.action_dim-1))
        
        state_t = torch.tensor(state,dtype  = torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
            return torch.argmax(q_values,dim = 1).item()
        
    def update(self,batch_size: int)->float:
        
        if (len(self.memory) < batch_size):
            return 0.0
            
        states,action,rewards,next_states,dones = self.memory.sample(batch_size)
        
        q_values = self.policy_net(states)
        action = action.to(torch.long)
        state_action_values = q_values.gather(1,action.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim = 1,keepdim = True)
            next_q_values = self.target_net(next_states).gather(1,next_actions).squeeze(1)
            target_q_values = rewards + (1.0 - dones.float()) * self.gamma * next_q_values
            
        loss = nn.MSELoss()(state_action_values,target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self): 
        self.target_net.load_state_dict(self.policy_net.state_dict())