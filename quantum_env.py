import torch
import numpy as np

class QuantumCircuitEnv:
    def __init__(self,num_qubits: int = 4,max_depth: int = 8):
        self.num_qubits = num_qubits
        self.max_depth = max_depth
        self.state_dim = 16
        self.action_dim = 4
        self.reset()
        
    def reset(self):
        self.current_step = 0
        self.state = np.ones(self.state_dim) / self.state_dim
        
        return torch.tensor(self.state,dtype = torch.float32)

    def step(self,action):
       self.current_step+=1
       
       action_effect = np.zeros(self.state_dim)
       action_effect[action % self.state_dim] += 0.2
       
       noise = np.random.normal(0,0.05,size = self.state_dim)
       self.temp_state  = np.clip(self.state + action_effect + noise,0,1)
       state_sum = np.sum(self.temp_state)
       self.state = self.temp_state / state_sum if state_sum > 0 else np.ones(self.state_dim) / self.state_dim
       
       target_state  = np.zeros(self.state_dim)
       target_state[0] = 1.0
       
       reward = float(np.dot(self.state,target_state))
       done = self.current_step >= self.max_depth
       
       return torch.tensor(self.state,dtype = torch.float32),reward,done