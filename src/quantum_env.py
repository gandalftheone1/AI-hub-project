import numpy as np
import torch


class QuantumCircuitEnv:

    def __init__(self, num_qubits: int = 4, max_depth: int = 100):
        self.num_qubits = num_qubits
        self.max_depth = max_depth
        self.state_dim = 16
        self.action_dim = 4
        self.bond_distance = 0.735
        self.reset(bond_distance=0.735)

    def reset(self, bond_distance: float = 0.735):
        from model_and_vqe import hamiltonian_energy_matrix_for_h2
        self.current_step = 0
        self.bond_distance = bond_distance
        
        self.state = np.ones(self.state_dim) / np.sqrt(self.state_dim)
        
        self.h2_diag = hamiltonian_energy_matrix_for_h2(bond_distance=bond_distance)
        
        return torch.tensor(self.state, dtype=torch.float32)

    def step(self, action):
        self.current_step += 1
        target_idx = action % self.state_dim
        
        probs = np.abs(self.state) ** 2
        probs[target_idx] += 1.0  
        
        probs = probs / np.sum(probs)
        
        noise = np.random.normal(0, 0.015, size=self.state_dim)
        noisy_probs = np.maximum(probs + noise, 1e-8)
        noisy_probs /= np.sum(noisy_probs)
        
        self.state = np.sqrt(noisy_probs)

        current_energy = float(np.sum(noisy_probs * self.h2_diag))
        fci_target = float(np.min(self.h2_diag))
        
        energy_error = abs(current_energy - fci_target)
        reward = - (energy_error ** 2) * 20.0 

        done = self.current_step >= self.max_depth
        return torch.tensor(noisy_probs, dtype=torch.float32), reward, done