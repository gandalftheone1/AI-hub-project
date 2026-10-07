import csv
import os
import numpy as np
import torch

from rl_training import DQNAgent
from quantum_env import QuantumCircuitEnv  
from model_and_vqe import (QuantumAttentionAutoencoder,energy_calculator,hamiltonian_energy_matrix_for_h2,)

def apply_qpu_noise(clean_state,noise_level=0.15):
    noise = np.random.normal(0,noise_level,size=clean_state.shape)
    noisy_state = clean_state + noise
    noisy_state = np.abs(noisy_state)
    noisy_state = noisy_state / np.sum(noisy_state,axis=-1,keepdims=True)
    return noisy_state

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = QuantumAttentionAutoencoder(input_dim=16,embed_dim=32,num_heads=4).to(device)
    autoencoder_path = "quantum_transformer_autoencoder.pt"
    if os.path.exists(autoencoder_path):
        autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))
    autoencoder.eval()

    bond_distances = [0.3, 0.5, 0.735, 1.0, 1.25, 1.5, 1.75, 2.0]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_filename = os.path.join(script_dir,"benchmark_results_technique2.csv")
    noise_level = 0.15  
    rl_agent = DQNAgent(state_dim=16, action_dim=4)
    rl_agent_path = "dueling_dqn_quantum.pt"
    
    if os.path.exists(rl_agent_path):
        state_dict = torch.load(rl_agent_path, map_location=device)
        if hasattr(rl_agent, 'policy_net'):
            rl_agent.policy_net.load_state_dict(state_dict)
            rl_agent.policy_net.eval()
        else: rl_agent.load_state_dict(state_dict)
    else: raise FileNotFoundError(f"❌ Checkpoint '{rl_agent_path}' not found! Run training first.")

    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Pipeline_Type","Bond_Distance_A", "True_Clean_Energy_Ha","Raw_Energy_Ha","Mitigated_Energy_Ha","Raw_Error_Ha","Mitigated_Error_Ha","Noise_Reduction_Pct","Raw_TVD","Mitigated_TVD"])
        file.flush()
        
        for dist in bond_distances:
          try:  
            h2_matrix = hamiltonian_energy_matrix_for_h2(bond_distance=dist)
            h2_diag = torch.tensor(h2_matrix, dtype=torch.float32, device=device)
            fci_ground_truth = float(np.min(h2_matrix))

            env = QuantumCircuitEnv(max_depth=100)
            state_tensor = env.reset(bond_distance=dist)

            if isinstance(state_tensor,torch.Tensor): state = state_tensor.detach().cpu().numpy().flatten()
            else: state = np.array(state_tensor).flatten()

            done = False
            step_count = 0
            while not done and step_count < 100:
                action = rl_agent.select_action(state,epsilon=0.0)  
                next_state_tensor, _, done = env.step(action)
                
                if isinstance(next_state_tensor, torch.Tensor): state = next_state_tensor.detach().cpu().numpy().flatten()
                else: state = np.array(next_state_tensor).flatten()
                step_count += 1

            clean_rl_state = state  
            noisy_rl_state_np = apply_qpu_noise(clean_rl_state, noise_level=noise_level)
            noisy_rl_state_tensor = torch.tensor(noisy_rl_state_np, dtype=torch.float32, device=device).unsqueeze(0)  
            clean_rl_state_tensor = torch.tensor(clean_rl_state,dtype=torch.float32,device=device).unsqueeze(0) 

            with torch.no_grad():
                mitigated_rl_state_tensor = autoencoder(noisy_rl_state_tensor)

            raw_energy = energy_calculator(prob_vector=noisy_rl_state_tensor,energies_vector=h2_diag).item()
            mitigated_energy = energy_calculator(prob_vector=mitigated_rl_state_tensor,energies_vector=h2_diag).item()
            
            raw_error = abs(raw_energy - fci_ground_truth)
            mit_error = abs(mitigated_energy - fci_ground_truth)

            noise_reduction_pct = (((raw_error - mit_error) / raw_error) * 100.0 if raw_error != 0 else 0.0)

            raw_tvd = (
                0.5 * torch.sum(torch.abs(noisy_rl_state_tensor - clean_rl_state_tensor)).item()
            )
            mit_tvd = (0.5 * torch.sum(torch.abs(mitigated_rl_state_tensor - clean_rl_state_tensor)).item())
            
            writer.writerow(["Hybrid_RL_Transformer",dist,fci_ground_truth,raw_energy,mitigated_energy,raw_error,mit_error,noise_reduction_pct,raw_tvd,mit_tvd,])
          except Exception as e:
                raise e

if __name__ == "__main__":
    main()