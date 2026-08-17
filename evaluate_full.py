import torch
import os
import csv
import numpy as np
from double_dqn import DQNAgent
from quantum_env import QuantumCircuitEnv
from model_and_vqe import QuantumAttentionAutoencoder,hamiltonian_energy_matrix_for_h2,energy_calculator

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("1. Loading Trained RL Agent...")
    env = QuantumCircuitEnv(num_qubits = 4)
    rl_agent = DQNAgent(state_dim = 16,action_dim = 4)
    rl_agent.policy_net.load_state_dict(torch.load("dueling_dqn_quantum.pt",map_location = device))
    rl_agent.policy_net.to(device)
    rl_agent.policy_net.eval()

    state = env.reset()
    if (isinstance(state,torch.Tensor)):
            state = state.detach().cpu().numpy().flatten()
        
    done = False
    while not done:
            action = rl_agent.select_action(state,epsilon = 0.0)
            next_state,reward,done = env.step(action)
            if (isinstance(next_state,torch.Tensor)):
                    next_state = next_state.detach().cpu().numpy().flatten()
            state = next_state
                
    raw_vector = torch.tensor(state,dtype=torch.float32,device = device).unsqueeze(0)

    transformer = QuantumAttentionAutoencoder(input_dim = 16,embed_dim = 32,num_heads = 4).to(device)
    transformer.load_state_dict(torch.load("quantum_transformer_autoencoder.pt",map_location = device))
    transformer.eval()
        
    exact_fci_energies_dict = {0.3: -0.60180, 0.5: -1.05510,0.735: -1.13730,1.0: -1.10110,1.5: -1.01300,2.0: -0.94160}   
        
    with torch.no_grad():
        no_noise_vector = transformer(raw_vector)
        bond_distances = [0.3,0.5,0.735,1.0,1.5,2.0]
        csv_filename = "benchmark_results.csv"
        file_exists = os.path.isfile(csv_filename)
        
        print("\n===========================================================")
        print("📊 HYBRID RL + TRANSFORMER BENCHMARK")
        print("===========================================================")
        print("Source     | Bond (Å) | Raw Energy (Ha) | Mitigated Energy (Ha)")
        print("-----------------------------------------------------------")

        with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(["Pipeline_Type", "Bond_Distance_A", "True_Clean_Energy_Ha", "Raw_Energy_Ha", "Mitigated_Energy_Ha", "Raw_Error_Ha", "Mitigated_Error_Ha", "Noise_Reduction_Pct"])
                
                for i in bond_distances:
                    h2_energies = hamiltonian_energy_matrix_for_h2(bond_distance = i)
                    h2_diag_energies = torch.tensor(h2_energies,dtype = torch.float32,device = device)
                    raw_energy = energy_calculator(raw_vector,h2_diag_energies)
                    no_noise_energy = energy_calculator(no_noise_vector,h2_diag_energies)

                    raw_val = raw_energy.item() if isinstance(raw_energy, torch.Tensor) else raw_energy
                    mit_val = no_noise_energy.item() if isinstance(no_noise_energy, torch.Tensor) else no_noise_energy

                    true_fci_energy = exact_fci_energies_dict.get(i,-1.13730)
                    raw_err = np.abs(raw_val - true_fci_energy)
                    mit_err = np.abs(mit_val - true_fci_energy) 
                    pct_reduction = ((raw_err - mit_err) / raw_err * 100.0) if raw_err != 0 else 0.0
                    
                    print(f"RL_Hybrid  |   {i:5.3f}  |   {raw_val:12.5f}  |   {mit_val:12.5f}")
                    writer.writerow(["RL_Hybrid", i, true_fci_energy, raw_val, mit_val, raw_err, mit_err, pct_reduction])

        print("\n===========================================================")
        print("✅ Run Complete. Results saved to 'benchmark_results.csv'.")
        print("===========================================================\n")
                    
    print("===========================================================")