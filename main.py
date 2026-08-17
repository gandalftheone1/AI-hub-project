import random
import csv
import os
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from generated_and_noised_quantum_circuits import quantum_circuit_generator
from model_and_vqe import QuantumAttentionAutoencoder,hamiltonian_energy_matrix_for_h2,energy_calculator,chemical_accuracy_evaluation

    
def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = QuantumAttentionAutoencoder(input_dim = 16,embed_dim = 32,num_heads = 4).to(device)
    model_path = "quantum_transformer_autoencoder.pt"
    model.load_state_dict(torch.load(model_path,map_location = device)) 
    model.eval()
    
    noise_test,no_noise_test = quantum_circuit_generator(samples = 200 ,num_qubits = 4,max_depth = 8,shots = 1024)
    noise_test = noise_test.to(device)
    no_noise_test = no_noise_test.to(device)
    
    with torch.no_grad():
        denoised_predictions = model(noise_test)
    
        noisy_err, noisy_acc = chemical_accuracy_evaluation(noise_test,no_noise_test)
        mean_err,mean_acc = chemical_accuracy_evaluation(denoised_predictions,no_noise_test)
        bond_distances = [0.3,0.5,0.735,1.0,1.5,2.0]
        
        csv_filename = "benchmark_results.csv"
        file_exists = os.path.isfile(csv_filename)
    
        with open(csv_filename,mode = 'a',newline ='') as file:
            writer = csv.writer(file)
            if not file_exists:
                    writer.writerow(["Pipeline_Type", "Bond_Distance_A", "True_Clean_Energy_Ha", "Raw_Energy_Ha", "Mitigated_Energy_Ha", "Raw_Error_Ha", "Mitigated_Error_Ha", "Noise_Reduction_Pct"])
            
            for i in bond_distances:
                h2_diag = torch.tensor(hamiltonian_energy_matrix_for_h2(bond_distance = i),dtype = torch.float32,device = device)
                
                noise_energy_temp = torch.mean(energy_calculator(prob_vector = noise_test,energies_vector = h2_diag))
                noise_energy = noise_energy_temp.item()
                no_noise_energy_temp= torch.mean(energy_calculator(prob_vector = no_noise_test,energies_vector = h2_diag))
                no_noise_energy = no_noise_energy_temp.item()
                error = np.abs(noise_energy - no_noise_energy)
                
                mean_energy_temp = torch.mean(energy_calculator(prob_vector = denoised_predictions,energies_vector = h2_diag))
                mean_energy = mean_energy_temp.item()
                mean_error = np.abs(mean_energy - no_noise_energy)
                noise_reduction_pct = ((error - mean_error) / error) * 100.0 if error != 0 else 0.0
                
                if mean_error <= 0.0016:
                        status = "🌟 CHEMICAL ACCURACY"
                elif mean_error <= 0.010:
                        status = "✅ NISQ-ACCURATE (<10 mHa)"
                else:
                        status = "⚠️ HIGH NOISE"
    
                print(f"\n[Bond Distance: {i:.3f} Å]")
                print(f"  • True Clean Energy : {no_noise_energy:+.5f} Ha")
                print(f"  • Raw Noisy Error   : {error:.6f} Ha")
                print(f"  • Mitigated Error   : {mean_error:.6f} Ha | {status}")
                print(f"  • Noise Eliminated  : {noise_reduction_pct:.2f}%")
                    
                print(f"Supervised |   {i:5.3f}  |   {noise_energy:12.5f}  |   {mean_energy:12.5f}")

                writer.writerow(["Supervised", i, no_noise_energy, noise_energy, mean_energy, error, mean_error, noise_reduction_pct])

    print("\n===========================================================")
    print("✅ Run Complete. Results saved to 'benchmark_results.csv'.")
    print("===========================================================\n")
        
    
if __name__ == "__main__":
    main() 
    