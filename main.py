import random
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from generated_and_noised_quantum_circuits import quantum_circuit_generator
from model_and_vqe import QuantumAttentionAutoencoder,hamiltonian_energy_matrix_for_h2,energy_calculator,chemical_accuracy_evaluation

    
def main():
    
    print("==================================================")
    print(" QUANTUM TRANSFORMER INFERENCE & VQE EVALUATION")
    print("==================================================\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = QuantumAttentionAutoencoder(input_dim = 16,embed_dim = 32,num_heads = 4)
    model_path = "quantum_transformer_autoencoder.pt"
    model.load_state_dict(torch.load(model_path,map_location = device)) 
    model.eval()
    
    print(f"✅ Successfully loaded model weights from '{model_path}'!\n")

    # 3. Generate UNSEEN Test Dataset (200 new noisy samples)
    print("Generating unseen noisy test dataset (200 samples)...")
    
    noise_test,no_noise_test = quantum_circuit_generator(samples = 200 ,num_qubits = 4,max_depth = 8,shots = 1024)
    noise_test = noise_test.to(device)
    no_noise_test = no_noise_test.to(device)
    
    with torch.no_grad():
       denoised_predictions = model(noise_test)
       
    print("\n--- PERFORMANCE EVALUATION ON UNSEEN DATA ---")
    
    
    noisy_err, noisy_acc = chemical_accuracy_evaluation(noise_test,no_noise_test)
    print(f"❌ Raw Unmitigated Input:")
    print(f"   - Mean Energy Error: {noisy_err:.6f} Hartree")
    print(f"   - Chemical Accuracy Rate: {noisy_acc:.1f}%")

    
    mean_err,mean_acc = chemical_accuracy_evaluation(denoised_predictions,no_noise_test)
    print(f"\n✅ Transformer Mitigated Output:")
    print(f"   - Mean Energy Error: {mean_err:.6f} Hartree")
    print(f"   - Chemical Accuracy Rate: {mean_acc:.1f}%")


    print("\n--- DISSOCIATION CURVE BENCHMARK (H2 Bond Stretching) ---")
    bond_distances = [0.3,0.5,0.735,1.0,1.5,2.0]
    
    for i in bond_distances:
        h2_diag = torch.tensor(hamiltonian_energy_matrix_for_h2(bond_distance = i))
        
        noise_energy = torch.mean(energy_calculator(prob_vector = noise_test,energies_vector = h2_diag))
        no_noise_energy= torch.mean(energy_calculator(prob_vector = no_noise_test,energies_vector = h2_diag))
        error = np.abs(noise_energy - no_noise_energy)
        
        mean_energy = torch.mean(energy_calculator(prob_vector = denoised_predictions,energies_vector = h2_diag))
        mean_error = np.abs(mean_energy - no_noise_energy)
        noise_reduction_pct = ((error - mean_error) / error) * 100.0
        
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
if __name__ == "__main__":
    main() 
    