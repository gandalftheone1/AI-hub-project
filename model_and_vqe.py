import numpy as np
import torch 
import torch.nn as nn
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper


class QuantumDenoisingAutoencoder(nn.Module):
    def __init__(self,dim :int = 16):
     super(QuantumDenoisingAutoencoder,self).__init__()
     
     self.encoder=nn.Sequential(
         nn.Linear(dim,64),
         nn.BatchNorm1d(64),
         nn.ReLU(),
         nn.Linear(64,32),
         nn.ReLU(),
         nn.Linear(32,16),
         nn.ReLU()
     )

     self.decoder=nn.Sequential(
         nn.Linear(16,32),
         nn.ReLU(),
         nn.Linear(32,64),
         nn.ReLU(),
         nn.Linear(64,dim),
         nn.Softmax(dim=-1)
          )
    
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    
def hamiltonian_energy_matrix_for_h2(bond_distance:float = 0.735)-> np.ndarray:
    
     driver = PySCFDriver(
        atom = f"H 0 0 0; H 0 0 {bond_distance}",
        basis = "sto3g"
    )
    
     problem = driver.run()
     second_q_op = problem.hamiltonian.second_q_op()
     nuclear_repulsion_energy = problem.nuclear_repulsion_energy
    
     mapper = JordanWignerMapper()
     qubit_op = mapper.map(second_q_op)
   
     hamiltonian_matrix = qubit_op.to_matrix()
   
     full_matrix = hamiltonian_matrix + nuclear_repulsion_energy * np.eye(16)
    
     diagonal_energies = np.real(np.diag(full_matrix))
    
     return diagonal_energies

def energy_calculator(prob_vector: torch.Tensor)->torch.Tensor:
    
   energies = torch.tensor(hamiltonian_energy_matrix_for_h2(bond_distance = 0.735),dtype = torch.float32).to(prob_vector.device)
   if (prob_vector.dim == 1): return torch.sum(prob_vector * energies)
   else: return torch.sum(prob_vector * energies,dim=-1)
   
def chemical_accuracy_evaluation(pred_vector: torch.Tensor, real_vector: torch.Tensor):
    
    pred_energies = energy_calculator(pred_vector)
    real_energies = energy_calculator(real_vector)
    
    abs_errors = np.abs(pred_energies-real_energies)
    chem_limit = 0.0016
    
    accurate_samples = (abs_errors <= chem_limit).sum().item()
    accuracy_prct = (accurate_samples / pred_vector.shape[0])*100
    mean_error = torch.mean(abs_errors).item()
    
    return accuracy_prct,mean_error

def main():
   
    print("==================================================")
    print("🧪 TESTING MODEL & VQE ENERGY ENGINE")
    print("==================================================\n")

    # 1. Instantiate the Autoencoder
    model = QuantumDenoisingAutoencoder(dim=16)
    model.eval()
    print("1. Model Architecture Initialized Successfully.")

    # 2. Generate a dummy batch of noisy inputs (Batch Size = 5, Qubit Dimension = 16)
    dummy_noisy_input = torch.rand(5, 16)
    dummy_noisy_input /= torch.sum(dummy_noisy_input, dim=-1, keepdim=True) # Normalize

    # 3. Test Forward Pass (y_pred)
    with torch.no_grad():
        y_pred = model(dummy_noisy_input)

    print(f"2. Forward Pass Output Shape: {y_pred.shape} (Expected: [5, 16])")

    # 4. Verify Softmax Constraint (sum of probabilities must equal 1.0)
    sums = torch.sum(y_pred, dim=-1)
    print(f"3. Probability Vector Sums: {sums.tolist()}")
    assert torch.allclose(sums, torch.ones(5), atol=1e-5), "❌ Error: Softmax constraint failed!"
    print("   ✅ Softmax verified: All probability vectors sum to 1.0!\n")

    # 5. Test VQE Energy Calculation
    # Create a dummy target y_true (representing ideal ground state)
    y_true = torch.zeros(5, 16)
    y_true[:, 5] = 0.85   # Main ground state component |0101>
    y_true[:, 10] = 0.15  # Secondary component |1010>

    pred_energies = energy_calculator(y_pred)
    true_energies = energy_calculator(y_true)

    print(f"4. Predicted Energies (Ha): {pred_energies.tolist()}")
    print(f"   True Target Energies (Ha): {true_energies.tolist()}\n")

    # 6. Test Chemical Accuracy Evaluation
    mean_err, acc_pct = chemical_accuracy_evaluation(y_pred, y_true)
    print("5. Chemical Accuracy Evaluation Test:")
    print(f"   - Mean Energy Error: {mean_err:.6f} Hartree")
    print(f"   - Samples meeting Chemical Accuracy (<=0.0016 Ha): {acc_pct:.1f}%\n")

    print("==================================================")
    print("🎉 ALL UNIT TESTS PASSED SUCCESSFULLY!")
    print("==================================================")
    
if __name__ == "__main__":
     main()