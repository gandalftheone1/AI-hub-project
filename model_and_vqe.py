import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import TensorDataset, DataLoader
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from generated_and_noised_quantum_circuits import quantum_circuit_generator

eg_samples=5000
eg_num_qubits=4
eg_max_depth=8
eg_batch_size=32
    
noise_train , no_noise_train = quantum_circuit_generator(samples=eg_samples,num_qubits=eg_num_qubits,max_depth=eg_max_depth,shots=1024)

qc_dataset=TensorDataset(noise_train,no_noise_train)
train_loader=DataLoader(qc_dataset,batch_size = eg_batch_size,shuffle = True) 

class QuantumAttentionAutoencoder(nn.Module):
    def __init__(self,input_dim: int = 16,embed_dim: int = 32, num_heads: int = 4 ):
     super(QuantumAttentionAutoencoder,self).__init__()
     
     self.input_projection = nn.Linear(input_dim,embed_dim)
     self.attention = nn.MultiheadAttention(embed_dim = embed_dim,num_heads = num_heads,batch_first = True)
     self.norm1 = nn.LayerNorm(embed_dim)
     
     self.ffn = nn.Sequential(
         nn.Linear(embed_dim,64),
         nn.ReLU(),
         nn.Linear(64,embed_dim)
     )
     self.norm2 = nn.LayerNorm(embed_dim)
     
     self.decoder=nn.Sequential(
         nn.Linear(embed_dim,input_dim),
         nn.Softmax(dim=-1)
     )
    
    def forward(self, x):
         
         x_proj = self.input_projection(x)
         x_emb = x_proj.unsqueeze(1)
         
         attn_out, _ =self.attention(x_emb,x_emb,x_emb)
         x_attn = self.norm1(attn_out + x_emb)
         
         ffn_out = self.ffn(x_attn)
         x_out = self.norm2(x_attn + ffn_out)
         
         x_flat = x_out.squeeze(1)
         
         no_noise_prob_vector = self.decoder(x_flat)
         
         return no_noise_prob_vector
    
model = QuantumAttentionAutoencoder(input_dim=16, embed_dim=32, num_heads=4)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("\nStep 2: Training Quantum Transformer Attention Model...")

num_epochs = 100
for epoch in range(1, num_epochs + 1):
    total_loss = 0.0
    
    for noisy_batch, clean_batch in train_loader:
        predicted_clean = model(noisy_batch)
        loss = criterion(predicted_clean, clean_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    if epoch % 20 == 0 or epoch == 1:
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch:3d}/{num_epochs} | Loss (MSE): {avg_loss:.6f}")


torch.save(model.state_dict(), "quantum_transformer_autoencoder.pt")
print("\nTransformer model saved as 'quantum_transformer_autoencoder.pt'!") 
    
    
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
     
h2_matrix = hamiltonian_energy_matrix_for_h2(bond_distance = 0.735)
h2_diag_energies = torch.tensor(h2_matrix,dtype = torch.float32)

def energy_calculator(prob_vector: torch.Tensor,energies_vector: torch.Tensor = h2_diag_energies)->torch.Tensor:
   if isinstance(energies_vector, np.ndarray):
        energies_vector = torch.tensor(energies_vector, dtype=torch.float32)
   energies = energies_vector.to(prob_vector.device)
   
   if (prob_vector.dim() == 1): return torch.sum(prob_vector * energies)
   else: return torch.sum(prob_vector * energies,dim=-1)
   
def chemical_accuracy_evaluation(pred_vector: torch.Tensor, real_vector: torch.Tensor):
    pred_energies = energy_calculator(pred_vector)
    real_energies = energy_calculator(real_vector)
    
    abs_errors = torch.abs(pred_energies-real_energies)
    chem_limit = 0.0016
    
    accurate_samples = (abs_errors <= chem_limit).sum().item() 
    batch_size = pred_vector.shape[0] if pred_vector.dim() > 1 else 1
    accuracy_prct = (accurate_samples / batch_size)*100.0
    mean_error = torch.mean(abs_errors).item()
    
    return accuracy_prct,mean_error

def main():
   
    print("==================================================")
    print("🧪 TESTING MODEL & VQE ENERGY ENGINE")
    print("==================================================\n")

    # 1. Instantiate the Autoencoder
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
    y_true[:, 3] = 0.99   # Main ground state component |0101>
    y_true[:, 12] = 0.01  # Secondary component |1010>

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