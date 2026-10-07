import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import TensorDataset, DataLoader
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper


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
     
     self.decoder = nn.Linear(embed_dim,input_dim)
      
    def forward(self, x):
         
         x_proj = self.input_projection(x)
         x_emb = x_proj.unsqueeze(1)
         
         attn_out, _ =self.attention(x_emb,x_emb,x_emb)
         x_attn = self.norm1(attn_out + x_emb)
         
         ffn_out = self.ffn(x_attn)
         x_out = self.norm2(x_attn + ffn_out)
         
         x_flat = x_out.squeeze(1)
         
         predicted_noise_delta = self.decoder(x_flat)
        
         denoised_logits = x - predicted_noise_delta
         
         return torch.softmax(denoised_logits,dim=-1)
    
    
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
   
     hamiltonian_matrix = qubit_op.to_matrix() + nuclear_repulsion_energy * np.eye(16)
    
     energies = np.real(np.diag(hamiltonian_matrix))
    
     return energies
     
h2_matrix = hamiltonian_energy_matrix_for_h2(bond_distance = 0.735)
h2_diag_energies = torch.tensor(h2_matrix,dtype = torch.float32)

def energy_calculator(prob_vector: torch.Tensor, energies_vector: torch.Tensor = h2_diag_energies) -> torch.Tensor:
    if isinstance(energies_vector, np.ndarray):
        energies_vector = torch.tensor(energies_vector, dtype=torch.float32)

    energies = energies_vector.to(prob_vector.device)
    safe_prob = torch.abs(prob_vector) + 1e-8
    normalized_prob = safe_prob / torch.sum(safe_prob, dim=-1, keepdim=True)
    
    if normalized_prob.dim() == 1:
        return torch.sum(normalized_prob * energies)
    else:
        return torch.sum(normalized_prob * energies, dim=-1)
   
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
   
    model = QuantumAttentionAutoencoder(input_dim=16, embed_dim=32, num_heads=4)
    model.eval()
    print("1. Model Architecture Initialized Successfully.")

    dummy_noisy_input = torch.rand(5, 16)
    dummy_noisy_input /= torch.sum(dummy_noisy_input, dim=-1, keepdim=True) # Normalize

    with torch.no_grad():
        y_pred = model(dummy_noisy_input)

    sums = torch.sum(y_pred, dim=-1)
    assert torch.allclose(sums, torch.ones(5), atol=1e-5)
   
    y_true = torch.zeros(5, 16)
    y_true[:, 3] = 0.99   
    y_true[:, 12] = 0.01  

    pred_energies = energy_calculator(y_pred)
    true_energies = energy_calculator(y_true)
    
if __name__ == "__main__":   
    main()