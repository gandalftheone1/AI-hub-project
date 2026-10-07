import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error
from qiskit import transpile

def get_QuEra_noise_model(config_quera_noise_factor: float = 1.0) -> NoiseModel:
    quera_noise_model = NoiseModel()
    
    p_reset = 0.004 * config_quera_noise_factor   
    error_reset = pauli_error([("X", p_reset), ("I", 1 - p_reset)])    
    quera_noise_model.add_all_qubit_quantum_error(error_reset, "reset")

    p_meas = 0.003 * config_quera_noise_factor
    error_meas = pauli_error([("X", p_meas), ("I", 1 - p_meas)])   
    quera_noise_model.add_all_qubit_quantum_error(error_meas, "measure")

   
    p_cz_active_qub = 0.005 * config_quera_noise_factor  
    cz_single_qubit_error = pauli_error(
        [
            ("X", 1 / 4 * p_cz_active_qub),  
            ("Y", 1 / 4 * p_cz_active_qub),
            ("Z", 1 / 2 * p_cz_active_qub),
            ("I", 1 - p_cz_active_qub),
        ]
    )
    cz_error = cz_single_qubit_error.tensor(cz_single_qubit_error)   
    quera_noise_model.add_all_qubit_quantum_error(    
        cz_error, ["cx", "ecr", "cz"]
    )
    
    p_u1 = 5e-4 * config_quera_noise_factor   
    p_u2 = 1e-3 * config_quera_noise_factor
    p_u3 = 1.5e-3 * config_quera_noise_factor

    sq_error_u1 = pauli_error(
        [
            ("X", 1 / 3 * p_u1),
            ("Y", 1 / 3 * p_u1),
            ("Z", 1 / 3 * p_u1),
            ("I", 1 - p_u1),
        ]
    )
   
    quera_noise_model.add_all_qubit_quantum_error(
        sq_error_u1, ["u1", "rz", "ry", "rx", "sx", "sxdg", "x", "y", "z", "h"]  
    )

    sq_error_u2 = pauli_error(
        [
            ("X", 1 / 3 * p_u2),
            ("Y", 1 / 3 * p_u2),
            ("Z", 1 / 3 * p_u2),
            ("I", 1 - p_u2),
        ]
    )
   
    quera_noise_model.add_all_qubit_quantum_error(sq_error_u2, ["u2"])

    sq_error_u3 = pauli_error(
        [
            ("X", 1 / 3 * p_u3),
            ("Y", 1 / 3 * p_u3),
            ("Z", 1 / 3 * p_u3),
            ("I", 1 - p_u3),
        ]
    )
  
    quera_noise_model.add_all_qubit_quantum_error(sq_error_u3, ["u3", "u"])  

    return quera_noise_model

def from_qbits_to_probability_vector(counts: dict,num_qubits: int, shots: int=1024)-> np.ndarray:
    vector_size = 2**num_qubits
    prob_vector=np.zeros(vector_size,dtype=np.float32)
    
    for bitestring, count in counts.items():
        idx = int(bitestring[::-1],2)
        prob_vector[idx]=count/shots
        
    return prob_vector
def quantum_circuit_generator(samples: int,num_qubits: int,max_depth: int,shots: int=1024):
    noise_circuits=[]
    no_noise_circuits=[]
    basis_gates=['cx', 'id', 'rz', 'sx', 'x', 'cz', 'u1', 'u2', 'u3']
    
    no_noise_sim=AerSimulator()
    for i in range(samples):
        random_depth=np.random.randint(1,max_depth)  
        qc=random_circuit(num_qubits = num_qubits,depth = random_depth,measure = True)
        qc_transpiled=transpile(qc,basis_gates = basis_gates)
        
        noise=np.random.uniform(0.5,3)
        current_noise_model=get_QuEra_noise_model(config_quera_noise_factor=noise)
        noise_sim=AerSimulator(noise_model=current_noise_model)
        noise_result=noise_sim.run(qc_transpiled,shots = shots).result()
        noise_counts=noise_result.get_counts()
        temp_noise_circuit=from_qbits_to_probability_vector(counts = noise_counts,num_qubits = num_qubits ,shots = shots)
        noise_circuits.append(temp_noise_circuit)
        
        no_noise_result=no_noise_sim.run(qc_transpiled,shots = shots).result()
        no_noise_counts=no_noise_result.get_counts()
        temp_no_noise_curcuit=from_qbits_to_probability_vector(counts = no_noise_counts,num_qubits = num_qubits,shots = shots)
        no_noise_circuits.append(temp_no_noise_curcuit)
        
    noise_tensor=torch.tensor(np.array(noise_circuits),dtype = torch.float32)
    no_noise_tensor=torch.tensor(np.array(no_noise_circuits),dtype = torch.float32)  
    
    return noise_tensor,no_noise_tensor    

def rl_quantum_circuit_generator(samples: int = 5000,num_qubits :int = 4,max_depth: int = 100,shots: int = 1024):
    from quantum_env import QuantumCircuitEnv
    from double_dqn import DuelingDQN
    
    env = QuantumCircuitEnv(num_qubits=num_qubits, max_depth=max_depth)
    
    rl_agent = DuelingDQN(state_dim=16, action_dim=env.action_dim)
    try:
        rl_agent.load_state_dict(torch.load("dueling_dqn_quantum.pt"))
        rl_agent.eval()
    except Exception:
        print(" Warning: Could not load dueling_dqn_quantum.pt, using randomly sampled RL trajectories.")

    noisy_circuits = []
    no_noise_circuits = []
    
    state = env.reset()
    
    for i in range(samples):
        with torch.no_grad():
            if np.random.rand() < 0.1: action = np.random.randint(0,env.action_dim)
            else:
                state_tensor = state if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32)
                if state_tensor.dim() == 1: state_tensor = state_tensor.unsqueeze(0)
                q_values = rl_agent(state_tensor)
                action = torch.argmax(q_values, dim=-1).item()
        
        noisy_probs_tensor, _, done = env.step(action)
        
        clean_probs = np.abs(env.state) ** 2
        clean_probs /= np.sum(clean_probs)
        
        noisy_np = noisy_probs_tensor.numpy() if isinstance(noisy_probs_tensor, torch.Tensor) else noisy_probs_tensor
        
        noisy_circuits.append(noisy_np)
        no_noise_circuits.append(clean_probs)
        
        state = env.reset() if done else noisy_probs_tensor

    return (torch.tensor(np.array(noisy_circuits), dtype=torch.float32), 
            torch.tensor(np.array(no_noise_circuits), dtype=torch.float32))
def main():
    eg_samples=5000
    eg_num_qubits=4
    eg_max_depth=100
    eg_batch_size=32
    
    noise_train , no_noise_train = quantum_circuit_generator(samples=eg_samples,num_qubits=eg_num_qubits,max_depth=eg_max_depth,shots=1024)
    
    qc_dataset=TensorDataset(noise_train,no_noise_train)
    train_loader=DataLoader(qc_dataset,batch_size = eg_batch_size,shuffle = True)
    
if __name__ == "__main__":
    main()