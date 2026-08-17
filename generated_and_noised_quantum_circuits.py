import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from qiskit.circuit.random import random_circuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error
from qiskit import transpile

def get_QuEra_noise_model(config_quera_noise_factor: float = 1.0) -> NoiseModel:
    quera_noise_model = NoiseModel()
    
    p_reset = 0.004 * config_quera_noise_factor   #0.4% chance of not getting back to the |0>-initial condition
    error_reset = pauli_error([("X", p_reset), ("I", 1 - p_reset)])    #with p_reset we have a chance of the X-gate application and with 1-p_reset the I-gate
    quera_noise_model.add_all_qubit_quantum_error(error_reset, "reset")

    p_meas = 0.003 * config_quera_noise_factor
    error_meas = pauli_error([("X", p_meas), ("I", 1 - p_meas)])   #0.3% chance of reading the condition wrong
    quera_noise_model.add_all_qubit_quantum_error(error_meas, "measure")

   
    p_cz_active_qub = 0.005 * config_quera_noise_factor   #0.5% chance of appplying a 2-qbit gate in an entagled situation
    cz_single_qubit_error = pauli_error(
        [
            ("X", 1 / 4 * p_cz_active_qub),   #different chances of applications among gates
            ("Y", 1 / 4 * p_cz_active_qub),
            ("Z", 1 / 2 * p_cz_active_qub),
            ("I", 1 - p_cz_active_qub),
        ]
    )
    cz_error = cz_single_qubit_error.tensor(cz_single_qubit_error)   #tensor product-"tensored error"
    quera_noise_model.add_all_qubit_quantum_error(    
        cz_error, ["cx", "ecr", "cz"]
    )
    
    p_u1 = 5e-4 * config_quera_noise_factor   #different errors,based on the complexity of the gates
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
    
    print(f" Έναρξη παραγωγής {samples} δειγμάτων για {num_qubits} qubits:")
    
    no_noise_sim=AerSimulator()
    for i in range(samples):
        random_depth=np.random.randint(1,max_depth)  #depth=number of concecutive quantum gates-not in parallel
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
        
        if(i+1) % 200==0:
            print(f" --> Ολοκληρώθηκαν  {i+1}/{samples} δείγματα ")
        
    noise_tensor=torch.tensor(np.array(noise_circuits),dtype = torch.float32)
    no_noise_tensor=torch.tensor(np.array(no_noise_circuits),dtype = torch.float32)  
    
    return noise_tensor,no_noise_tensor    

def main():
    eg_samples=5000
    eg_num_qubits=4
    eg_max_depth=8
    eg_batch_size=32
    
    noise_train , no_noise_train = quantum_circuit_generator(samples=eg_samples,num_qubits=eg_num_qubits,max_depth=eg_max_depth,shots=1024)
    
    print(f"\n Το Dataset δημιουργήθηκε επιτυχώς!")
    print(f"X_train Shape (Noisy Input):  {noise_train.shape}")  
    print(f"Y_train Shape (Clean Target): {no_noise_train.shape}")  
    
    print(f"Clean Tensor Sum: {torch.sum(no_noise_train[0]).item():.4f}") # Τυπώνει 1.0000
    print(f"Noisy Tensor Sum: {torch.sum(noise_train[0]).item():.4f}")    # Τυπώνει 1.0000 
    
    qc_dataset=TensorDataset(noise_train,no_noise_train)
    train_loader=DataLoader(qc_dataset,batch_size = eg_batch_size,shuffle = True)
    
    for noise_qc_batch, no_noise_qc_batch in train_loader:
        print(f"\nBatch Input Shape:  {noise_qc_batch.shape}")
        print(f"Batch Target Shape: {no_noise_qc_batch.shape}")
        break

if __name__ == "__main__":
    main()