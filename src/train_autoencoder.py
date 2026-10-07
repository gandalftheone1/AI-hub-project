import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model_and_vqe import QuantumAttentionAutoencoder, energy_calculator
from quantum_env import QuantumCircuitEnv
from rl_training import DQNAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_qpu_noise(clean_state: np.ndarray, noise_level: float = 0.15) -> np.ndarray:

    clean_state = np.clip(clean_state, 0.0, None)
    clean_state = clean_state / np.sum(clean_state)

    noise = np.random.normal(0, noise_level, size=clean_state.shape)
    noisy_state = np.abs(clean_state + noise)

    return noisy_state / np.sum(noisy_state)


def generate_dataset(rl_agent: DQNAgent, num_samples: int = 10000):
    bond_distances = [0.3, 0.5, 0.735, 1.0, 1.25, 1.5, 1.75, 2.0]
    clean_states = []
    noisy_states = []

    for i in range(num_samples):

        dist = np.random.choice(bond_distances)
        env = QuantumCircuitEnv(max_depth=100)
        state_tensor = env.reset(bond_distance=dist)

        state = state_tensor.detach().cpu().numpy().flatten()

        for step in range(20):
            action = rl_agent.select_action(state, epsilon=0.0)
            next_state_tensor, _, done = env.step(action)
            state = next_state_tensor.detach().cpu().numpy().flatten()
            if done: break

        state = np.clip(state, 0.0, None)
        state = state / np.sum(state)

        noisy_state = apply_qpu_noise(state, noise_level=0.15)

        clean_states.append(state)
        noisy_states.append(noisy_state)

    return (torch.tensor(np.array(noisy_states), dtype=torch.float32),torch.tensor(np.array(clean_states), dtype=torch.float32),)

def physics_informed_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    mse = nn.MSELoss()(y_pred, y_true)

    tvd = 0.5 * torch.mean(torch.sum(torch.abs(y_pred - y_true), dim=-1))

    norm_penalty = torch.mean((torch.sum(y_pred, dim=-1) - 1.0) ** 2)

    pred_energy = energy_calculator(y_pred)
    true_energy = energy_calculator(y_true)
    energy_loss = torch.mean(torch.abs(pred_energy - true_energy))

    return mse + 0.1 * tvd + 0.5 * norm_penalty + 2.0 * energy_loss

def train_encoder():
    rl_agent = DQNAgent(state_dim=16, action_dim=4)
    try:
        rl_agent.policy_net.load_state_dict(torch.load("dueling_dqn_quantum.pt", map_location=device))
        rl_agent.policy_net.eval()
    except Exception as e:
        print(f" Warning: Could not load 'dueling_dqn_quantum.pt' ({e}). Using un-trained policy.")

    X_noisy, Y_clean = generate_dataset(rl_agent, num_samples=10000)
    dataset = TensorDataset(X_noisy, Y_clean)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = QuantumAttentionAutoencoder(input_dim=16, embed_dim=32, num_heads=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()
    for epoch in range(1, 101):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            out = model(batch_x)
            loss = physics_informed_loss(out, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_loss / len(loader)
            print(f"   Epoch {epoch:3d}/100 | Combined Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), "quantum_transformer_autoencoder.pt")

if __name__ == "__main__":
    train_encoder()