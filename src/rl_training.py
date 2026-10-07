import torch
import numpy as np
from double_dqn import DuelingDQN, DQNAgent
from quantum_env import QuantumCircuitEnv

STATE_DIM = 16
ACTION_DIM = 4
EPISODES = 2500  
BATCH_SIZE = 64  
SYNC_TARGET_EVERY = 20

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.997  

bond_distances = [0.3, 0.5, 0.735, 1.0, 1.25, 1.5, 1.75, 2.0]


def train():
    env = QuantumCircuitEnv(num_qubits=4, max_depth=100)
    agent = DQNAgent(state_dim=STATE_DIM,action_dim=ACTION_DIM,lr=1e-3,gamma=0.99,buffer_capacity=20000,
    )
    epsilon = EPSILON_START

    best_reward = -float("inf")

    for episode in range(EPISODES):
        
        dist = np.random.choice(bond_distances)
        state = env.reset(bond_distance=dist)

        if isinstance(state, torch.Tensor): state = state.detach().cpu().numpy().flatten()
        else: state = np.array(state).flatten()

        total_reward = 0.0
        done = False
        episode_loss = 0.0
        step_count = 0

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done = env.step(action)

            if isinstance(next_state, torch.Tensor): next_state = next_state.detach().cpu().numpy().flatten()
            else: next_state = np.array(next_state).flatten()

            agent.memory.push(state, action, reward, next_state, done)

            if len(agent.memory) > BATCH_SIZE:
                loss = agent.update(BATCH_SIZE)
                if loss is not None:
                    episode_loss += (loss.item() if isinstance(loss, torch.Tensor) else loss)

            state = next_state
            total_reward += reward
            step_count += 1

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if episode % SYNC_TARGET_EVERY == 0:
            agent.update_target_network()
            avg_loss = episode_loss / max(1, step_count)
        
        if total_reward > best_reward and episode > 300:
            best_reward = total_reward
            torch.save(agent.policy_net.state_dict(), "dueling_dqn_quantum.pt")

    torch.save(agent.policy_net.state_dict(), "dueling_dqn_quantum.pt")
 
if __name__ == "__main__":
    train()