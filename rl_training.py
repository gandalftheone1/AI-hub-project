import torch
import numpy as np
from double_dqn import DuelingDQN,DQNAgent
from quantum_env import QuantumCircuitEnv

STATE_DIM = 16
ACTION_DIM = 4
EPISODES = 200
BATCH_SIZE = 32
SYNC_TARGET_EVERY = 10

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.98

env = QuantumCircuitEnv(num_qubits = 4)
agent = DQNAgent(
    state_dim = STATE_DIM,
    action_dim = ACTION_DIM,
    lr = 1e-3,
    gamma = 0.99,
    buffer_capacity = 10000
)
epsilon = EPSILON_START

print("🚀 STARTING DUELING DOUBLE-DQN REINFORCEMENT LEARNING...")
print("=======================================================")

for episodes in range(0,EPISODES):
    state = env.reset()
    if (isinstance(state,torch.Tensor)):
        state = state.numpy().flatten()
    else: np.array(state).flatten()
    total_reward = 0.0
    done = False
    
    while not done:
        action = agent.select_action(state,epsilon)
        next_state,reward,done = env.step(action)
        
        if (isinstance(state,torch.Tensor)):
            next_state = next_state.numpy().flatten()
        else: np.array(next_state).flatten() 
            
        agent.memory.push(state,action,reward,next_state,done)
        loss = agent.update(BATCH_SIZE)
        state = next_state
        total_reward += reward
    
    epislon = max(EPSILON_END , epsilon * EPSILON_DECAY)
    
    if episodes % SYNC_TARGET_EVERY == 0:
        agent.update_target_network()
        print(f"Episode {episodes:3d}/{EPISODES} | Total Reward: {total_reward:.4f} | Epsilon: {epsilon:.3f} | Last Loss: {loss:.5f}")


torch.save(agent.policy_net.state_dict(), "dueling_dqn_quantum.pt")
print("\n✅ RL Training Complete! Saved weights to 'dueling_dqn_quantum.pt'.")
    
