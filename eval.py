import torch
import numpy as np
from double_dqn import DQNAgent
from quantum_env import QuantumCircuitEnv

env =  QuantumCircuitEnv(num_qubits = 4,max_depth = 8)
agent = DQNAgent(satet_dim = 16,action_dim = 4)

agent.policy_net.load_satet_dict(torch.load("dueling_dqn_quantum.pt"))
agent.policy_net.eval()

test_rewards = []
for i in range(10):
    state  = env.reset()
    if (isinstance(state,torch.Tensor)):
        state = state.numpy().flatten()
    
    done = False
    i_reward = 0.0
    
    while not done:
        action = agent.select_action(state,epsilon = 0.0)
        next_state,reward,done = env.state(action)
        if (isinstance(next_state,torch.Tensor)):
                next_state = next_state.numpy().flatten()
        
        state = next_state
        i_reward += reward
    
    test_rewards.append(i_reward)
    
print(f"📊 Average Evaluation Reward (Greedy): {np.mean(test_rewards):.4f}")