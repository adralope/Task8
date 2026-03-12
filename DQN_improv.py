import gymnasium as gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque





# DQN Network
class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        Q = self.output(x)
        return Q



class ReplayMemory():

    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)



def train():

    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        domain_randomize=True
    )

    state, _ = env.reset()
    state = state.flatten()

    num_states = state.shape[0]
    num_actions = 5

    # policy and target networks improved
    policy_dqn = DQN(num_states, num_actions)
    target_dqn = DQN(num_states, num_actions)

    target_dqn.load_state_dict(policy_dqn.state_dict())

    optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=5e-4)

    memory = ReplayMemory(50000)

    discount_factor_g = 0.99
    mini_batch_size = 64
    episodes = 200
    max_steps = 1000

    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05

    target_update = 10

    step_count = 0
    network_sync_rate = 1000



    for episode in range(episodes):

        state, _ = env.reset()
        state = state.flatten()

        episode_reward = 0

        for step in range(max_steps):

            # epsilon greedy
            if random.random() < epsilon:
                action = random.randint(0, num_actions - 1)
            else:
                with torch.no_grad():
                    q_vals = policy_dqn(torch.tensor(state, dtype=torch.float32))
                    action = torch.argmax(q_vals).item()

            actions = [
                [0.0, 0.0, 0.0],   # nothing
                [-1.0, 0.0, 0.0],  # left
                [1.0, 0.0, 0.0],   # right
                [0.0, 1.0, 0.0],   # gas
                [0.0, 0.0, 0.8]    # brake
            ]

            env_action = np.array(actions[action])

            new_state, reward, terminated, truncated, _ = env.step(env_action)
            new_state = new_state.flatten()

            memory.append((state, action, reward, new_state, terminated))

            state = new_state
            episode_reward += reward

            step_count += 1

            if step_count > network_sync_rate:
                target_dqn.load_state_dict(policy_dqn.state_dict())
                step_count = 0


            # training step
            if len(memory) > mini_batch_size:

                mini_batch = memory.sample(mini_batch_size)

                states, actions_batch, rewards, new_states, terminations = zip(*mini_batch)

                states = torch.tensor(states, dtype=torch.float32)
                actions_batch = torch.tensor(actions_batch)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                new_states = torch.tensor(new_states, dtype=torch.float32)
                terminations = torch.tensor(terminations, dtype=torch.float32)

                current_q = policy_dqn(states).gather(1, actions_batch.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    next_q = target_dqn(new_states).max(1)[0]
                    target_q = rewards + (1 - terminations) * discount_factor_g * next_q

                loss = F.mse_loss(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            if terminated or truncated:
                break


        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        
        

        print("Episode:", episode, "Reward:", episode_reward)

        


    env.close()
    



if __name__ == "__main__":
    train()
