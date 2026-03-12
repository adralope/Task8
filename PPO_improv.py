import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F



Total = 200
Steps = 1024
GAMMA = 0.99
EPS_CLIP = 0.2

# better learning rate
LR = 1e-4





class ActorCriticNN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(ActorCriticNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)

        self.actor = nn.Linear(64, out_dim)
        self.critic = nn.Linear(64, 1)

        # standard deviation
        self.log_std = nn.Parameter(torch.zeros(out_dim))


    def forward(self, obs):

        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))

        action = self.actor(activation2)
        value = self.critic(activation2)

        return action, value



def compute_rtgs(batch_rews):

    batch_rtgs = []
    discounted_reward = 0

    for r in reversed(batch_rews):
        discounted_reward = r + GAMMA * discounted_reward
        batch_rtgs.insert(0, discounted_reward)

    return torch.tensor(batch_rtgs, dtype=torch.float32)



def train():

    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=True,
        continuous=True
    )

    obs_dim = np.prod(env.observation_space.shape)
    act_dim = env.action_space.shape[0]

    model = ActorCriticNN(obs_dim, act_dim)

    optimizer = optim.Adam(model.parameters(), lr=LR)


    for episode in range(Total):

        batch_obs = []
        batch_acts = []
        batch_rews = []
        batch_log_probs = []

        obs, _ = env.reset()
        obs = obs.flatten()

        episode_reward = 0

        for t in range(Steps):

            obs_ten = torch.tensor(obs, dtype=torch.float32)

            mean, V = model(obs_ten)

            std = torch.exp(model.log_std)
            dist = Normal(mean, std)

            action = dist.sample()
            log_prob = dist.log_prob(action).sum()

            next_obs, reward, terminated, truncated, _ = env.step(action.detach().numpy())

            done = terminated or truncated

            batch_obs.append(obs)
            batch_acts.append(action.detach().numpy())
            batch_rews.append(reward)
            batch_log_probs.append(log_prob.detach())

            obs = next_obs.flatten()

            episode_reward += reward

            if done:
                break


        batch_rtgs = compute_rtgs(batch_rews)

        batch_obs = torch.tensor(batch_obs, dtype=torch.float32)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float32)
        old_log_probs = torch.stack(batch_log_probs)

        # further advanced calculations
        V, _ = self.evaluate(batch_obs, batch_acts)
        A_K = batch_rtgs - V.detach()
        A_K = (A_K - A_K.mean()) / (A_K.std() + 1e-10)

        for _ in range(4):

            mean, V = model(batch_obs)

            std = torch.exp(model.log_std)
            dist = Normal(mean, std)

            new_log_probs = dist.log_prob(batch_acts).sum(dim=1)

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = (batch_rtgs - V.squeeze()).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print("Episode:", episode, "Reward:", episode_reward)

       


    env.close()



if __name__ == "__main__":
    train()
