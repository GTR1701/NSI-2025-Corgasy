from base_bot import BaseBot
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
# Wymaga GameEnv
from src.env import GameEnv 

class PolicyNet(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, output_size=2):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, output_size), nn.Softmax(dim=-1))
    def forward(self, x):
        return self.net(x)

class ReinforceTorchBot(BaseBot):
    def __init__(self):
        self.policy_net = PolicyNet()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.01)
        self.log_probs = []
        self.rewards = []
        self.gamma = 0.99

    def take_action(self, obs: np.ndarray) -> int:
        x_np = np.zeros(15, dtype=np.float32); x_np[:len(obs)] = obs
        state_tensor = torch.from_numpy(x_np).float()
        
        probs = self.policy_net(state_tensor)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        
        self.log_probs.append(action_dist.log_prob(action).unsqueeze(0))
        return action.item()

    def train(self):
        """Aktualizacja wag sieci na podstawie zebranych nagród (REINFORCE)."""
        R = 0; returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns); 
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = [-log_prob * R for log_prob, R in zip(self.log_probs, returns)]
            
        self.optimizer.zero_grad()
        loss = torch.cat(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        self.log_probs = []; self.rewards = []
        return loss.item()

def create_bot() -> BaseBot:
    return ReinforceTorchBot()

def calculate_reward(game_state: Dict[str, Any]) -> float:
    return -10.0 if game_state["player_dead"] else 1

def train_bot(episodes: int = 500):
    """Przykładowa pętla treningowa REINFORCE."""
    bot = create_bot()
    env = GameEnv(calculate_reward=calculate_reward, render_mode="headless") 

    for i in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = bot.take_action(obs)
            obs, reward, done, _, info = env.step(action)
            
            bot.rewards.append(reward)
            episode_reward += reward

        loss = bot.train()
        if (i + 1) % 50 == 0:
             print(f"Episode {i+1}/{episodes}, Reward: {episode_reward:.2f}, Loss: {loss:.4f}")
    env.close()