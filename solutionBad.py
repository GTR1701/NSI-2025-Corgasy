from networkx import display
import numpy as np
from base_bot import BaseBot
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.env import GameEnv

# region Ustawienia i stałe
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Opcjonalne stałe konfiguracyjne
BENCHMARK_EPISODES = 50  # Liczba epizodów w benchmarku
WATCH_GAME = True        # Czy wyświetlić wizualizację po benchmarku

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9   # Moderate exploration
EPS_END = 0.05    # Some final exploration
EPS_DECAY = 10  # Faster initial decay to reduce early failures
TAU = 0.005
LR = 0.01

# endregion

# region Sieć neuronowa DQN
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # Simpler network for basic learning
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.HuberLoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss

def calculate_reward(game_state: dict) -> float:
    """
    Funkcja obliczająca nagrodę na podstawie stanu gry.
    Wywoływana na każdym kroku gry oraz po zakończeniu gry.
    
    Args:
        game_state: słownik zawierający informacje o stanie gry
        
    Returns:
        float: wartość nagrody
    """
    reward = 0.0
    
    # Check if player died - small penalty
    if game_state.get('player_dead', False):
        reward -= 100.0  # Very small death penalty
    else:
        # LARGE survival bonus - make staying alive very attractive
        reward += 10.0  # Large survival reward
    
    # MASSIVE rewards for any progress - this is the key learning signal
    score = game_state.get('score', 0)
    if hasattr(calculate_reward, 'last_score'):
        if score > calculate_reward.last_score:
            reward += 100.0  # Huge reward for any wall bounce
    calculate_reward.last_score = score
    
    # Large reward for coin collection
    coins = game_state.get('collected_coins', 0)
    if hasattr(calculate_reward, 'last_coins'):
        if coins > calculate_reward.last_coins:
            reward += 35.0  # Large coin reward
    calculate_reward.last_coins = coins
    
    # Reward just for trying different actions (encourage exploration)
    # This helps break action 0 bias
    player_y = game_state.get('player_pos_y', 0)
    if hasattr(calculate_reward, 'last_y'):
        y_change = abs(player_y - calculate_reward.last_y)
        if y_change > 5:  # Player moved vertically (likely jumped)
            reward += 5.0  # Small reward for movement variety
    calculate_reward.last_y = player_y
    
    return reward


env = GameEnv(calculate_reward=calculate_reward)

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


episode_durations = []

# endregion
# region Bot

class MojBot(BaseBot):
    """
    Klasa bota uczestnika.
    Musi dziedziczyć po BaseBot.
    """
    
    def __init__(self):
        # Inicjalizacja bota
        self.action_count = {0: 0, 1: 0}
        pass
    
    def take_action(self, obs) -> int:
        """
        Funkcja podejmująca decyzję o akcji na podstawie obserwacji.
        
        Args:
            obs: numpy array o rozmiarze (15,) z danymi o stanie gry lub tensor
            
        Returns:
            int: akcja do wykonania (0 lub 1)
        """
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1

        if sample > eps_threshold:
            # Use neural network for action selection
            with torch.no_grad():
                state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                action_probs = policy_net(state_tensor)
                action = action_probs.max(1)[1].item()
                return action
        else:
            # FORCED BALANCED EXPLORATION - ensure both actions get tried equally
            if not hasattr(self, 'action_count'):
                self.action_count = {0: 0, 1: 0}
            
            # Force balance: if one action is tried much more, favor the other
            total_actions = sum(self.action_count.values())
            if total_actions > 100:  # After some initial exploration
                action_0_ratio = self.action_count[0] / total_actions
                if action_0_ratio > 0.7:  # Too much action 0
                    action = 1  # Force jump
                elif action_0_ratio < 0.3:  # Too much action 1  
                    action = 0  # Force no jump
                else:
                    action = random.choice([0, 1])  # Balanced random
            else:
                action = random.choice([0, 1])  # Pure random initially
            
            self.action_count[action] += 1
            return action
    

    


# endregion
# region Funkcje wymagane

def create_bot() -> BaseBot:
    """
    Funkcja tworząca i zwracająca instancję bota.
    
    Returns:
        BaseBot: instancja klasy bota dziedziczącej po BaseBot
    """
    return MojBot()

# region Trening bota

def train_bot():
    if torch.cuda.is_available():
        num_episodes = 1000  # Shorter training to focus on action balance
      
    else:
        num_episodes = 50
    
    # Reset reward function tracking
    calculate_reward.last_score = 0
    calculate_reward.last_coins = 0
    calculate_reward.last_y = 0

    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, info = env.reset()
        
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        bot = MojBot()
        for t in count():
            # Convert tensor state to numpy for take_action
            state_np = state.cpu().numpy().squeeze()
            action = bot.take_action(state_np)

            observation, reward, terminated, truncated, _ = env.step(action)
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated



            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Convert action to tensor before storing in memory
            action_tensor = torch.tensor([[action]], device=device, dtype=torch.long)
            
            # Store the transition in memory
            memory.push(state, action_tensor, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            loss = optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                # plot_durations()
                break

        if (i_episode + 1) % 10 == 0:
            avg_duration = sum(episode_durations[-50:]) / min(50, len(episode_durations))
            # Report action balance
            total_actions = sum(bot.action_count.values())
            action_1_ratio = bot.action_count[1] / max(1, total_actions)
            print(f"Episode {i_episode+1}/{num_episodes}, Avg Duration: {avg_duration:.1f}, Jump Rate: {action_1_ratio:.2f}, Loss: {loss.item() if loss else 'N/A'}")