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
EPS_START = 0.95  # Start with more exploration
EPS_END = 0.02   # Keep some exploration
EPS_DECAY = 5000  # Slower decay for better spike learning
TAU = 0.005
LR = 1e-4  # Lower learning rate for stability

# endregion

# region Sieć neuronowa DQN
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # Larger network for better spike pattern recognition
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, n_actions)
        
        # Dropout for better generalization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        x = F.relu(self.layer3(x))
        return self.layer4(x)

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
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))


    spike_weight = 1.5
    weighted_loss = spike_weight * loss

    # Optimize the model
    optimizer.zero_grad()
    weighted_loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

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
    
    # Check if player died (hit spikes)
    if game_state.get('player_dead', False):
        return -100.0  # Heavy penalty for dying
    
    # Base survival reward
    reward += 1.0
    
    # Reward for score increase (successful wall bounces)
    score = game_state.get('score', 0)
    if hasattr(calculate_reward, 'last_score'):
        if score > calculate_reward.last_score:
            reward += 20.0  # Reward for progress
    calculate_reward.last_score = score
    
    # Reward for coin collection
    coins = game_state.get('collected_coins', 0)
    if hasattr(calculate_reward, 'last_coins'):
        if coins > calculate_reward.last_coins:
            reward += 50.0  # Big reward for coins
    calculate_reward.last_coins = coins
    
    # Penalty for being close to spikes using original engine data
    player_y = game_state.get('player_pos_y', 0)
    spikes = game_state.get('spikes_pos_y', [])
    
    min_distance = float('inf')
    for spike_y in spikes:
        if isinstance(spike_y, (int, float)) and spike_y >= 0:
            distance = abs(player_y - spike_y)
            min_distance = min(min_distance, distance)
    
    # Apply distance-based penalty
    if min_distance < 30:
        reward -= 5.0  # Close to spike
    elif min_distance < 50:
        reward -= 2.0  # Moderately close
    elif min_distance < 80:
        reward -= 0.5  # Slightly close
    
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

        # Extract key information from observation
        player_x, player_y = obs[0], obs[1]
        player_velocity_dir = obs[2]  # 1 if moving right, -1 if left
        player_gravity = obs[3]
        coin_x, coin_y = obs[4], obs[5]
        spike_positions = obs[6:15]  # Last 9 values are spike positions
        
        # Enhanced observation processing with intelligent spike emphasis
        obs_enhanced = self._create_enhanced_observation(obs, player_y, spike_positions)
        
        if sample > eps_threshold:
            # Use neural network for action selection
            with torch.no_grad():
                state_tensor = torch.tensor(obs_enhanced, dtype=torch.float32, device=device).unsqueeze(0)
                action = policy_net(state_tensor).max(1)[1].item()
                return action
        else:
            # Intelligent exploration - avoid obvious spike dangers
            return self._intelligent_exploration(player_y, spike_positions, player_gravity)
    
    def _create_enhanced_observation(self, obs, player_y, spike_positions):
        """Create enhanced observation with intelligent spike danger analysis"""
        obs_enhanced = obs.copy()
        
        # Analyze each spike position and apply dynamic scaling
        for i, spike_y in enumerate(spike_positions):
            if spike_y >= 0:  # Valid spike position
                distance = abs(player_y - spike_y)
                
                if distance < 25:  # Immediate danger zone
                    obs_enhanced[6 + i] = spike_y * 15.0  # Maximum emphasis
                elif distance < 50:  # High danger zone
                    obs_enhanced[6 + i] = spike_y * 10.0  # High emphasis
                elif distance < 80:  # Moderate danger zone
                    obs_enhanced[6 + i] = spike_y * 6.0   # Medium emphasis
                elif distance < 120:  # Awareness zone
                    obs_enhanced[6 + i] = spike_y * 3.0   # Low emphasis
                else:  # Distant spikes
                    obs_enhanced[6 + i] = spike_y * 1.5   # Minimal emphasis
            # Keep -1 values unchanged for empty spike slots
        
        return obs_enhanced
    
    def _intelligent_exploration(self, player_y, spike_positions, gravity):
        """Smart exploration that actively avoids spike dangers"""
        # Calculate danger scores for different actions
        current_danger = self._calculate_danger_score(player_y, spike_positions)
        
        # Estimate position after jump (player typically moves up 15-25 pixels)
        jump_y_estimate = player_y - 20
        jump_danger = self._calculate_danger_score(jump_y_estimate, spike_positions)
        
        # Estimate position after falling (gravity effect)
        fall_y_estimate = player_y + abs(gravity) * 2
        fall_danger = self._calculate_danger_score(fall_y_estimate, spike_positions)
        
        # Decision making based on danger levels
        if current_danger > 10:  # Critical danger
            # Emergency mode - choose safest option immediately
            return 1 if jump_danger < fall_danger else 0
        elif current_danger > 5:  # High danger
            # Bias heavily towards safer action
            safer_action = 1 if jump_danger < fall_danger else 0
            return safer_action if random.random() < 0.85 else (1 - safer_action)
        elif current_danger > 2:  # Moderate danger
            # Bias towards safer action but allow some variation
            safer_action = 1 if jump_danger < fall_danger else 0
            return safer_action if random.random() < 0.7 else (1 - safer_action)
        else:  # Low danger
            # More random exploration when relatively safe
            return random.choice([0, 1])
    
    def _calculate_danger_score(self, y_position, spike_positions):
        """Calculate comprehensive danger score for a position"""
        danger_score = 0
        
        for spike_y in spike_positions:
            if spike_y >= 0:  # Valid spike
                distance = abs(y_position - spike_y)
                
                # Progressive danger scoring based on distance
                if distance < 15:
                    danger_score += 25  # Extreme danger
                elif distance < 30:
                    danger_score += 15  # Very high danger
                elif distance < 50:
                    danger_score += 8   # High danger
                elif distance < 80:
                    danger_score += 4   # Moderate danger
                elif distance < 120:
                    danger_score += 2   # Low danger
                elif distance < 160:
                    danger_score += 1   # Minimal danger
        
        return danger_score

# endregion
# region Funkcje wymagane

def create_bot() -> BaseBot:
    """
    Funkcja tworząca i zwracająca instancję bota.
    
    Returns:
        BaseBot: instancja klasy bota dziedziczącej po BaseBot
    """
    return MojBot()


def create_bot() -> BaseBot:
    return MojBot()

# region Trening bota

def train_bot():
    if torch.cuda.is_available():
        num_episodes = 1500  # More episodes for complex spike patterns
      
    else:
        num_episodes = 100
    
    # Reset reward function tracking
    calculate_reward.last_score = 0
    calculate_reward.last_coins = 0

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
            optimize_model()

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

        if (i_episode + 1) % 50 == 0:
            print(f"Episode {i_episode+1}/{num_episodes}")