from base_bot import BaseBot
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pickle
import os

BENCHMARK_EPISODES = 50
WATCH_GAME = True

class ParameterOptimizer(nn.Module):
    """Neural network to optimize heuristic parameters"""
    def __init__(self, input_size=10, hidden_size=128):
        super(ParameterOptimizer, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 20)  # 20 parameters to optimize
        )
        
        # Initialize with current good parameters
        self.base_params = torch.tensor([
            # Original 8 parameters
            0.1,   # DISTANCE_FROM_GAP_WEIGHT
            5.0,   # GAP_SIZE_WEIGHT
            1.5,   # MIN_VIABLE_GAP_WEIGHT
            0.7,   # TOP_SPIKE_MARGIN_WEIGHT
            0.7,   # BOTTOM_SPIKE_MARGIN_WEIGHT
            0.2,   # COIN_GREED_WEIGHT
            25.0,  # MIN_SAFE_DEVIATION_WEIGHT
            100.0, # MAX_SAFE_DEVIATION_WEIGHT
            # New action control parameters
            80.0,  # CEILING_AVOIDANCE_THRESHOLD
            60.0,  # FLOOR_AVOIDANCE_THRESHOLD
            10.0,  # TARGET_POSITION_TOLERANCE
            -4.0,  # VELOCITY_BRAKE_THRESHOLD
            # Dynamic calculation parameters
            20.0,  # BASE_UNCERTAINTY_MARGIN
            30.0,  # SPIKE_DENSITY_UNCERTAINTY_FACTOR
            15.0,  # OVERLAP_PENALTY_PER_PAIR
            100.0, # EDGE_EXCLUSION_ZONE
            30.0,  # LARGE_GAP_BONUS
            45.0,  # MAX_OVERLAP_PENALTY
            # Emergency and fallback parameters
            352.0, # DEFAULT_TARGET_POSITION
            2.0    # EMERGENCY_GAP_MULTIPLIER
        ], dtype=torch.float32)
        
    def forward(self, context):
        """
        Context includes: spike density, coin presence, current position relative to screen
        """
        adjustments = self.network(context)
        # Apply constraints to keep parameters in reasonable ranges
        optimized_params = self.base_params + adjustments * torch.tensor([
            # Original parameters
            0.1,   # DISTANCE_FROM_GAP_WEIGHT: 0.01-0.3
            2.0,   # GAP_SIZE_WEIGHT: 2.0-8.0
            0.5,   # MIN_VIABLE_GAP_WEIGHT: 1.0-2.5
            0.3,   # TOP_SPIKE_MARGIN_WEIGHT: 0.3-1.2
            0.3,   # BOTTOM_SPIKE_MARGIN_WEIGHT: 0.3-1.2
            0.3,   # COIN_GREED_WEIGHT: 0.05-0.8
            15.0,  # MIN_SAFE_DEVIATION_WEIGHT: 10-50
            50.0,  # MAX_SAFE_DEVIATION_WEIGHT: 50-200
            # Action control parameters
            30.0,  # CEILING_AVOIDANCE_THRESHOLD: 50-110
            30.0,  # FLOOR_AVOIDANCE_THRESHOLD: 30-90
            10.0,  # TARGET_POSITION_TOLERANCE: 2-20
            2.0,   # VELOCITY_BRAKE_THRESHOLD: -6 to -2
            # Dynamic calculation parameters
            15.0,  # BASE_UNCERTAINTY_MARGIN: 5-35
            20.0,  # SPIKE_DENSITY_UNCERTAINTY_FACTOR: 10-50
            10.0,  # OVERLAP_PENALTY_PER_PAIR: 5-25
            50.0,  # EDGE_EXCLUSION_ZONE: 50-150
            20.0,  # LARGE_GAP_BONUS: 10-50
            25.0,  # MAX_OVERLAP_PENALTY: 20-70
            # Emergency parameters
            100.0, # DEFAULT_TARGET_POSITION: 250-450
            1.0    # EMERGENCY_GAP_MULTIPLIER: 1.0-3.0
        ])
        
        # Clamp to ensure positive values and reasonable ranges
        optimized_params = torch.clamp(optimized_params, min=torch.tensor([
            0.01, 1.0, 0.5, 0.1, 0.1, 0.05, 5.0, 30.0,  # Original 8
            30.0, 15.0, 1.0, -8.0, 3.0, 5.0, 2.0, 25.0, 5.0, 10.0, 150.0, 0.5  # New 12
        ]), max=torch.tensor([
            0.5, 10.0, 3.0, 1.5, 1.5, 1.0, 60.0, 250.0,  # Original 8
            150.0, 120.0, 25.0, -1.0, 50.0, 80.0, 40.0, 200.0, 80.0, 100.0, 550.0, 4.0  # New 12
        ]))
        
        return optimized_params


class MojBot(BaseBot):
    def __init__(self, parameter_optimizer=None):
        self.FLOOR_Y = 704
        self.CEILING_Y = 0
        self.BIRD_RADIUS = 25.0
        
        self.SPIKE_PADDING = 45.0
        self.prev_y = None
        self.parameter_optimizer = parameter_optimizer
        
        # Default parameters (will be overridden by neural network if available)
        # Original gap analysis parameters
        self.DISTANCE_FROM_GAP_WEIGHT = 0.1
        self.GAP_SIZE_WEIGHT = 5.0
        self.MIN_VIABLE_GAP_WEIGHT = 1.5
        self.TOP_SPIKE_MARGIN_WEIGHT = 0.7
        self.BOTTOM_SPIKE_MARGIN_WEIGHT = 0.7
        self.COIN_GREED_WEIGHT = 0.2
        self.MIN_SAFE_DEVIATION_WEIGHT = 25.0
        self.MAX_SAFE_DEVIATION_WEIGHT = 100.0
        
        # Action control parameters
        self.CEILING_AVOIDANCE_THRESHOLD = 80.0
        self.FLOOR_AVOIDANCE_THRESHOLD = 60.0
        self.TARGET_POSITION_TOLERANCE = 10.0
        self.VELOCITY_BRAKE_THRESHOLD = -4.0
        
        # Dynamic calculation parameters
        self.BASE_UNCERTAINTY_MARGIN = 20.0
        self.SPIKE_DENSITY_UNCERTAINTY_FACTOR = 30.0
        self.OVERLAP_PENALTY_PER_PAIR = 15.0
        self.EDGE_EXCLUSION_ZONE = 100.0
        self.LARGE_GAP_BONUS = 30.0
        self.MAX_OVERLAP_PENALTY = 45.0
        
        # Emergency and fallback parameters
        self.DEFAULT_TARGET_POSITION = 352.0
        self.EMERGENCY_GAP_MULTIPLIER = 2.0
        
    def update_parameters(self, spikes, coin_y, curr_y):
        """Update parameters based on current game context using neural network"""
        if self.parameter_optimizer is None:
            return
            
        # Create context vector for neural network
        active_spikes = [s for s in spikes if s != -1.0]
        spike_density = len(active_spikes) / 9.0  # Normalize to 0-1
        coin_present = 1.0 if coin_y != -1.0 else 0.0
        position_ratio = curr_y / self.FLOOR_Y  # Normalize to 0-1
        
        # Additional context features
        spike_spread = 0.0
        if len(active_spikes) > 1:
            spike_spread = (max(active_spikes) - min(active_spikes)) / self.FLOOR_Y
        
        coin_distance = 0.0
        if coin_y != -1.0:
            coin_distance = abs(coin_y - curr_y) / self.FLOOR_Y
            
        # Calculate velocity if available
        velocity = 0.0
        if hasattr(self, 'prev_y') and self.prev_y is not None:
            velocity = (curr_y - self.prev_y) / 20.0  # Normalize velocity
            
        # Calculate urgency metrics
        distance_to_ceiling = curr_y / self.FLOOR_Y
        distance_to_floor = (self.FLOOR_Y - curr_y) / self.FLOOR_Y
        
        context = torch.tensor([
            spike_density,
            coin_present, 
            position_ratio,
            spike_spread,
            coin_distance,
            len(active_spikes) / 15.0,  # Max possible spikes normalized
            min(active_spikes) / self.FLOOR_Y if active_spikes else 0.5,  # Highest spike position
            max(active_spikes) / self.FLOOR_Y if active_spikes else 0.5,  # Lowest spike position
            velocity,  # Bird movement velocity
            min(distance_to_ceiling, distance_to_floor)  # Distance to nearest boundary
        ], dtype=torch.float32)
        
        with torch.no_grad():
            optimized_params = self.parameter_optimizer(context)
            
        # Update all parameters
        self.DISTANCE_FROM_GAP_WEIGHT = optimized_params[0].item()
        self.GAP_SIZE_WEIGHT = optimized_params[1].item()
        self.MIN_VIABLE_GAP_WEIGHT = optimized_params[2].item()
        self.TOP_SPIKE_MARGIN_WEIGHT = optimized_params[3].item()
        self.BOTTOM_SPIKE_MARGIN_WEIGHT = optimized_params[4].item()
        self.COIN_GREED_WEIGHT = optimized_params[5].item()
        self.MIN_SAFE_DEVIATION_WEIGHT = optimized_params[6].item()
        self.MAX_SAFE_DEVIATION_WEIGHT = optimized_params[7].item()
        
        # Action control parameters
        self.CEILING_AVOIDANCE_THRESHOLD = optimized_params[8].item()
        self.FLOOR_AVOIDANCE_THRESHOLD = optimized_params[9].item()
        self.TARGET_POSITION_TOLERANCE = optimized_params[10].item()
        self.VELOCITY_BRAKE_THRESHOLD = optimized_params[11].item()
        
        # Dynamic calculation parameters
        self.BASE_UNCERTAINTY_MARGIN = optimized_params[12].item()
        self.SPIKE_DENSITY_UNCERTAINTY_FACTOR = optimized_params[13].item()
        self.OVERLAP_PENALTY_PER_PAIR = optimized_params[14].item()
        self.EDGE_EXCLUSION_ZONE = optimized_params[15].item()
        self.LARGE_GAP_BONUS = optimized_params[16].item()
        self.MAX_OVERLAP_PENALTY = optimized_params[17].item()
        
        # Emergency parameters
        self.DEFAULT_TARGET_POSITION = optimized_params[18].item()
        self.EMERGENCY_GAP_MULTIPLIER = optimized_params[19].item()

    def get_target_y(self, spikes, coin_y):
        active_spikes = sorted([s for s in spikes if s != -1.0])
        obstacles = [self.CEILING_Y] + active_spikes + [self.FLOOR_Y]
        
        best_target = 352.0 
        best_score = -1
        best_gap_center = 352.0
        
        # Calculate dynamic minimum viable gap accounting for random spike generation
        # Base requirement: bird diameter + safety padding
        base_gap_requirement = (self.BIRD_RADIUS * 2) + self.SPIKE_PADDING
        
        # Additional uncertainty margin for random spike positioning and overlapping
        # Analyze spike density to adjust uncertainty
        active_spike_count = len([s for s in spikes if s != -1.0])
        spike_density_factor = min(active_spike_count / 9.0, 1.0)  # Normalize to 0-1
        
        # Higher spike density = more uncertainty, larger gaps needed
        uncertainty_margin = self.BASE_UNCERTAINTY_MARGIN + (spike_density_factor * self.SPIKE_DENSITY_UNCERTAINTY_FACTOR)
        
        # Check for potential overlapping by looking at spike proximity
        overlap_penalty = 0.0
        sorted_spikes = sorted([s for s in active_spikes if s != -1.0])
        for i in range(len(sorted_spikes) - 1):
            spike_distance = sorted_spikes[i+1] - sorted_spikes[i]
            if spike_distance < self.SPIKE_PADDING * 2:  # Spikes are too close
                overlap_penalty += self.OVERLAP_PENALTY_PER_PAIR  # Add extra margin for each close pair
        
        min_viable_gap = base_gap_requirement + uncertainty_margin + min(overlap_penalty, self.MAX_OVERLAP_PENALTY)
        
        # Distance from ceiling/floor to exclude small holes near edges
        edge_exclusion_zone = self.EDGE_EXCLUSION_ZONE
        
        # Get current bird position (we'll get this from take_action call)
        current_y = getattr(self, 'current_bird_y', 352.0)
        
        # Evaluate all viable gaps with a scoring system
        for i in range(len(obstacles) - 1):
            top_obstacle = obstacles[i]
            bottom_obstacle = obstacles[i+1]
            
            safe_top = top_obstacle + self.SPIKE_PADDING
            safe_bottom = bottom_obstacle - self.SPIKE_PADDING
            
            if safe_bottom > safe_top:
                gap_size = safe_bottom - safe_top
                gap_center = (safe_top + safe_bottom) / 2
                
                # Exclude gaps that are too small for the bird
                if gap_size < min_viable_gap:
                    continue
                    
                # Exclude small gaps near ceiling or floor
                too_close_to_ceiling = (top_obstacle == self.CEILING_Y and gap_size < edge_exclusion_zone)
                too_close_to_floor = (bottom_obstacle == self.FLOOR_Y and gap_size < edge_exclusion_zone)
                
                if too_close_to_ceiling or too_close_to_floor:
                    continue
                
                # Calculate score: heavily favor gap size, with minor distance penalty
                distance_from_current = abs(gap_center - current_y)
                
                # Score formula: prioritize gap size significantly over distance
                # Large gaps get high base scores, small distance penalty
                gap_score = gap_size * self.GAP_SIZE_WEIGHT
                distance_penalty = min(distance_from_current * self.DISTANCE_FROM_GAP_WEIGHT, edge_exclusion_zone / 2.0)  # Dynamic penalty cap
                
                total_score = gap_score - distance_penalty
                
                # Bonus for very large gaps (encourage using big gaps even if farther)
                if gap_size > min_viable_gap * self.MIN_VIABLE_GAP_WEIGHT:
                    total_score += self.LARGE_GAP_BONUS
                
                # Select gap with best score
                if total_score > best_score:
                    best_score = total_score
                    best_gap_center = gap_center
                    best_target = gap_center
        
        # Handle case where no viable gaps exist due to random spike generation
        if best_score <= 0:
            # Emergency fallback: find the largest available gap even if suboptimal
            largest_gap_size = 0
            emergency_target = self.DEFAULT_TARGET_POSITION
            
            for i in range(len(obstacles) - 1):
                top_obstacle = obstacles[i]
                bottom_obstacle = obstacles[i+1]
                
                safe_top = top_obstacle + self.SPIKE_PADDING * self.TOP_SPIKE_MARGIN_WEIGHT  # Reduced safety margin
                safe_bottom = bottom_obstacle - self.SPIKE_PADDING * self.BOTTOM_SPIKE_MARGIN_WEIGHT
                
                if safe_bottom > safe_top:
                    gap_size = safe_bottom - safe_top
                    if gap_size > largest_gap_size:
                        largest_gap_size = gap_size
                        emergency_target = (safe_top + safe_bottom) / 2
            
            # If even emergency gaps are too small, aim for default position
            if largest_gap_size < self.BIRD_RADIUS * self.EMERGENCY_GAP_MULTIPLIER:
                best_target = self.DEFAULT_TARGET_POSITION
            else:
                best_target = emergency_target
                best_gap_center = emergency_target
        
        # If we found a good gap, consider coin collection only if it's safe
        if coin_y != -1.0 and best_score > 0:
            # Check if coin is within the best gap and not too far from center
            coin_distance_from_center = abs(coin_y - best_gap_center)
            max_safe_deviation = (best_gap_center - current_y) * self.COIN_GREED_WEIGHT  # Slightly increased from 0.1
            max_safe_deviation = max(max_safe_deviation, self.MIN_SAFE_DEVIATION_WEIGHT)  # Increased minimum from 20.0
            max_safe_deviation = min(max_safe_deviation, self.MAX_SAFE_DEVIATION_WEIGHT)  # Increased maximum from 60.0
            
            if coin_distance_from_center <= max_safe_deviation:
                best_target = coin_y
            # Otherwise stick to the perfect center of the best scoring gap
                            
        return best_target

    def take_action(self, obs: np.ndarray) -> int:
        curr_y = obs[1]
        coin_y = obs[5]
        spikes = obs[6:15]
        
        # Update parameters based on current context
        self.update_parameters(spikes, coin_y, curr_y)
        
        # Store current position for gap selection algorithm
        self.current_bird_y = curr_y
        
        velocity = 0.0
        if self.prev_y is not None:
            velocity = curr_y - self.prev_y
        self.prev_y = curr_y

        target_y = self.get_target_y(spikes, coin_y)
        
        if curr_y < self.CEILING_AVOIDANCE_THRESHOLD: return 0
        if velocity < self.VELOCITY_BRAKE_THRESHOLD: return 0

        if curr_y > target_y + self.TARGET_POSITION_TOLERANCE:
            return 1
            
        if curr_y > self.FLOOR_Y - self.FLOOR_AVOIDANCE_THRESHOLD:
            return 1


# Global variable to store the trained neural network
trained_parameter_optimizer = None

def create_bot() -> BaseBot:
    global trained_parameter_optimizer
    
    # Try to load from model.pth if available
    if os.path.exists('model.pth'):
        try:
            model_optimizer = ParameterOptimizer()
            model_optimizer.load_state_dict(torch.load('model.pth'))
            model_optimizer.eval()
            print("Loaded trained bot from model.pth")
            return MojBot(parameter_optimizer=model_optimizer)
        except Exception as e:
            print(f"Failed to load model.pth: {e}")
            print("Falling back to trained_parameter_optimizer")
    
    return MojBot(parameter_optimizer=trained_parameter_optimizer)

def calculate_reward(game_state: dict) -> float:
    if game_state["player_dead"]: return -10.0
    return 1.0 + game_state["collected_coins"] * 5.0

def train_bot():
    """Train the parameter optimization neural network"""
    global trained_parameter_optimizer
    
    # Initialize neural network
    parameter_optimizer = ParameterOptimizer()
    optimizer = optim.Adam(parameter_optimizer.parameters(), lr=0.001)
    
    # Training parameters
    population_size = 200
    generations = 1500
    elite_size = 10
    
    print("Starting parameter optimization training...")
    
    # Evolution Strategy training
    best_score = -float('inf')
    best_params = None
    generation_scores = []
    
    for generation in range(generations):
        population = []
        scores = []
        
        # Generate population
        for _ in range(population_size):
            # Create random context for training
            context = torch.rand(10)  # Updated context size
            
            # Get parameters from neural network
            with torch.no_grad():
                params = parameter_optimizer(context)
            
            # Create bot with these parameters
            bot = MojBot()
            # Assign all 20 parameters
            bot.DISTANCE_FROM_GAP_WEIGHT = params[0].item()
            bot.GAP_SIZE_WEIGHT = params[1].item()
            bot.MIN_VIABLE_GAP_WEIGHT = params[2].item()
            bot.TOP_SPIKE_MARGIN_WEIGHT = params[3].item()
            bot.BOTTOM_SPIKE_MARGIN_WEIGHT = params[4].item()
            bot.COIN_GREED_WEIGHT = params[5].item()
            bot.MIN_SAFE_DEVIATION_WEIGHT = params[6].item()
            bot.MAX_SAFE_DEVIATION_WEIGHT = params[7].item()
            bot.CEILING_AVOIDANCE_THRESHOLD = params[8].item()
            bot.FLOOR_AVOIDANCE_THRESHOLD = params[9].item()
            bot.TARGET_POSITION_TOLERANCE = params[10].item()
            bot.VELOCITY_BRAKE_THRESHOLD = params[11].item()
            bot.BASE_UNCERTAINTY_MARGIN = params[12].item()
            bot.SPIKE_DENSITY_UNCERTAINTY_FACTOR = params[13].item()
            bot.OVERLAP_PENALTY_PER_PAIR = params[14].item()
            bot.EDGE_EXCLUSION_ZONE = params[15].item()
            bot.LARGE_GAP_BONUS = params[16].item()
            bot.MAX_OVERLAP_PENALTY = params[17].item()
            bot.DEFAULT_TARGET_POSITION = params[18].item()
            bot.EMERGENCY_GAP_MULTIPLIER = params[19].item()
            
            # Simulate score (in real training, you'd run the actual game)
            # For now, use a heuristic fitness function
            fitness = evaluate_parameters(params)
            
            population.append((context, params, fitness))
            scores.append(fitness)
        
        # Select elite individuals
        population.sort(key=lambda x: x[2], reverse=True)
        elite = population[:elite_size]
        
        current_best = elite[0][2]
        if current_best > best_score:
            best_score = current_best
            best_params = elite[0][1]
            
        generation_scores.append(current_best)
        
        # Update neural network using elite individuals
        total_loss = 0
        for context, target_params, fitness in elite:
            predicted_params = parameter_optimizer(context)
            
            # Loss is weighted by fitness (better performers have more influence)
            weight = fitness / sum([individual[2] for individual in elite])
            loss = weight * nn.MSELoss()(predicted_params, target_params)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if generation % 10 == 0:
            print(f"Generation {generation}: Best Score = {current_best:.3f}, Avg Score = {np.mean(scores):.3f}")
    
    # Save trained model
    trained_parameter_optimizer = parameter_optimizer
    torch.save(parameter_optimizer.state_dict(), 'parameter_optimizer.pth')
    
    print(f"Training completed! Best score: {best_score:.3f}")
    print("Best parameters found:")
    if best_params is not None:
        param_names = [
            "DISTANCE_FROM_GAP_WEIGHT", "GAP_SIZE_WEIGHT", "MIN_VIABLE_GAP_WEIGHT",
            "TOP_SPIKE_MARGIN_WEIGHT", "BOTTOM_SPIKE_MARGIN_WEIGHT", "COIN_GREED_WEIGHT",
            "MIN_SAFE_DEVIATION_WEIGHT", "MAX_SAFE_DEVIATION_WEIGHT", "CEILING_AVOIDANCE_THRESHOLD",
            "FLOOR_AVOIDANCE_THRESHOLD", "TARGET_POSITION_TOLERANCE", "VELOCITY_BRAKE_THRESHOLD",
            "BASE_UNCERTAINTY_MARGIN", "SPIKE_DENSITY_UNCERTAINTY_FACTOR", "OVERLAP_PENALTY_PER_PAIR",
            "EDGE_EXCLUSION_ZONE", "LARGE_GAP_BONUS", "MAX_OVERLAP_PENALTY", 
            "DEFAULT_TARGET_POSITION", "EMERGENCY_GAP_MULTIPLIER"
        ]
        for name, value in zip(param_names, best_params):
            print(f"  {name}: {value:.4f}")

def evaluate_parameters(params):
    """
    Fitness function to evaluate parameter quality
    This is a heuristic evaluation - in real training you'd run actual games
    """
    # Extract parameters
    distance_weight = params[0].item()
    gap_weight = params[1].item()
    viable_gap_weight = params[2].item()
    top_margin_weight = params[3].item()
    bottom_margin_weight = params[4].item()
    coin_weight = params[5].item()
    min_deviation = params[6].item()
    max_deviation = params[7].item()
    
    # Heuristic scoring based on parameter balance
    score = 0.0
    
    # Gap size should be prioritized (higher weight = better)
    score += gap_weight * 2.0
    
    # Distance weight should be moderate (not too high, not too low)
    optimal_distance = 0.15
    score -= abs(distance_weight - optimal_distance) * 10
    
    # Margin weights should be balanced
    margin_balance = abs(top_margin_weight - bottom_margin_weight)
    score -= margin_balance * 5
    
    # Coin greed should be moderate
    if 0.1 <= coin_weight <= 0.4:
        score += 5.0
    else:
        score -= abs(coin_weight - 0.25) * 10
    
    # Deviation parameters should be well-spaced
    if max_deviation > min_deviation + 20:
        score += 3.0
    else:
        score -= 5.0
    
    # Penalty for extreme values
    extreme_penalty = 0
    for param in params:
        if param < 0.01 or param > 500:
            extreme_penalty += 10
    score -= extreme_penalty
    
    # Add some randomness to simulate game variability
    score += random.uniform(-2.0, 2.0)
    
    return score

# Load pre-trained model if available
def load_trained_model():
    """Load previously trained model if it exists"""
    global trained_parameter_optimizer
    
    if os.path.exists('parameter_optimizer.pth'):
        try:
            trained_parameter_optimizer = ParameterOptimizer()
            trained_parameter_optimizer.load_state_dict(torch.load('parameter_optimizer.pth'))
            trained_parameter_optimizer.eval()
            print("Loaded pre-trained parameter optimizer")
        except Exception as e:
            print(f"Failed to load pre-trained model: {e}")
            trained_parameter_optimizer = None

# Load model when module is imported
load_trained_model()