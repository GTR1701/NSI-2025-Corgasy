from base_bot import BaseBot
import numpy as np

BENCHMARK_EPISODES = 500
WATCH_GAME = True

class MojBot(BaseBot):
    def __init__(self):
        self.FLOOR_Y = 704
        self.CEILING_Y = 0
        self.BIRD_RADIUS = 25.0
        
        self.SPIKE_PADDING = 45.0
        self.prev_y = None

    def get_target_y(self, spikes, coin_y):
        active_spikes = sorted([s for s in spikes if s != -1.0])
        obstacles = [self.CEILING_Y] + active_spikes + [self.FLOOR_Y]
        
        best_target = 352.0 
        best_score = -1
        best_gap_center = 352.0
        
        # Minimum gap size needed for the bird (bird diameter + safety margin)
        min_viable_gap = (self.BIRD_RADIUS * 2) + self.SPIKE_PADDING
        
        # Distance from ceiling/floor to exclude small holes near edges
        edge_exclusion_zone = 100.0
        
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
                gap_score = gap_size * 2.0  # Double weight for gap size
                distance_penalty = min(distance_from_current * 0.1, 50.0)  # Cap distance penalty
                
                total_score = gap_score - distance_penalty
                
                # Bonus for very large gaps (encourage using big gaps even if farther)
                if gap_size > min_viable_gap * 1.5:
                    total_score += 30.0
                
                # Select gap with best score
                if total_score > best_score:
                    best_score = total_score
                    best_gap_center = gap_center
                    best_target = gap_center
        
        # If we found a good gap, consider coin collection only if it's safe
        if coin_y != -1.0 and best_score > 0:
            # Check if coin is within the best gap and not too far from center
            coin_distance_from_center = abs(coin_y - best_gap_center)
            max_safe_deviation = (best_gap_center - current_y) * 0.12  # Slightly increased from 0.1
            max_safe_deviation = max(max_safe_deviation, 25.0)  # Increased minimum from 20.0
            max_safe_deviation = min(max_safe_deviation, 70.0)  # Increased maximum from 60.0
            
            if coin_distance_from_center <= max_safe_deviation:
                best_target = coin_y
            # Otherwise stick to the perfect center of the best scoring gap
                            
        return best_target

    def take_action(self, obs: np.ndarray) -> int:
        curr_y = obs[1]
        coin_y = obs[5]
        spikes = obs[6:15]
        
        # Store current position for gap selection algorithm
        self.current_bird_y = curr_y
        
        velocity = 0.0
        if self.prev_y is not None:
            velocity = curr_y - self.prev_y
        self.prev_y = curr_y

        target_y = self.get_target_y(spikes, coin_y)
        
        if curr_y < 80: return 0
        if velocity < -4.0: return 0

        if curr_y > target_y + 10:
            return 1
            
        if curr_y > self.FLOOR_Y - 60:
            return 1

        return 0


def create_bot() -> BaseBot:
    return MojBot()

def calculate_reward(game_state: dict) -> float:
    if game_state["player_dead"]: return -10.0
    return 1.0 + game_state["collected_coins"] * 5.0

def train_bot():
    pass