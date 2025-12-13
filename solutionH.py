from base_bot import BaseBot
import numpy as np

BENCHMARK_EPISODES = 50
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
        max_gap = 0
        
        for i in range(len(obstacles) - 1):
            top_obstacle = obstacles[i]
            bottom_obstacle = obstacles[i+1]
            
            safe_top = top_obstacle + self.SPIKE_PADDING
            safe_bottom = bottom_obstacle - self.SPIKE_PADDING
            
            if safe_bottom > safe_top:
                gap_size = safe_bottom - safe_top
                mid_gap = (safe_top + safe_bottom) / 2
                
                if gap_size > max_gap:
                    max_gap = gap_size

                    best_target = mid_gap
                    
                    if coin_y != -1.0:
                        risk_factor = abs(coin_y - mid_gap)
                        
                        if risk_factor < 30.0:
                            best_target = coin_y
                            
        return best_target

    def take_action(self, obs: np.ndarray) -> int:
        curr_y = obs[1]
        coin_y = obs[5]
        spikes = obs[6:15]
        
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