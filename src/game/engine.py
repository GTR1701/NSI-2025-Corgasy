import os
import pygame
import random
import sys


class MockSound:
    def play(self):
        pass

    def set_volume(self, v):
        pass


class MockMixer:
    def Sound(self, file):
        return MockSound()

    def init(self):
        pass

    def quit(self):
        pass


class GameEngine:
    def __init__(self, headless=False):
        self.headless = headless
        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            sys.modules["pygame.mixer"] = MockMixer()
            pygame.mixer = MockMixer()
        else:
            pygame.mixer.init()

        pygame.init()

        self.scale = 2
        self.user_x = 288 * self.scale
        self.user_y = 352 * self.scale
        self.fps = 60
        self.clock = pygame.time.Clock()

        if self.headless:
            self.screen = pygame.display.set_mode((self.user_x, self.user_y))
        else:
            self.screen = pygame.display.set_mode((self.user_x, self.user_y))
            pygame.display.set_caption("Game")

        from src.game.coin_class import Coin
        from src.game.player_class import Player
        from src.game.spike_classes import (
            Ceilling_Spike,
            East_Wall_Spike,
            Floor_Spike,
            West_Wall_Spike,
        )

        self.Player = Player
        self.Coin = Coin
        self.SpikeClasses = {
            "floor": Floor_Spike,
            "ceiling": Ceilling_Spike,
            "east": East_Wall_Spike,
            "west": West_Wall_Spike,
        }

        self.coin_total = 0

        try:
            self.font = pygame.font.Font("font/Pixeltype.ttf", 40 * self.scale)
        except:
            self.font = pygame.font.SysFont("Arial", 40)

        self.player = None
        self.floor_spikes = []
        self.ceiling_spikes = []
        self.east_spikes = []
        self.west_spikes = []
        self.coin_list = []
        self.score = 0
        self.game_over = False

    def draw_text(self, text, size, x, y):
        if self.headless:
            return
        font_surface = self.font.render(text, True, "white")
        text_rect = font_surface.get_rect()
        text_rect.midtop = (x, y)
        self.screen.blit(font_surface, text_rect)

    def generate_coins(self):
        if len(self.coin_list) < 1:
            coin = self.Coin(self.scale)
            coin.set_position(
                random.randint(40 * self.scale, self.user_x - 40 * self.scale),
                random.randint(40 * self.scale, self.user_y - 40 * self.scale),
            )
            self.coin_list.append(coin)

    def generate_spikes(self, score):
        east_spikes = []
        west_spikes = []

        number_of_spikes = 2 + (score // 5)
        if number_of_spikes > 9:
            number_of_spikes = 9

        current_wall = "east" if self.player.rect.x > self.user_x // 2 else "west"

        available_slots = list(range(1, 12))
        random.shuffle(available_slots)

        for i in range(number_of_spikes):
            if not available_slots:
                break
            slot = available_slots.pop()
            y_pos = slot * (25 * self.scale)

            if current_wall == "west":
                spike_e = self.SpikeClasses["east"](self.scale)
                spike_e.set_position(self.user_x - spike_e.rect.width, y_pos)
                east_spikes.append(spike_e)

            else:
                spike_w = self.SpikeClasses["west"](self.scale)
                spike_w.set_position(0, y_pos)
                west_spikes.append(spike_w)

        return east_spikes, west_spikes

    def reset_game(self):
        self.player = self.Player(self.scale, self.user_x, self.user_y)

        self.floor_spikes = [
            self.SpikeClasses["floor"](self.scale) for _ in range(0, self.user_x, 30)
        ]
        for i, spike in enumerate(self.floor_spikes):
            spike.set_position(i * 30, self.user_y - 30)

        self.ceiling_spikes = [
            self.SpikeClasses["ceiling"](self.scale) for _ in range(0, self.user_x, 30)
        ]
        for i, spike in enumerate(self.ceiling_spikes):
            spike.set_position(i * 30, 0)

        self.east_spikes = []
        self.west_spikes = []
        self.coin_list = []
        self.score = 0
        self.coin_total = 0
        self.game_over = False

        return self.get_state_dict()

    def step(self, action):
        if self.game_over:
            return self.get_state_dict(), True, {}

        if action == 1:
            self.player.jump()

        self.player.update()

        wall_hit = self.player.check_wall_collision()

        if wall_hit != 0:
            self.player.after_collision()
            self.score += 1
            self.generate_coins()

            e_spikes, w_spikes = self.generate_spikes(self.score)
            self.east_spikes = e_spikes
            self.west_spikes = w_spikes

        # Kolizja z kolcami
        active_spikes = (
            self.floor_spikes
            + self.ceiling_spikes
            + self.east_spikes
            + self.west_spikes
        )
        for spike in active_spikes:
            if spike.rect.colliderect(self.player.rect):
                self.player.death()
                self.game_over = True
                return self.get_state_dict(), True, {}

        # Zbieranie monet
        for coin in self.coin_list:
            if coin.rect.colliderect(self.player.rect):
                self.coin_list.remove(coin)
                self.coin_total += 1

        # Nagroda z funkcji nagrody uczestnika

        return self.get_state_dict(), False, {}

    def get_state_dict(self):
        return {
            "player_pos_x": self.player.rect.x,
            "player_pos_y": self.player.rect.y,
            "player_direction": 1 if self.player.velocity > 0 else -1,
            "player_gravity": self.player.gravity,
            "spikes_pos_y": [
                s.rect.y
                for s in (
                    self.east_spikes if self.player.velocity > 0 else self.west_spikes
                )
            ],
            "coin_x": self.coin_list[0].rect.x if self.coin_list else -1,
            "coin_y": self.coin_list[0].rect.y if self.coin_list else -1,
            "score": self.score,
            "collected_coins": self.coin_total,
            "player_dead": self.player.dead,
            "right_wall_pos": self.user_x,
            "left_wall_pos": 0,
            "floor_pos": self.user_y,
            "ceiling_pos": 0,
        }

    def render_frame(self):
        if self.headless:
            return

        self.screen.fill((200, 200, 200))
        pygame.draw.circle(
            self.screen, "white", (self.user_x // 2, self.user_y // 2), 60 * self.scale
        )
        self.draw_text(
            str(self.score), 100 * self.scale, self.user_x // 2, self.user_y // 4 * 3
        )

        player_group = pygame.sprite.Group(self.player)
        player_group.draw(self.screen)

        active_spikes = (
            self.floor_spikes
            + self.ceiling_spikes
            + self.east_spikes
            + self.west_spikes
        )
        spike_group = pygame.sprite.Group(active_spikes)
        spike_group.draw(self.screen)

        coin_group = pygame.sprite.Group(self.coin_list)
        coin_group.draw(self.screen)

        pygame.display.update()
        self.clock.tick(self.fps)
