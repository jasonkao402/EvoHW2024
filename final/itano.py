import pygame
import numpy as np
import random
import math
from collections import deque
# Pygame setup
pygame.init()
width, height = 800, 800
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
TRAIL_COLOR = (255, 100, 100)

# Missile parameters
num_missiles = 16
missile_speed = 10
turbulence_strength = 2
turbulence_angle_range = 0.3
max_turn_rate = 0.05
trail_length = 64
# Player (F-22 jet) setup
player_pos = np.array([width / 2, height / 2])

# Missile class
class Missile:
    def __init__(self, position, direction=None):
        self.position = np.array(position)
        self.direction = np.random.uniform(-1, 1, 2)
        self.direction = self.direction / np.linalg.norm(self.direction)  # Normalize to make it a unit vector
        self.trail = deque(maxlen=trail_length)
        self.random_turbulence = np.random.uniform(-turbulence_strength, turbulence_strength, 2)
        self.lifetime = 360
    
    def reset(self):
        self.lifetime = 360
        self.position = np.random.uniform(0, width, 2)
        self.direction = np.random.uniform(-1, 1, 2)
        self.direction = self.direction / np.linalg.norm(self.direction)

    def update(self, target_pos):
        # Calculate target direction
        
        target_direction = target_pos - self.position

        if self.lifetime < 0:
            self.reset()

        elif self.lifetime % 30 == 0:
            turbulence = turbulence_strength * np.clip(np.linalg.norm(target_direction)*.1, 1, 100)
            self.random_turbulence = np.random.uniform(-turbulence, turbulence, 2)
            # target_direction += self.random_turbulence
        target_direction += self.random_turbulence
        target_direction /= np.linalg.norm(target_direction)  # Normalize

        # Calculate the cross product (z-component only for 2D)
        cross_prod = self.direction[0] * target_direction[1] - self.direction[1] * target_direction[0]

        # Determine rotation direction based on cross product
        turn_direction = np.sign(cross_prod)

        # Calculate the angle between the current direction and target direction
        dot_product = np.dot(self.direction, target_direction)
        angle_to_target = math.acos(np.clip(dot_product, -1.0, 1.0))  # Clamp for stability

        # Limit the turn rate to create inertia
        turn_angle = min(max_turn_rate, angle_to_target) * turn_direction

        # Rotate direction vector by the limited angle
        cos_theta = math.cos(turn_angle)
        sin_theta = math.sin(turn_angle)
        new_direction = np.array([
            self.direction[0] * cos_theta - self.direction[1] * sin_theta,
            self.direction[0] * sin_theta + self.direction[1] * cos_theta
        ])
        
        # Update missile direction and position
        self.direction = new_direction
        self.position += self.direction * missile_speed

        # Add current position to trail, limit trail length
        self.trail.append(self.position.copy())
        self.lifetime -= 1

    def draw(self, surface):
        # Draw trail with fading effect
        for i, pos in enumerate(self.trail):
            fade_factor = int(255 * (i / len(self.trail)))
            trail_color = (TRAIL_COLOR[0], TRAIL_COLOR[1], TRAIL_COLOR[2], fade_factor)
            trail_size  = (i / len(self.trail)) * 5 + 2
            pygame.draw.circle(surface, trail_color, pos.astype(int), trail_size)
        # Draw missile as a solid circle
        pygame.draw.circle(surface, RED, self.position.astype(int), 5)
        # pygame.draw.line(surface, BLACK, self.position, self.position + self.direction * 50, 2)

# Initialize missiles
missiles = [Missile(
    position=pos,
) for pos in np.random.uniform(0, width, (num_missiles, 2))]

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update player position to follow mouse
    player_pos = np.array(pygame.mouse.get_pos())

    # Clear screen
    screen.fill(WHITE)

    # Draw player (jet)
    pygame.draw.circle(screen, BLUE, player_pos.astype(int), 8)

    # Update and draw missiles
    for missile in missiles:
        missile.update(player_pos)
        missile.draw(screen)

    # Refresh display
    pygame.display.flip()
    clock.tick(60)
pygame.quit()
