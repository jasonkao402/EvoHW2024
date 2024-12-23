import matplotlib.pyplot as plt
import numpy as np
import os
import time
import pandas as pd


WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600
FIELD_SIZE = 10
ZOOM = 500 / FIELD_SIZE
OFFSET = (WINDOW_WIDTH - FIELD_SIZE * ZOOM) // 2
OFFSET_POS = np.ones(2) *  OFFSET

# WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
# pygame.display.set_caption("Dodgeball Simulation GA")

WHITE = (255, 255, 255)
GRAY  = (32, 32, 32)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)
GREEN = (0, 232, 152)
BLUE  = (0, 118, 214)
CYAN  = (16, 210, 250)

running = True
drawing = True
# clock = pygame.time.Clock()
# font = pygame.freetype.SysFont('Consolas', 20)
def generate_base_track(num_points):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radii = np.ones(num_points)  # Start with a perfect circle
    return angles, radii

from noise import pnoise1  # Perlin noise library

def apply_perlin_noise(angles, radii, scale=5, noise_amplitude=0.3, seed=42):
    np.random.seed(seed)
    radii_with_noise = radii + np.array([pnoise1(angle * scale) for angle in angles]) * noise_amplitude
    return radii_with_noise

def polar_to_cartesian(angles, radii):
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return x, y

def fitness_function(angles, radii):
    # Ensure start and end points align
    loop_closeness = abs(radii[0] - radii[-1])
    
    # Smoothness (no abrupt changes in radius)
    smoothness = np.mean(np.abs(np.diff(radii, 2)))  # Second-order differences
    
    # Length (ensure the track is of reasonable size)
    length = np.sum(np.sqrt(np.diff(radii)**2 + np.diff(angles)**2))
    desired_length = 20  # Example target length
    
    fitness = - (loop_closeness + smoothness + abs(length - desired_length))
    return fitness

def evolve_tracks(num_generations=100, population_size=50, mutation_rate=0.1):
    population = [apply_perlin_noise(*generate_base_track(50)) for _ in range(population_size)]
    for generation in range(num_generations):
        fitness_scores = [fitness_function(*track) for track in population]
        
        # Selection (elitism: keep the top 2 tracks)
        sorted_population = [track for _, track in sorted(zip(fitness_scores, population), reverse=True)]
        next_generation = sorted_population[:2]
        
        # Crossover
        while len(next_generation) < population_size:
            parent1, parent2 = np.random.choice(sorted_population[:10], 2, replace=False)
            midpoint = len(parent1[1]) // 2
            child_radii = np.concatenate((parent1[1][:midpoint], parent2[1][midpoint:]))
            child_angles = np.linspace(0, 2 * np.pi, len(child_radii), endpoint=False)
            next_generation.append((child_angles, child_radii))
        
        # Mutation
        for i in range(2, population_size):
            if np.random.rand() < mutation_rate:
                next_generation[i] = (next_generation[i][0], apply_perlin_noise(
                    next_generation[i][0], next_generation[i][1], noise_amplitude=0.05))
        
        population = next_generation
    return population

def plot_track(angles, radii):
    x, y = polar_to_cartesian(angles, radii)
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, '-o')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Generated Race Track")
    plt.show()

# Generate and visualize a track
angles, radii = generate_base_track(50)
radii = apply_perlin_noise(angles, radii)
plot_track(angles, radii)

# frameCount = 0
# while running:
    # WINDOW.fill(GRAY)
    
    # textSur, rect = font.render(f"Frame: {frameCount}", GREEN)
    # WINDOW.blit(textSur, OFFSET_POS/2)
    # pygame.draw.rect(WINDOW, GREEN, (OFFSET, OFFSET, FIELD_SIZE * ZOOM, FIELD_SIZE * ZOOM), 3)

    # pygame.display.flip()
    # clock.tick(60)
    # frameCount += 1

# 結束 pygame
# pygame.quit()