import pygame
import numpy as np

# Pygame setup
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# PSO Parameters
num_particles = 30
dimensions = 2  # x and y positions
w = 0.9         # Initial inertia weight
c1 = 2.0        # Cognitive component
c2 = 2.0        # Social component
generations = 10
max_velocity = 10
min_velocity = -10

# Adaptive PSO Parameters
w_max = 0.9
w_min = 0.4

# Initialize particles (position and velocity)
particles_pos = np.random.uniform(low=0, high=[width, height], size=(num_particles, dimensions))
particles_vel = np.random.uniform(low=-5, high=5, size=(num_particles, dimensions))

# Initialize best positions
personal_best_pos = np.copy(particles_pos)
personal_best_value = np.full(num_particles, np.inf)
global_best_pos = None
global_best_value = np.inf

# Objective function: Minimize distance to mouse cursor
def objective_function(position, target_position):
    distance = np.linalg.norm(position - target_position) ** 2
    return distance

# Adaptive Inertia weight function
def adaptive_inertia_weight(current_gen, max_gen):
    return w_max - (w_max - w_min) * (current_gen / max_gen)

# Main Pygame loop
running = True
generation = 0

while running:
    # Handle events (e.g., quit)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Adaptive PSO: Update inertia weight
    w = adaptive_inertia_weight(generation, generations)
    target_position = np.array(pygame.mouse.get_pos())
    for i in range(num_particles):
        # Evaluate fitness of each particle
        fitness = objective_function(particles_pos[i], target_position)
        
        # Update personal best
        if fitness < personal_best_value[i]:
            personal_best_value[i] = fitness
            personal_best_pos[i] = particles_pos[i]
        
        # Update global best
        if fitness < global_best_value:
            global_best_value = fitness
            global_best_pos = particles_pos[i]

    # Update velocities and positions
    for i in range(num_particles):
        r1, r2 = np.random.rand(2)  # Random coefficients between 0 and 1
        
        # Update velocity using adaptive inertia weight
        cognitive_component = c1 * r1 * (personal_best_pos[i] - particles_pos[i])
        social_component = c2 * r2 * (global_best_pos - particles_pos[i])
        particles_vel[i] = w * particles_vel[i] + cognitive_component + social_component

        # Clamp velocity to avoid overshooting
        particles_vel[i] = np.clip(particles_vel[i], min_velocity, max_velocity)

        # Update position
        particles_pos[i] += particles_vel[i]

        # Keep particles within the screen boundaries
        # particles_pos[i] = np.clip(particles_pos[i], [0, 0], [width, height])

    # Draw everything
    screen.fill((255, 255, 255))  # Clear screen with white background

    # Draw particles
    for pos in particles_pos:
        pygame.draw.circle(screen, (0, 150, 0), pos.astype(int), 5)

    # Draw the target (mouse position)
    target_position = pygame.mouse.get_pos()
    pygame.draw.circle(screen, (255, 0, 0), target_position, 8)

    # Update display
    pygame.display.flip()
    clock.tick(30)  # Limit to 30 FPS
    
    # Increment generation
    
    generation += 1
    print(f"Generation {generation:3d}, Best Fitness: {global_best_value:.3f}, Best Position: {global_best_pos}")
    if generation > generations:
        generation = 0  # Restart for re-evaluation if target changes significantly
        global_best_value = np.inf
        personal_best_value = np.full(num_particles, np.inf)
        # personal_best_pos *= 0.98

pygame.quit()
