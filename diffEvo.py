import pygame
import numpy as np

# Pygame setup
pygame.init()
width, height = 600, 600
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Differential Evolution Parameters
population_size = 20
dimensions = 2  # x and y movement
F = 0.9         # Mutation factor
CR = 0.5        # Crossover probability
# generations = 200
relax_interval = 10
relax_fraction = 0.2  # Relax 20% of the population

# Agent setup
agent_radius = 10
target_radius = 5
agent_color = (0, 100, 250)
target_color = (250, 50, 50)
background_color = (255, 255, 255)

# Initialize Population: Each agent has an initial position and velocity
population = np.random.uniform(low=-5, high=5, size=(population_size, dimensions))

# Agent position (start at the center of the screen)
agent_position = np.array([width / 2, height / 2])

# Objective function: Minimize distance to mouse cursor
def objective_function(self_position, target_position):
    return np.linalg.norm(self_position - target_position) ** 2

# Relax operation: Reinitialize a portion of the population
def relax_population(population, fraction):
    num_to_relax = int(population_size * fraction)
    indices_to_relax = np.random.choice(population_size, num_to_relax, replace=False)
    population[indices_to_relax] = np.random.uniform(low=-5, high=5, size=(num_to_relax, dimensions))

# Evaluate fitness for the initial population
fitness = np.ones(population_size) * float('inf')

# Differential Evolution Loop
running = True
generation = 0

while running:
    # Handle events (e.g., quit)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    target_position = np.array(pygame.mouse.get_pos())
    # Differential Evolution Steps
    for i in range(population_size):
        # Mutation
        idxs = [idx for idx in range(population_size) if idx != i]
        r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
        mutant_vector = population[r1] + F * (population[r2] - population[r3])
        
        # Crossover
        trial_vector = np.copy(population[i])
        for d in range(dimensions):
            if np.random.rand() < CR:
                trial_vector[d] = mutant_vector[d]
        
        # Selection
        trial_fitness = objective_function(agent_position+trial_vector, target_position)
        if trial_fitness < fitness[i]:
            population[i] = trial_vector
            fitness[i] = trial_fitness

    # Relax population periodically
    if generation % relax_interval == 0:
        fitness = np.array([objective_function(agent_position+p, target_position) for p in population])
        relax_population(population, relax_fraction)

    # Get the best solution and move the agent
    # print(np.mean(fitness), np.std(fitness))
    best_index = np.argmin(fitness)
    best_velocity = population[best_index]
    best_velocity = best_velocity / np.linalg.norm(best_velocity) * 10 if np.linalg.norm(best_velocity) > 10 else best_velocity
    print(f"Generation {generation:3d}, Best Fitness: {fitness[best_index]:.3f}, Best Velocity: {best_velocity}")
    agent_position += best_velocity
    generation += 1

    # Draw everything
    screen.fill(background_color)
    
    # Draw the agent
    pygame.draw.circle(screen, agent_color, agent_position.astype(int), agent_radius)
    for p in population:
        pygame.draw.line(screen, (0, 0, 0), agent_position, agent_position + p * 10, 1)

    
    # Draw the target (mouse position)
    
    pygame.draw.circle(screen, target_color, target_position, target_radius)

    # Update display
    pygame.display.flip()
    clock.tick(30)  # Limit to 30 FPS

pygame.quit()
