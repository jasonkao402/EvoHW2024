import numpy as np
import matplotlib.pyplot as plt, matplotlib.colors as mcolors
import random
from tqdm import trange

RUNS = 10
# Fitness function for the OneMax problem
def fitness(individual):
    return sum(individual)**2

# Roulette wheel selection
def roulette_wheel_selection(population, fitness_values, size=2):
    total_fitness = sum(fitness_values)
    selection_probs = [f / total_fitness for f in fitness_values]
    idx = np.random.choice(len(population), size=size, p=selection_probs)
    return population[idx[0]], population[idx[1]]


# tournament selection
def tournament_selection(population, fitness_values, tournament_size=2):
    tournament_indices = np.random.choice(len(population), size=tournament_size*2, replace=False)
    tournament_fitness = [fitness_values[i] for i in tournament_indices]
    idx = np.argmax(tournament_fitness[:tournament_size]), np.argmax(tournament_fitness[tournament_size:])
    return population[tournament_indices[idx[0]]], population[tournament_indices[idx[1]]]

# One-point crossover
def one_point_crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    offspring1 = parent1[:point] + parent2[point:]
    offspring2 = parent2[:point] + parent1[point:]
    return offspring1, offspring2

# Genetic Algorithm function
def genetic_algorithm(pop_size=200, bit_length=50, generations=100, runs=RUNS, pc=1.0, selection='roulette'):
    best_fitness_per_generation = np.zeros((generations+1, runs))
    np.random.seed(42)
    for run in trange(runs):
        # Random initialization of population
        population = [np.random.randint(2, size=bit_length).tolist() for _ in range(pop_size)]

        # List to record best fitness values per generation
        best_fitness_in_run = []

        for generation in range(generations):
            # Calculate fitness for the current population
            fitness_values = list(map(fitness, population))

            # Record the best fitness of this generation
            best_fitness_in_run.append(max(fitness_values))

            # Create new population (Generational replacement, no elitism)
            new_population = []

            while len(new_population) < pop_size:
                # Parent selection via roulette wheel
                # parent1, parent2 = roulette_wheel_selection(population, fitness_values)
                if selection == 'roulette':
                    parent1, parent2 = roulette_wheel_selection(population, fitness_values)
                elif selection == 'tournament':
                    parent1, parent2 = tournament_selection(population, fitness_values)

                # Recombination (One-point crossover)
                if random.random() < pc:
                    offspring1, offspring2 = one_point_crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1, parent2

                # Add offspring to new population
                new_population.extend([offspring1, offspring2])

            # Update population with new generation
            population = new_population[:pop_size]
        
        best_fitness_in_run.append(max(fitness_values))
        best_fitness_per_generation[:, run] = best_fitness_in_run

    return best_fitness_per_generation

for selection in ['roulette', 'tournament']:
    best_fitness = genetic_algorithm(selection=selection)
    best_fitness**=0.5
    plt.figure(figsize=(10, 6))
    for i in range(RUNS):
        plt.plot(best_fitness[:, i], alpha=0.2, color=mcolors.TABLEAU_COLORS['tab:blue'])
    plt.plot(np.mean(best_fitness, axis=1), color=mcolors.TABLEAU_COLORS['tab:green'], label='Average')
    plt.axhline(y=50, color='r', linestyle='--', label='Optimal Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.yticks(np.arange(min(best_fitness.flatten()), 51, 1))
    plt.legend()
    if selection == 'roulette':
        plt.title(f'Genetic Algorithm - OneMax Problem (50-bit) - Roulette Selection')
    elif selection == 'tournament':
        plt.title(f'Genetic Algorithm - OneMax Problem (50-bit) - Tournament Selection')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'oneMax_GA_{selection}.pdf')
    # plt.show()
