import numpy as np
import matplotlib.pyplot as plt
import random

# Fitness function for the OneMax problem
def fitness(individual):
    return sum(individual)

# Roulette wheel selection
def roulette_wheel_selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    selection_probs = [f / total_fitness for f in fitness_values]
    return population[np.random.choice(len(population), p=selection_probs)]

# tournament selection
def tournament_selection(population, fitness_values, tournament_size=2):
    tournament_indices = np.random.choice(len(population), size=tournament_size)
    tournament_fitness = [fitness_values[i] for i in tournament_indices]
    return population[tournament_indices[np.argmax(tournament_fitness)]]

# One-point crossover
def one_point_crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    offspring1 = parent1[:point] + parent2[point:]
    offspring2 = parent2[:point] + parent1[point:]
    return offspring1, offspring2

# Genetic Algorithm function
def genetic_algorithm(pop_size=200, bit_length=50, generations=100, runs=10, pc=1.0):
    best_fitness_per_generation = np.zeros(generations)

    for run in range(runs):
        # Random initialization of population
        population = [np.random.randint(2, size=bit_length).tolist() for _ in range(pop_size)]

        # List to record best fitness values per generation
        best_fitness_in_run = []

        for generation in range(generations):
            # Calculate fitness for the current population
            fitness_values = [fitness(ind) for ind in population]

            # Record the best fitness of this generation
            best_fitness_in_run.append(max(fitness_values))

            # Create new population (Generational replacement, no elitism)
            new_population = []

            while len(new_population) < pop_size:
                # Parent selection via roulette wheel
                parent1 = roulette_wheel_selection(population, fitness_values)
                parent2 = roulette_wheel_selection(population, fitness_values)

                # Recombination (One-point crossover)
                if random.random() < pc:
                    offspring1, offspring2 = one_point_crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1, parent2

                # Add offspring to new population
                new_population.extend([offspring1, offspring2])

            # Update population with new generation
            population = new_population[:pop_size]

        # Add best fitness values of the current run to the total
        best_fitness_per_generation += np.array(best_fitness_in_run)

    # Average over the 10 runs
    best_fitness_per_generation /= runs

    return best_fitness_per_generation

# Running the GA and plotting results
best_fitness = genetic_algorithm()

# Plotting the best fitness over generations
plt.plot(best_fitness)
plt.xlabel('Generation')
plt.ylabel('Average Best Fitness (10 Runs)')
plt.title('Genetic Algorithm - OneMax Problem (50-bit)')
plt.grid(True)
plt.show()
