import numpy as np

def attackScore(ball_pos, blue_team):
    score = 0
    for player in blue_team:
        dist = np.linalg.norm(ball_pos - player)
        score += 100 / (dist+1)
    return score

def dodgeScore(ball_pos, player_pos):
    score = np.linalg.norm(ball_pos - player_pos) ** 2
    # score = 0
    # for player in blue_team:
        # dist = np.linalg.norm(ball_pos - player)
        # score += dist**2
    return score

def teamSpacingScore(blue_team, min_distance=10):
    # Encourage players to maintain a minimum distance from each other
    spacing_score = 0
    num_players = len(blue_team)
    
    for i in range(num_players):
        for j in range(i + 1, num_players):
            dist = np.linalg.norm(blue_team[i] - blue_team[j])
            # Penalize if distance is less than a threshold
            if dist < min_distance:
                spacing_score -= (min_distance - dist)**2
            else:
                spacing_score += dist**0.5  # small reward for keeping good distance
    
    return spacing_score

def boundaryAvoidanceScore(blue_team, field_size):
    # Encourage players to stay away from the boundaries of the field
    boundary_penalty = 0
    for player in blue_team:
        x, y = player
        dist_to_boundary = min(x, field_size - x, y, field_size - y)
        if dist_to_boundary < 10:  # Assume a buffer zone of 10 units
            boundary_penalty -= (10 - dist_to_boundary)**2
    return -boundary_penalty  # Negative penalty means it's subtracted from fitness

def movementEfficiencyScore(previous_positions, current_positions):
    # Minimize excessive movement; reward for efficient dodging
    efficiency_score = 0
    for prev, curr in zip(previous_positions, current_positions):
        movement_distance = np.linalg.norm(curr - prev)
        efficiency_score -= movement_distance**2  # Penalize unnecessary movement
    return efficiency_score

def threatProximityScore(ball_pos, ball_velocity, blue_team):
    # Penalize players in the direct threat path of the ball
    penalty_score = 0
    for player in blue_team:
        # Vector from ball to player
        to_player_vector = player - ball_pos
        # Normalize vectors
        normalized_ball_velocity = ball_velocity / np.linalg.norm(ball_velocity)
        normalized_to_player = to_player_vector / np.linalg.norm(to_player_vector)
        
        # Calculate cosine similarity
        cosine_similarity = np.dot(normalized_ball_velocity, normalized_to_player)
        
        # If player is in the direct threat path (cosine similarity close to 1)
        if cosine_similarity > 0.9:  # Threshold can be adjusted
            penalty_score -= 100  # Heavier penalty for being in the path
    
    return penalty_score

def totalFitness(ball_pos, ball_velocity, blue_team, previous_positions, field_size):
    # Combine multiple fitness components
    dodge_score = dodgeScore(ball_pos, blue_team)
    spacing_score = teamSpacingScore(blue_team)
    boundary_score = boundaryAvoidanceScore(blue_team, field_size)
    efficiency_score = movementEfficiencyScore(previous_positions, blue_team)
    threat_penalty = threatProximityScore(ball_pos, ball_velocity, blue_team)
    
    # Adjust the weights based on importance
    total_score = (
        1.0 * dodge_score + 
        0.5 * spacing_score + 
        0.3 * boundary_score + 
        0.2 * efficiency_score + 
        1.5 * threat_penalty
    )
    
    return total_score

def elu6(x: np.ndarray) -> np.ndarray:
    # Apply ELU activation function and cap the output at 6
    return np.clip(np.where(x > 0, x, np.exp(x) - 1), None, 6)

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        # Initialize neural network layers with random weights and biases
        self.layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            weights = np.random.randn(layer_sizes[i], layer_sizes[i + 1])
            biases = np.random.randn(layer_sizes[i + 1])
            self.layers.append((weights, biases))
    
    def forward(self, x):
        # Forward pass through the neural network
        for weights, biases in self.layers:
            x = np.dot(x, weights) + biases
            x = elu6(x)
        return x

    def get_weights(self):
        # Flatten the weights and biases into a chromosome
        return np.concatenate([weights.flatten() for weights, _ in self.layers] + 
                              [biases.flatten() for _, biases in self.layers])
    
    def set_weights(self, chromosome):
        # Set the weights and biases from a chromosome
        idx = 0
        for i in range(len(self.layers)):
            weights, biases = self.layers[i]
            weight_size = weights.size
            self.layers[i] = (
                chromosome[idx:idx + weight_size].reshape(weights.shape),
                chromosome[idx + weight_size:idx + weight_size + biases.size]
            )
            idx += weight_size + biases.size
            
def tournament_selection(population, fitness_values, tournament_size=3):
    tournament_indices = np.random.choice(len(population), size=tournament_size*2, replace=False)
    tournament_fitness = [fitness_values[i] for i in tournament_indices]
    idx = np.argmax(tournament_fitness[:tournament_size]), np.argmax(tournament_fitness[tournament_size:])
    return population[tournament_indices[idx[0]]], population[tournament_indices[idx[1]]]

def one_point_crossover(parent1, parent2):
    chromosome1 = parent1.get_weights()
    chromosome2 = parent2.get_weights()
    point = np.random.randint(1, len(chromosome1) - 1)
    offspring1 = NeuralNetwork(6, [10, 10], 2).set_weights(np.concatenate([chromosome1[:point], chromosome2[point:]]))
    offspring2 = NeuralNetwork(6, [10, 10], 2).set_weights(np.concatenate([chromosome2[:point], chromosome1[point:]]))
    return offspring1, offspring2

def one_point_mutation(chromosome, mutation_rate, std_dev=0.1):
    mutated_chromosome = chromosome.copy()
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            mutated_chromosome[i] += np.random.normal(0, std_dev)
    return mutated_chromosome