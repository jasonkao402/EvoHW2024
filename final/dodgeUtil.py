import numpy as np
FIELD_SIZE = 100

def distantScore(target_pos, player_pos):
    dist = np.linalg.norm(target_pos - player_pos)
    return dist

def boundaryAvoidanceScore(player_pos, margin=5):
    # Encourage players to stay away from the boundaries of the field
    boundary_penalty = 0
    x, y = player_pos
    dist_to_boundary = min(x, FIELD_SIZE - x, y, FIELD_SIZE - y)
    if dist_to_boundary < margin:  # Assume a buffer zone of 10 units
        # boundary_penalty -= (margin - dist_to_boundary)**2
        boundary_penalty -= dist_to_boundary**2
    return boundary_penalty  # Negative penalty means it's subtracted from fitness

def movementEfficiencyScore(player_vel):
    # Minimize excessive movement; reward for efficient dodging
    return -np.linalg.norm(player_vel)

def threatProximityScore(ball_pos, ball_velocity, player_pos):
    # Penalize players in the direct threat path of the ball
    penalty_score = 0
    to_player_vector = player_pos - ball_pos
    # Normalize vectors
    normalized_ball_velocity = ball_velocity / np.linalg.norm(ball_velocity)
    normalized_to_player = to_player_vector / np.linalg.norm(to_player_vector)
    
    # Calculate cosine similarity
    cosine_similarity = np.dot(normalized_ball_velocity, normalized_to_player)
    
    # If player is in the direct threat path (cosine similarity close to 1)
    if cosine_similarity > 0.9:  # Threshold can be adjusted
        penalty_score -= 100  # Heavier penalty for being in the path
    
    return penalty_score

def totalFitness(target_pos, player_pos, player_vel):
    # Combine multiple fitness components
    dist_score = -distantScore(target_pos, player_pos)
    # dodge_score = dodgeScore(ball_pos, player_pos)
    # spacing_score = teamSpacingScore(team_pos)
    boundary_score = boundaryAvoidanceScore(player_pos)
    efficiency_score = movementEfficiencyScore(player_vel)
    # threat_penalty = threatProximityScore(ball_pos, ball_velocity, player_pos)
    
    # Adjust the weights based on importance
    total_score = sum([
        5.0 * dist_score,
        # 3.0 * dodge_score,
        # 1.0 * spacing_score,
        1.0 * boundary_score,
        1.0 * efficiency_score,
        # 1.0 * threat_penalty
    ])
    
    return total_score

def elu6(x: np.ndarray) -> np.ndarray:
    # Apply ELU activation function and cap the output at 6
    return np.clip(np.where(x > 0, x, np.exp(x) - 1), None, 6)

def ReLU6(x: np.ndarray) -> np.ndarray:
    # Apply ReLU activation function
    return np.clip(x, -6, 6)

def sigmoid(x: np.ndarray) -> np.ndarray:
    # Apply sigmoid activation function
    return 1 / (1 + np.exp(-x))
class PlayerNeuralNetwork:
    default_architecture = (4, [6, 6], 2)
    def __init__(self, input_size, hidden_sizes, output_size):
        # Initialize neural network layers with random weights and biases
        self.layers = []
        
        layer_sizes = [input_size, *hidden_sizes, output_size]
        for i in range(len(layer_sizes) - 1):
            weights = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(1 / layer_sizes[i]) # Xavier initialization
            biases = np.random.randn(layer_sizes[i + 1])
            self.layers.append((weights, biases))
    
    def forward(self, x):
        # Forward pass through the neural network
        for weights, biases in self.layers:
            x = np.dot(x, weights) + biases
            x = np.tanh(x)
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
