import numpy as np
FIELD_SIZE = 1
PROXIMITY = 0.05
CENTER = np.array([FIELD_SIZE, FIELD_SIZE]) / 2

def directionScore(target_pos, player_pos, player_vel):
    dir_score = 0
    # Normalize vectors
    dir_vector = target_pos - player_pos
    normalized_vel = player_vel / np.linalg.norm(player_vel)
    normalized_to_player = dir_vector / np.linalg.norm(dir_vector)
    
    # Calculate cosine similarity
    cosine_similarity = np.dot(normalized_vel, normalized_to_player) 
    
    if cosine_similarity > 0.7:  # Threshold can be adjusted
        dir_score += cosine_similarity
    
    return dir_score

def totalFitness(target_pos, target_vel, player_pos, player_vel):
    # Combine multiple fitness components
    dist_score = -np.sum(np.abs(target_pos - player_pos))
    efficiency_score = -np.linalg.norm(player_vel)
    dir_score = directionScore(target_pos, player_pos, player_vel)
    # touch_dist = int(np.linalg.norm(target_pos - player_pos) < PROXIMITY)
    
    # Adjust the weights based on importance
    total_score = sum([
        4.0 * dist_score,
        # 1.0 * center_score,
        1.0 * efficiency_score,
        5.0 * dir_score,
        # 100 * touch_dist,
    ])
    return total_score

def sigmoid(x: np.ndarray) -> np.ndarray:
    # Apply sigmoid activation function
    return 1 / (1 + np.exp(-x))

def sigmoid_centered(x: np.ndarray) -> np.ndarray:
    # Apply sigmoid activation function and center the output at 0
    return 2 / (1 + np.exp(-x)) - 1

class PlayerNeuralNetwork:
    default_architecture = (6, [6, 4], 2)
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
    tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
    tournament_fitness = [-fitness_values[i] for i in tournament_indices]
    idx = np.argsort(tournament_fitness)[:2]
    return population[tournament_indices[idx[0]]], population[tournament_indices[idx[1]]]
