import numpy as np
PROXIMITY = 0.05

def directionScore(target_pos, player_pos, player_vel, threshold=0.8):
    # Normalize vectors
    dir_vector = target_pos - player_pos
    normalized_vel = player_vel / np.linalg.norm(player_vel)
    normalized_to_player = dir_vector / np.linalg.norm(dir_vector)
    
    # Calculate cosine similarity
    cosine_similarity = np.dot(normalized_vel, normalized_to_player) 
    
    return cosine_similarity if cosine_similarity > threshold else 0

def totalFitness(target_pos, target_vel, player_pos, player_vel, dist_rank, max_dist):
    # Combine multiple fitness components
    dist_score = -np.linalg.norm(target_pos - player_pos)**2 / max_dist
    # efficiency_score = -np.linalg.norm(player_vel)
    dir_score = directionScore(target_pos, player_pos, player_vel)
    
    # Adjust the weights based on importance
    total_score = sum([
        1.0 * dist_score,
        1.0 * dist_rank,
        1.0 * dir_score,
    ])
    return total_score

def sigmoid(x: np.ndarray) -> np.ndarray:
    # Apply sigmoid activation function
    return 1 / (1 + np.exp(-x))

def sigmoid_centered(x: np.ndarray) -> np.ndarray:
    # Apply sigmoid activation function and center the output at 0
    return 2 / (1 + np.exp(-x)) - 1

class PlayerNeuralNetwork:
    default_architecture = (6, [8, 8], 2)
    def __init__(self, input_size, hidden_sizes, output_size):
        # Initialize neural network layers with random weights and biases
        self.layers = []
        
        layer_sizes = [input_size, *hidden_sizes, output_size]
        for i in range(len(layer_sizes) - 1):
            # Kaiming initialization
            weights = np.random.normal(0, np.sqrt(2 / layer_sizes[i]), (layer_sizes[i], layer_sizes[i + 1]))
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
            
def tournament_selection(population, fitness_values, tournament_size, topk):
    tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
    tournament_fitness = [fitness_values[i] for i in tournament_indices]
    idx = np.argpartition(tournament_fitness, -topk)[-topk:]
    return [population[tournament_indices[j]] for j in idx]
