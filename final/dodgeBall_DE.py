import pygame
import numpy as np
import pygame.freetype
from dodgeUtil import totalFitness, PlayerNeuralNetwork # 初始化 pygame
pygame.init()

# 設定視窗大小和顏色
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 700
FIELD_SIZE = 1
ZOOM = 600
OFFSET = (WINDOW_WIDTH - FIELD_SIZE * ZOOM) // 2
OFFSET_POS = np.ones(2) *  OFFSET
PLAYER_RADIUS = 5
BALL_RADIUS = 15
BALL_SPEED = 0.01
BALL_DRAG = 1

WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Dodgeball Simulation DE")

# 定義顏色
WHITE = (255, 255, 255)
GRAY  = (32, 32, 32)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)
GREEN = (0, 232, 152)
BLUE  = (0, 118, 214)

# DE parameters
F = 0.5  # Mutation factor
E = 0.1  # Elitism factor
CR = 0.8  # Crossover probability
generations = 100  # Number of generations
pop_size = 100  # Population size
episode_length = 120  # Length of each episode
discount = 0.99  # Discount factor
weight_bound = 4.0  # Bound for weights and biases

ball_pos = np.random.uniform(0, FIELD_SIZE, 2)
ball_vel = np.zeros(2)

# 遊戲主迴圈
running = True
frameCount = 0
clock = pygame.time.Clock()
font = pygame.freetype.SysFont('Consolas', 20)

# Initialize population
nn_population = [PlayerNeuralNetwork(*PlayerNeuralNetwork.default_architecture) for _ in range(pop_size)]
shape_of_weights = nn_population[0].get_weights()
for layer in nn_population[0].layers:
    print(layer[0].shape, layer[1].shape)
print(shape_of_weights.shape)

population = np.array([player.get_weights() for player in nn_population])
agent_position_ = np.random.uniform(0, FIELD_SIZE, 2)  # Single random starting position for all agents
accumulated_rewards = np.zeros(pop_size)

shape_of_weights = nn_population[0].get_weights()
for layer in nn_population[0].layers:
    print(layer[0].shape, layer[1].shape)
print(shape_of_weights.shape)

for gen in range(generations):
    for event in pygame.event.get():
        # force quit
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            ball_pos = (np.array(pygame.mouse.get_pos()) - OFFSET_POS) / ZOOM
            
    if not running:
        break
    # Randomize start and target positions
    agent_positions = np.random.uniform(0, FIELD_SIZE, (pop_size, 2))  # Random starting positions
    agent_vel = np.zeros((pop_size, 2))
    # ball_pos = np.random.uniform(0, FIELD_SIZE, 2)
    _ang = np.random.uniform(0, 2*np.pi)
    ball_vel = np.array([np.cos(_ang), np.sin(_ang)]) * BALL_SPEED
    # target_positions = ball_pos
    prev_rewards = np.copy(accumulated_rewards)
    accumulated_rewards = np.zeros(pop_size)
    
    # Run episode
    max_idx = 0
    for step in range(episode_length):
        # 更新球位置
        ball_pos += ball_vel
        ball_vel *= BALL_DRAG
        # 碰撞邊界反彈
        if ball_pos[0] <= 0 or ball_pos[0] >= FIELD_SIZE:
            ball_vel[0] *= -1
        if ball_pos[1] <= 0 or ball_pos[1] >= FIELD_SIZE:
            ball_vel[1] *= -1
        
        
        for i in range(pop_size):
            inputs = np.array([*agent_positions[i]/FIELD_SIZE, *ball_pos/FIELD_SIZE, *ball_vel])
            # if i == max_idx:
            #     print(inputs, end='\r')
            agent_vel[i] = nn_population[i].forward(inputs)
            agent_positions[i] += agent_vel[i]
            # agent_positions[i] = np.clip(agent_positions[i], 0, FIELD_SIZE)
            fitness = totalFitness(ball_pos, ball_vel, agent_positions[i], agent_vel[i])
            # biased towards the end of the episode
            accumulated_rewards[i] += fitness * discount ** (episode_length - step)
        max_idx = np.argmax(accumulated_rewards)
        
        # 清空畫面
        WINDOW.fill(GRAY)
        
        # 繪製躲球方（藍色）
        for i, (player, vel) in enumerate(zip(agent_positions, agent_vel)):
            if i == max_idx:
                pygame.draw.circle(WINDOW, GREEN, player * ZOOM + OFFSET_POS, PLAYER_RADIUS+2)
            else:
                pygame.draw.circle(WINDOW, BLUE,  player * ZOOM + OFFSET_POS, PLAYER_RADIUS)
            pygame.draw.line(WINDOW, WHITE, player * ZOOM + OFFSET_POS, (player + vel * BALL_RADIUS) * ZOOM + OFFSET_POS, 1)
            textSur, rect = font.render(f"[{i:2d}]", GREEN)
            WINDOW.blit(textSur, player * ZOOM + OFFSET_POS)
            
        # 繪製球（黑色）
        pygame.draw.circle(WINDOW, RED, ball_pos * ZOOM + OFFSET_POS, BALL_RADIUS)
        
        textSur, rect = font.render(f"Frame: {frameCount}, Episode: {frameCount//episode_length}", GREEN)
        WINDOW.blit(textSur, OFFSET_POS/2)
        # 繪製場地邊界
        pygame.draw.rect(WINDOW, GREEN, (OFFSET, OFFSET, FIELD_SIZE * ZOOM, FIELD_SIZE * ZOOM), 3)

        # 更新顯示
        pygame.display.flip()
        
        # 控制更新速度
        clock.tick(60)
        frameCount += 1
        
    # Genetic operations
    new_population = np.zeros_like(population)
    for i in range(pop_size):
        # Mutation: Select three random individuals
        idxs = np.random.choice([x for x in range(pop_size) if x != i], 3, replace=False)
        x1, x2, x3 = population[idxs]
        mutant = x1 + F * (x2 - x3)

        # Crossover
        trial = np.copy(population[i])
        mutate_indices = np.random.uniform(size=shape_of_weights.shape) < CR
        trial[mutate_indices] = mutant[mutate_indices]

        # Selection
        # overwrite if trial_fitness is better than previous
        if accumulated_rewards[i] >= prev_rewards[i]:
            new_population[i] = trial
        else:
            new_population[i] = population[i]
        # Bound the weights
        new_population[i] = np.clip(new_population[i], -weight_bound, weight_bound)
        
    population = new_population
    for i in range(pop_size):
        nn_population[i].set_weights(population[i])
    # Logging progress
    best_fitness = max(accumulated_rewards)
    mean_fitness = np.mean(accumulated_rewards)
    diversity = np.std(accumulated_rewards)
    print(f"Generation {gen + 1}, Best Fitness: {best_fitness:9.2f}, Mean Fitness: {mean_fitness:9.2f}, Diversity: {diversity:9.2f}")

    
# 結束 pygame
pygame.quit()

print("Best solution: ", nn_population[max_idx].layers)