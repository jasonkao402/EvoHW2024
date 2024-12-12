import pygame
import numpy as np
import pygame.freetype
from dodgeUtil import totalFitness, PlayerNeuralNetwork, distantScore, tournament_selection
# 初始化 pygame
pygame.init()

# 設定視窗大小和顏色
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 700
FIELD_SIZE = 100
ZOOM = 6
OFFSET = (WINDOW_WIDTH - FIELD_SIZE * ZOOM) // 2
OFFSET_POS = np.ones(2) *  OFFSET
WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Dodgeball Simulation")

# 定義顏色
WHITE = (255, 255, 255)
GRAY  = (32, 32, 32)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)
GREEN = (0, 232, 152)
BLUE  = (0, 118, 214)

# 定義參數
PLAYER_RADIUS = 10
BALL_RADIUS = 15
BALL_SPEED = 0.5
BALL_DRAG = 0.98

generations = 100  # Number of generations
pop_size = 80  # Population size
episode_length = 120  # Length of each episode
discount = 0.9  # Discount factor
crossover_rate = 0.8  # Crossover probability
mutate_rate = 0.1  # Mutation rate

ball_pos = np.random.uniform(0, FIELD_SIZE, 2)
ball_speed = np.zeros(2)

# 遊戲主迴圈
running = True
frameCount = 0
clock = pygame.time.Clock()
font = pygame.freetype.SysFont('Arial', 24)

# Initialize population
nn_population = [PlayerNeuralNetwork(*PlayerNeuralNetwork.default_architecture) for _ in range(pop_size)]
shape_of_weights = nn_population[0].get_weights()
for layer in nn_population[0].layers:
    print(layer[0].shape, layer[1].shape)
print(shape_of_weights.shape)

population = np.array([player.get_weights() for player in nn_population])
agent_position_ = np.random.uniform(0, FIELD_SIZE, 2)  # Single random starting position for all agents
accumulated_rewards = np.zeros(pop_size)

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
    # agent_positions = np.random.uniform(0, FIELD_SIZE, (pop_size, 2))  # Random starting positions
    agent_position_ = np.array([FIELD_SIZE, FIELD_SIZE]) - ball_pos
    agent_positions = np.tile(agent_position_, (pop_size, 1))
    agent_vel = np.zeros((pop_size, 2))
    # ball_pos = np.random.uniform(0, FIELD_SIZE, 2)
    ball_speed = np.random.uniform(-BALL_SPEED, BALL_SPEED, 2)
    # target_positions = ball_pos
    prev_rewards = np.copy(accumulated_rewards)
    accumulated_rewards = np.zeros(pop_size)
    
    # Run episode
    max_idx = 0
    for step in range(episode_length):
        # 更新球位置
        ball_pos += ball_speed
        ball_speed *= BALL_DRAG
        # 碰撞邊界反彈
        if ball_pos[0] <= BALL_RADIUS or ball_pos[0] >= FIELD_SIZE - BALL_RADIUS:
            ball_speed[0] *= -1
        if ball_pos[1] <= BALL_RADIUS or ball_pos[1] >= FIELD_SIZE - BALL_RADIUS:
            ball_speed[1] *= -1
        
        for i in range(pop_size):
            inputs = np.array([*agent_positions[i]/FIELD_SIZE, *ball_pos/FIELD_SIZE])
            # if i == max_idx:
            #     print(inputs, end='\r')
            agent_vel[i] = nn_population[i].forward(inputs)
            agent_positions[i] += agent_vel[i]
            # agent_positions[i] = np.clip(agent_positions[i], 0, FIELD_SIZE)
            fitness = totalFitness(ball_pos, agent_positions[i], agent_vel[i])
            # biased towards the end of the episode
            accumulated_rewards[i] += fitness * discount ** (episode_length - step)
        max_idx = np.argmax(accumulated_rewards)
        
        # 清空畫面
        WINDOW.fill(GRAY)
        # 繪製丟球方（紅色）
        # for player in red_team:
        #     pygame.draw.circle(WINDOW, RED, player * ZOOM + OFFSET_POS, PLAYER_RADIUS)
        
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
        # Select parents
        parent1, parent2 = tournament_selection(population, accumulated_rewards, tournament_size=5)
        # Crossover - single point
        if np.random.rand() < crossover_rate:
            crossover_point = np.random.randint(1, shape_of_weights.size)
            new_population[i] = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        else:
            new_population[i] = parent1
        # Mutation
        if np.random.rand() < mutate_rate:
            mutate_idx = np.random.randint(0, shape_of_weights.size)
            new_population[i][mutate_idx] += np.random.randn() * 0.1
        # Bound the weights
        new_population[i] = np.clip(new_population[i], -10, 10)
    population = new_population
    
    for i in range(pop_size):
        nn_population[i].set_weights(population[i])
    # Logging progress
    best_fitness = max(accumulated_rewards)
    diversity = np.std(accumulated_rewards)
    print(f"Generation {gen + 1}, Best Fitness: {best_fitness:9.2f}, Diversity: {diversity:9.2f}")

    # Return the best solution
    # best_idx = np.argmax(accumulated_rewards)
    
# 結束 pygame
pygame.quit()

print("Best solution: ", nn_population[max_idx].layers)