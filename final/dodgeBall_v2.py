import pygame
import numpy as np
import pygame.freetype
from dodgeUtil import totalFitness, PlayerNeuralNetwork, distantScore
# 初始化 pygame
pygame.init()

# 設定視窗大小和顏色
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
FIELD_SIZE = 100
ZOOM = 7
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

# DE parameters
F = 0.9  # Mutation factor
CR = 0.6  # Crossover probability
generations = 100  # Number of generations
pop_size = 60  # Population size
episode_length = 150  # Length of each episode

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
print(nn_population[0].layers)
print(shape_of_weights.shape)

population = np.array([player.get_weights() for player in nn_population])

for gen in range(generations):
    for event in pygame.event.get():
        # force quit
        if event.type == pygame.QUIT:
            running = False
    if not running:
        break
    # Randomize start and target positions
    # agent_positions = np.random.uniform(0, FIELD_SIZE, (pop_size, 2))  # Random starting positions
    agent_position_ = np.random.uniform(0, FIELD_SIZE, 2)  # Single random starting position for all agents
    agent_positions = np.tile(agent_position_, (pop_size, 1))
    agent_vel = np.zeros((pop_size, 2))
    # ball_pos = np.random.uniform(0, FIELD_SIZE, 2)
    ball_speed = np.random.uniform(-BALL_SPEED, BALL_SPEED, 2)
    # target_positions = ball_pos

    accumulated_rewards = np.zeros(pop_size)

    # Run episode
    for step in range(episode_length):
        # 更新球位置
        ball_pos += ball_speed
        ball_speed *= 0.98  # 模擬空氣阻力
        # 碰撞邊界反彈
        if ball_pos[0] <= 0 or ball_pos[0] >= FIELD_SIZE:
            ball_speed[0] *= -1
        if ball_pos[1] <= 0 or ball_pos[1] >= FIELD_SIZE:
            ball_speed[1] *= -1
            
        for i in range(pop_size):
            inputs = np.array([*(ball_pos-agent_positions[i])/FIELD_SIZE])
            agent_vel[i] = nn_population[i].forward(inputs)
            agent_positions[i] += agent_vel[i]
            agent_positions[i] = np.clip(agent_positions[i], 0, FIELD_SIZE)
            fitness = totalFitness(ball_pos, agent_positions[i], agent_vel[i])
            accumulated_rewards[i] += fitness * (step / episode_length)
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
            textSur, rect = font.render(f"{accumulated_rewards[i]:7.2f}", GREEN)
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
        trial_fitness = accumulated_rewards[i]  # Fitness is based on accumulated rewards
        if trial_fitness > accumulated_rewards[i]:
            new_population[i] = trial
        else:
            new_population[i] = population[i]

    population = new_population
    for i in range(pop_size):
        nn_population[i].set_weights(population[i])
    # Logging progress
    best_fitness = max(accumulated_rewards)
    print(f"Generation {gen + 1}, Best Fitness: {best_fitness}")

    # Return the best solution
    # best_idx = np.argmax(accumulated_rewards)
    
# 結束 pygame
pygame.quit()
