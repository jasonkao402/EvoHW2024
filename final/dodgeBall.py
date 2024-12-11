import pygame
import random
import numpy as np
import pygame.freetype
from dodgeUtil import totalFitness, PlayerNeuralNetwork, tournament_selection, one_point_crossover, one_point_mutation, two_point_crossover, distantScore
# 初始化 pygame
pygame.init()

# 設定視窗大小和顏色
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600
FIELD_SIZE = 500
OFFSET = (WINDOW_WIDTH - FIELD_SIZE) // 2
OFFSET_POS = np.ones(2) *  OFFSET
WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Dodgeball Simulation")

# 定義顏色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# 定義參數
PLAYER_RADIUS = 10
BALL_RADIUS = 15
NUM_PLAYERS = 40
OFFSPRING = 30
BALL_SPEED = 16
EPISODE_LENGTH = 10

# 初始化球員位置
red_team = []
for i in range(NUM_PLAYERS):
    # 均勻分佈在場外（場地左、右、上三邊）
    x = np.random.uniform(0, 1)
    y = np.random.uniform(0, 1)
    if i % 2 == 0:
        if x < 0.5:
            x = 0
        else:
            x = FIELD_SIZE
        y *= FIELD_SIZE
    else:
        if y < 0.5:
            y = 0
        else:
            y = FIELD_SIZE
        x *= FIELD_SIZE
    red_team.append((x, y))
red_team = np.array(red_team)
# blue_team = np.random.uniform(50, FIELD_SIZE - 50, (NUM_PLAYERS, 2))

# 初始化球位置
ball_pos = np.array(red_team[0][:])
ball_speed = np.zeros(2)

def find_nearby(center_pos, teammates, radius=200, amount=3):
    result = np.zeros((amount, 2))
    dists = np.linalg.norm(teammates - center_pos, axis=1)
    # ignore self (dist < EPS) and dist > radius
    dists[dists < 1e-6] = float('inf')
    dists[dists > radius] = float('inf')
    idx = np.argsort(dists)
    for i in range(amount):
        result[i] = teammates[idx[i]]
    return result[:amount]

# 計算向量單位向量
def calculate_direction(source, target):
    direction = np.array(target) - np.array(source)
    return direction / np.linalg.norm(direction)

def calculate_distance(source, target):
    return np.linalg.norm(np.array(target) - np.array(source))

# 遊戲主迴圈
running = True
frameCount = 0
clock = pygame.time.Clock()
font = pygame.freetype.SysFont('Arial', 24)
# Example of evolving the weights for a single player's movement
generations = 50
pc, pm = 0.9, 0.1  # Crossover and mutation probabilities
# Initialize population
population = [PlayerNeuralNetwork(*PlayerNeuralNetwork.default_architecture) for _ in range(NUM_PLAYERS)]
print(population[0].layers)
print(population[0].get_weights().shape)

for player in population:
    player.position = np.random.uniform(50, FIELD_SIZE - 50, 2)
    player.velocity = np.random.uniform(-1, 1, 2)
    player.ball_pos = np.zeros(2)
    player.ball_vel = np.zeros(2)
max_idx = 0
# def nearest
accumulated_fitness = np.zeros(NUM_PLAYERS)
while running:
    blue_team = np.array([player.position for player in population])
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # 點擊滑鼠以丟球
            target = find_nearby(np.array(pygame.mouse.get_pos()) - OFFSET_POS, blue_team, amount=1)[0]
            # print(target)
            if np.linalg.norm(ball_pos - target) > 0:
                direction = calculate_direction(ball_pos, target)
                ball_speed = BALL_SPEED * direction

    # 更新球位置
    ball_pos += ball_speed
    ball_speed *= 0.99  # 模擬空氣阻力
    # 碰撞邊界反彈
    if ball_pos[0] <= 0 or ball_pos[0] >= FIELD_SIZE:
        ball_speed[0] *= -1
    if ball_pos[1] <= 0 or ball_pos[1] >= FIELD_SIZE:
        ball_speed[1] *= -1

    # t = 
    # print(t.shape)
    accumulated_fitness += np.array([-distantScore(ball_pos, player) for player in blue_team])
        
    if frameCount % EPISODE_LENGTH == 0:
        best_idx = np.argmax(accumulated_fitness)
        print(population[best_idx].layers)
        for i, player in enumerate(population):
            player.fitness = accumulated_fitness[i]
            
        new_population = []
        while len(new_population) < OFFSPRING:
            parent1, parent2 = tournament_selection(population, accumulated_fitness, tournament_size=5)
            
            if random.random() < pc:
                offspring1, offspring2 = two_point_crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1, parent2
            # if random.random() < pm:
            offspring1 = one_point_mutation(offspring1, pm)
            offspring2 = one_point_mutation(offspring2, pm)
            new_population.extend([offspring1, offspring2])
        
        # overwrite the worst half of the population
        population = sorted(population, key=lambda x: x.fitness, reverse=True)[:-OFFSPRING] + new_population
        accumulated_fitness = np.zeros(NUM_PLAYERS)
        
        # reset the position and velocity of the population
        for player in population:
            ball_pos = np.array(random.choice(red_team))
            # deg = np.random.uniform(0, 2*np.pi)
            # player.position = np.array([FIELD_SIZE/2 + np.cos(deg)*FIELD_SIZE/2, FIELD_SIZE/2 + np.sin(deg)*FIELD_SIZE/2])
            player.position = np.random.uniform(50, FIELD_SIZE - 50, 2)
            player.velocity = np.random.uniform(-1, 1, 2)
            player.ball_pos = np.zeros(2)
            player.ball_vel = np.zeros(2)
        
    # 繪製丟球方（紅色）
    for player in red_team:
        pygame.draw.circle(WINDOW, RED, player + OFFSET_POS, PLAYER_RADIUS)
    
    # 繪製躲球方（藍色）
    for i, player in enumerate(blue_team):
        
        if i == max_idx:
            pygame.draw.circle(WINDOW, GREEN, player + OFFSET_POS, PLAYER_RADIUS+2)
        else:
            pygame.draw.circle(WINDOW, BLUE,  player + OFFSET_POS, PLAYER_RADIUS)
        textSur, rect = font.render(f"{accumulated_fitness[i]:7.2f}", BLACK)
        WINDOW.blit(textSur, player + OFFSET_POS)

    # 繪製球（黑色）
    pygame.draw.circle(WINDOW, BLACK, ball_pos + OFFSET_POS, BALL_RADIUS)
        
    for i, player in enumerate(population):
        
        # nearest = find_nearby(player, blue_team, amount=1)[0] - player
        # pygame.draw.line(WINDOW, GREEN, player + OFFSET_POS, player + OFFSET_POS + nearest, 2)
        input_vec = np.array([player.position[0]/FIELD_SIZE, player.position[1]/FIELD_SIZE, ball_pos[0]/FIELD_SIZE, ball_pos[1]/FIELD_SIZE])
        player.velocity = population[i].forward(input_vec)
        pygame.draw.line(WINDOW, BLACK, player.position + OFFSET_POS, player.position + OFFSET_POS + player.velocity * PLAYER_RADIUS, 2)
        player.position += player.velocity
        player.position = np.clip(player.position, [PLAYER_RADIUS, PLAYER_RADIUS], [FIELD_SIZE - PLAYER_RADIUS, FIELD_SIZE - PLAYER_RADIUS])
        # if np.linalg.norm(ball_pos - player.position) < PLAYER_RADIUS + BALL_RADIUS:
        #     # 球回到紅隊成員位置
        #     ball_pos = np.array(random.choice(red_team))
        #     ball_speed = np.zeros(2)  # 球速歸零
        #     # break
    
    textSur, rect = font.render(f"Frame: {frameCount}, Episode: {frameCount//EPISODE_LENGTH}", BLACK)
    WINDOW.blit(textSur, OFFSET_POS/2)
    # 繪製場地邊界
    pygame.draw.rect(WINDOW, GREEN, (OFFSET, OFFSET, FIELD_SIZE, FIELD_SIZE), 3)

    # 更新顯示
    pygame.display.flip()
    
    # 控制更新速度
    clock.tick(40)
    frameCount += 1
    # 清空畫面
    WINDOW.fill(WHITE)
# 結束 pygame
pygame.quit()
