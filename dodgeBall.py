import pygame
import random
import math
import numpy as np
# 初始化 pygame
pygame.init()

# 設定視窗大小和顏色
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600
FIELD_SIZE = 500
OFFSET = (WINDOW_WIDTH - FIELD_SIZE) // 2
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
BALL_RADIUS = 5
NUM_PLAYERS = 6
BALL_SPEED = 20

# 初始化球員位置
red_team = []
for i in range(NUM_PLAYERS):
    # 均勻分佈在場外（場地左、右、上三邊）
    x = np.random.uniform(-FIELD_SIZE, FIELD_SIZE)
    y = np.random.uniform(-FIELD_SIZE, FIELD_SIZE)
    if i % 2 == 0:
        if x < 0:
            x = 0
        else:
            x = FIELD_SIZE
    else:
        if y < 0:
            y = 0
        else:
            y = FIELD_SIZE
    red_team.append((x, y))
red_team = np.array(red_team)
blue_team = np.random.randint(50, FIELD_SIZE - 50, (NUM_PLAYERS, 2))

# 初始化球位置
ball_pos = np.array(red_team[0][:])
ball_speed = np.zeros(2)

def find_closest_blue(mouse_pos, blue_team):
    closest_player = None
    min_distance = float('inf')
    for player in blue_team:
        dist = np.linalg.norm(np.array(mouse_pos) - player)
        if dist < min_distance:
            min_distance = dist
            closest_player = player
    return closest_player

# 計算向量單位向量
def calculate_direction(source, target):
    direction = np.array(target) - np.array(source)
    return direction / np.linalg.norm(direction)

# 遊戲主迴圈
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # 點擊滑鼠以丟球
            target = find_closest_blue(pygame.mouse.get_pos(), blue_team)
            print(target)
            if target:
                direction = calculate_direction(ball_pos, target)
                ball_speed = BALL_SPEED * direction

    # 更新球位置
    ball_pos += ball_speed
    

    # 碰撞檢測（檢查球是否碰到藍隊成員）
    for player in blue_team:
        if np.linalg.norm(ball_pos - player) < PLAYER_RADIUS + BALL_RADIUS:
            # 球回到紅隊成員位置
            ball_pos = random.choice(red_team)
            ball_speed = np.zeros(2)  # 球速歸零
            break

    # 清空畫面
    WINDOW.fill(WHITE)

    # 繪製場地邊界
    pygame.draw.rect(WINDOW, GREEN, (OFFSET, OFFSET, FIELD_SIZE, FIELD_SIZE), 3)

    # 繪製丟球方（紅色）
    for player in red_team:
        pygame.draw.circle(WINDOW, RED, player, PLAYER_RADIUS)

    # 繪製躲球方（藍色）
    for player in blue_team:
        pygame.draw.circle(WINDOW, BLUE, player, PLAYER_RADIUS)

    # 繪製球（黑色）
    pygame.draw.circle(WINDOW, BLACK, (int(ball_pos[0]), int(ball_pos[1])), BALL_RADIUS)

    # 更新顯示
    pygame.display.flip()

    # 控制更新速度
    clock.tick(45)

# 結束 pygame
pygame.quit()
