import pygame
import random
import numpy as np
import pygame.freetype
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
BALL_RADIUS = 5
NUM_PLAYERS = 10
BALL_SPEED = 16

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
blue_team = np.random.randint(50, FIELD_SIZE - 50, (NUM_PLAYERS, 2))

# 初始化球位置
ball_pos = np.array(red_team[0][:])
ball_speed = np.zeros(2)

def find_closest_blue(mouse_pos, blue_team):
    closest_player = np.zeros(2)
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

def calculate_distance(source, target):
    return np.linalg.norm(np.array(target) - np.array(source))

# 遊戲主迴圈
running = True
clock = pygame.time.Clock()
font = pygame.freetype.SysFont(None, 20)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # 點擊滑鼠以丟球
            target = find_closest_blue(np.array(pygame.mouse.get_pos())-OFFSET_POS, blue_team)
            # print(target)
            if np.linalg.norm(ball_pos - target) > 0:
                direction = calculate_direction(ball_pos, target)
                ball_speed = BALL_SPEED * direction

    # 更新球位置
    ball_pos += ball_speed
    ball_speed *= 0.98  # 模擬空氣阻力
    

    # 碰撞檢測（檢查球是否碰到藍隊成員）
    for player in blue_team:
        if np.linalg.norm(ball_pos - player) < PLAYER_RADIUS + BALL_RADIUS:
            # 球回到紅隊成員位置
            ball_pos = np.array(random.choice(red_team))
            ball_speed = np.zeros(2)  # 球速歸零
            break

    # 清空畫面
    WINDOW.fill(WHITE)
    
    # show score text in-game
    score_a = attackScore(ball_pos, blue_team)
    score_d = dodgeScore(ball_pos, blue_team)
    
    textSur, rect = font.render(f"ATK Score : {score_a:.2f}  DODGE Score : {score_d:.2f}", BLUE, WHITE)
    WINDOW.blit(textSur, OFFSET_POS)
    # 繪製場地邊界
    pygame.draw.rect(WINDOW, GREEN, (OFFSET, OFFSET, FIELD_SIZE, FIELD_SIZE), 3)

    # 繪製丟球方（紅色）
    for player in red_team:
        pygame.draw.circle(WINDOW, RED, player+OFFSET_POS, PLAYER_RADIUS)

    # 繪製躲球方（藍色）
    for player in blue_team:
        pygame.draw.circle(WINDOW, BLUE, player+OFFSET_POS, PLAYER_RADIUS)

    # 繪製球（黑色）
    pygame.draw.circle(WINDOW, BLACK, ball_pos+OFFSET_POS, BALL_RADIUS)

    # 更新顯示
    pygame.display.flip()
    
    # 控制更新速度
    clock.tick(50)

# 結束 pygame
pygame.quit()
