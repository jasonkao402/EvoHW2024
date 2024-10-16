import numpy as np
import random
import matplotlib.pyplot as plt
import heapq
import itertools
from tqdm import trange, tqdm
# 設定隨機種子以便於重現結果
N = 100
np.random.seed(42)
plt.figure(figsize=(17, 8))
# 隨機生成城市位置
points = np.random.rand(N, 2)*10
# points = np.array([np.cos(np.linspace(0, 2*np.pi, N)), np.sin(np.linspace(0, 2*np.pi, N))]).T
def prim_mst(distance_matrix):
    visited = [False] * N
    min_edge = [(0, 0)]  # (邊的權重，目標節點)
    total_weight = 0
    
    while len(min_edge) > 0:
        weight, node = heapq.heappop(min_edge)
        if visited[node]:
            continue
        visited[node] = True
        total_weight += weight
        
        for neighbor in range(N):
            if not visited[neighbor]:
                heapq.heappush(min_edge, (distance_matrix[node][neighbor], neighbor))
    
    return total_weight

# 計算 TSP 問題的下界
def tsp_lower_bound(distance_matrix):
    mst_weight = prim_mst(distance_matrix)  # 計算最小生成樹的權重

    # 找出與城市0相關的兩個最小邊，將其加到最小生成樹的權重中
    first_min_edge = float('inf')
    second_min_edge = float('inf')
    
    for i in range(1, N):
        if distance_matrix[0][i] < first_min_edge:
            second_min_edge = first_min_edge
            first_min_edge = distance_matrix[0][i]
        elif distance_matrix[0][i] < second_min_edge:
            second_min_edge = distance_matrix[0][i]
    
    lower_bound = mst_weight + first_min_edge + second_min_edge
    return lower_bound

def generate_distance_matrix():
    matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            distance = np.linalg.norm(points[i] - points[j])
            matrix[i, j] = distance
            matrix[j, i] = distance
    return matrix

# 計算一條路徑的總距離
def calculate_total_distance(route, distance_matrix) -> float:
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance_matrix[route[i], route[i+1]]
    total_distance += distance_matrix[route[-1], route[0]]  # 回到起點的距離
    return total_distance

# 初始化種群，生成多條隨機路徑
def init_population(pop_size, n_cities):
    population = [np.random.permutation(n_cities) for _ in range(pop_size)]
    return population

# 選擇操作，基於適應度
def selection(population: list[np.ndarray], distance_matrix, keepPercent=0.1) -> list[np.ndarray]:
    fitness_scores = []
    for route in population:
        fitness_scores.append(1 / calculate_total_distance(route, distance_matrix))
    
    total_fitness = sum(fitness_scores)
    prob_distribution = [fitness / total_fitness for fitness in fitness_scores]
    
    selected_idx = np.random.choice(len(population), size=int(keepPercent * len(population)), replace=False, p=prob_distribution)
    return [population[i] for i in selected_idx]

# 使用Edge Recombination進行交叉操作
def edge_recombination(parent1, parent2):
    # 建立邊列表
    edges = {i: set() for i in parent1}
    for i in range(len(parent1)):
        left_p1, right_p1 = parent1[i-1], parent1[(i+1) % len(parent1)]
        left_p2, right_p2 = parent2[i-1], parent2[(i+1) % len(parent2)]
        edges[parent1[i]].update([left_p1, right_p1])
        edges[parent2[i]].update([left_p2, right_p2])

    # 隨機選擇一個起始點
    current = random.choice(parent1)
    child = [current]
    
    # 構建子代
    while len(child) < len(parent1):
        for edge in edges.values():
            edge.discard(current)  # 移除已經選擇的點

        if edges[current]:
            current = min(edges[current], key=lambda x: len(edges[x]))  # 選擇最少鄰邊的點
        else:
            current = random.choice([city for city in parent1 if city not in child])
        child.append(current)
    
    return child

# 使用Insert Mutation進行突變操作
def insert_mutate(route, mutation_rate=0.01):
    if random.random() < mutation_rate:
        # 隨機選擇兩個位置
        i, j = random.sample(range(len(route)), 2)
        # 插入操作
        gene = route.pop(i)
        route.insert(j, gene)
    return route

def swap_mutate(route, mutation_rate=0.01):
    for i in range(len(route)):
        if random.random() < mutation_rate:
            swap_with = random.randint(0, len(route) - 1)
            route[i], route[swap_with] = route[swap_with], route[i]
    return route

def scramble_mutate(route, mutation_rate=0.01):
    # Only mutate with a certain probability
    if random.random() > mutation_rate:
        return route  # No mutation, return the original route
    
    # Randomly choose two points to define the sub-section [i, j]
    n = len(route)
    i, j = sorted(random.sample(range(n), 2))  # i < j guaranteed by sorted
    
    # Scramble the section between indices i and j
    scrambled_section = route[i:j+1]
    random.shuffle(scrambled_section)
    
    # Return the new route with the scrambled section
    new_route = route[:i] + scrambled_section + route[j+1:]
    return new_route
# 3-opt 優化
def three_opt(route, distance_matrix):
    route = route
    n = len(route)
    improved = True
    best_dist = float('inf')
    while improved:
        improved = False
        for (i, j, k) in tqdm(itertools.combinations(range(n), 3)):
            # 將三條邊進行3-opt交換
            new_route = route[:i] + route[i:j][::-1] + route[j:k][::-1] + route[k:]
            new_dist = calculate_total_distance(new_route, distance_matrix)
            if new_dist < best_dist:
                best_dist = new_dist
                route = new_route
                improved = True
    return route

# 繪製最佳路徑
def plot_route(route, text, history):
    # print(route)
    # route_points = points[route + [route[0]]]  # 按順序排列並返回到起點
    route_points = np.append(points[route], [points[route[0]]], axis=0)
    
    plt.clf()
    plt.subplot(1, 2, 1)
    for i, point in enumerate(points):
        plt.text(point[0], point[1]+0.1, str(i), fontsize=12)
    
    plt.plot(route_points[:, 0], route_points[:, 1], 'r')
    plt.title(text)

    plt.subplot(1, 2, 2)
    plt.plot(history)
    plt.title("Evolution curve")
    plt.tight_layout()
    plt.pause(0.001)

# 遺傳算法主函數
def genetic_algorithm(distance_matrix, lBound, pop_size=200, generations=500, mutation_rate=0.9, keepPercent=0.25):
    # 初始化種群
    population = init_population(pop_size, N)
    # print(population.shape)
    best_route = np.arange(N, dtype=int)
    best_distance = float('inf')
    best_history = []
    plt.ion()  # 打開交互模式
    for generation in trange(generations):
        new_population = []
        selected = selection(population, distance_matrix, keepPercent)
        for i in range(pop_size - len(selected)):
            parent1, parent2 = np.random.choice(len(selected), size=2, replace=False)
            child1 = edge_recombination(selected[parent1], selected[parent2])
            # varible mutation_rate
            child1 = insert_mutate(child1, (1-i/pop_size) * mutation_rate)
            # child1 = swap_mutate(child1, (1-i/pop_size) * mutation_rate)
            # child1 = scramble_mutate(child1, (1-i/pop_size) * mutation_rate)
            new_population.append(child1)
        population = selected + new_population
        
        # 找到當前代中的最佳路徑
        routeDistances = [calculate_total_distance(route, distance_matrix) for route in population]
        best_idx = np.argmin(routeDistances)
        current_best_route = population[best_idx]
        current_best_distance = min(routeDistances)
        
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_route = current_best_route

        best_history.append(best_distance)
        # 每10代打印最佳距離並畫出路徑
        if (generation + 1) % 5 == 0:
            text = f"Generation {generation + 1:4d}/{generations:4d}: Best Distance = {best_distance:.6f}"
            # print(text)
            plot_route(best_route, text, best_history)
            
    plt.ioff()  # 關閉交互模式
    # 3-opt優化
    current_best_route = three_opt(current_best_route, distance_matrix)
    current_best_distance = calculate_total_distance(current_best_route, distance_matrix)
    if current_best_distance < best_distance:
        best_distance = current_best_distance
        best_route = current_best_route
        text = f"3-opt: Best Distance = {best_distance:.6f}"
        best_history.append(best_distance)
        plot_route(best_route, text, best_history)
    return best_route, best_distance

# 測試
distance_matrix = generate_distance_matrix()
tspLBound = tsp_lower_bound(distance_matrix)
best_route, best_distance = genetic_algorithm(distance_matrix, tspLBound)
print("\nLower bound:", tspLBound)
print("Optimal distance:", best_distance)
plt.show()  # 確保最後一代的圖像顯示