import pygame
import numpy as np
import pygame.freetype
from dodgeUtil import totalFitness, PlayerNeuralNetwork
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium import spaces
import gymnasium as gym

# Initialize pygame
pygame.init()

# Settings
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 700
FIELD_SIZE = 100
ZOOM = 6
OFFSET = (WINDOW_WIDTH - FIELD_SIZE * ZOOM) // 2
OFFSET_POS = np.ones(2) * OFFSET
WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Dodgeball Simulation")

# Define colors
WHITE = (255, 255, 255)
GRAY = (32, 32, 32)
RED = (255, 0, 0)
BLUE = (0, 118, 214)
GREEN = (0, 232, 152)

# Define parameters
PLAYER_RADIUS = 10
BALL_RADIUS = 15
BALL_SPEED = 0.5
BALL_DECAY = 0.98  # Simulated air resistance

# Pygame clock and font
clock = pygame.time.Clock()
font = pygame.freetype.SysFont('Arial', 24)

# Define environment
class DodgeballEnv(gym.Env):
    def __init__(self):
        super(DodgeballEnv, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)  # Agent movement (x, y)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32  # Agent x/y, ball x/y (normalized)
        )
        self.reset()

    def reset(self, seed=None, **maybe_options):
        self.agent_position = np.random.uniform(0, FIELD_SIZE, 2)
        self.ball_position = np.random.uniform(0, FIELD_SIZE, 2)
        self.ball_velocity = np.random.uniform(-BALL_SPEED, BALL_SPEED, 2)
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.agent_position / FIELD_SIZE, self.ball_position / FIELD_SIZE])

    def step(self, action):
        # Update agent position
        self.agent_position += action
        self.agent_position = np.clip(self.agent_position, 0, FIELD_SIZE)

        # Update ball position
        self.ball_position += self.ball_velocity
        self.ball_velocity *= BALL_DECAY

        # Ball collision with walls
        if self.ball_position[0] <= BALL_RADIUS or self.ball_position[0] >= FIELD_SIZE - BALL_RADIUS:
            self.ball_velocity[0] *= -1
        if self.ball_position[1] <= BALL_RADIUS or self.ball_position[1] >= FIELD_SIZE - BALL_RADIUS:
            self.ball_velocity[1] *= -1

        # Compute fitness
        fitness = totalFitness(self.ball_position, self.agent_position, action)

        # Return observation, reward, done, and info
        out_of_bounds = np.any(self.agent_position < 0) or np.any(self.agent_position > FIELD_SIZE)
        return self._get_obs(), fitness, out_of_bounds, False, {}

# Initialize environment and model
env = DummyVecEnv([lambda: DodgeballEnv()])
model = SAC("MlpPolicy", env, verbose=1)

# Training parameters
TRAINING_TIMESTEPS = 10000
EVAL_EPISODES = 10

def train_and_visualize():
    model.learn(total_timesteps=TRAINING_TIMESTEPS, progress_bar=True)

    # Evaluate the policy
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=EVAL_EPISODES)
    print(f"Mean reward: {mean_reward}, Std: {std_reward}")

    obs = env.reset()
    for _ in range(1000):
        # Predict action
        action, _ = model.predict(obs, deterministic=True)

        # Take a step in the environment
        obs, reward, done, info = env.step(action)

        # Visualization
        WINDOW.fill(GRAY)
        agent_position = obs[0, :2] * FIELD_SIZE
        ball_position = obs[0, 2:] * FIELD_SIZE

        pygame.draw.circle(WINDOW, BLUE, agent_position * ZOOM + OFFSET_POS, PLAYER_RADIUS)
        pygame.draw.circle(WINDOW, RED, ball_position * ZOOM + OFFSET_POS, BALL_RADIUS)

        pygame.draw.rect(WINDOW, GREEN, (OFFSET, OFFSET, FIELD_SIZE * ZOOM, FIELD_SIZE * ZOOM), 3)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

# Run training and visualization
train_and_visualize()
