import torch
import numpy as np
from collections import defaultdict
from PPO_agent import PPOAgent, PPO

def simulate_episode(env, agent, num_agents):
    trajectories = defaultdict(list)
    states = env.reset(num_agents)
    done = False

    while not done:
        states_tensor = torch.tensor(states, dtype=torch.float32)
        logits, values = agent(states_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        # Step in environment
        next_states, rewards, done, _ = env.step(actions.numpy())
        
        # Store trajectories
        trajectories["states"].append(states_tensor)
        trajectories["actions"].append(actions)
        trajectories["log_probs"].append(log_probs)
        trajectories["values"].append(values)
        trajectories["rewards"].append(rewards)
        trajectories["dones"].append(done)

        states = next_states

    # Compute returns and advantages
    returns, advantages = ppo.compute_returns_and_advantages(
        trajectories["rewards"],
        trajectories["values"],
        trajectories["dones"]
    )
    trajectories["returns"] = returns
    trajectories["advantages"] = advantages
    return trajectories

class DodgeballEnv:
    def __init__(self, field_size, ball_speed, num_agents):
        self.field_size = field_size
        self.ball_speed = ball_speed
        self.num_agents = num_agents

    def reset(self, num_agents):
        self.agents = np.random.rand(num_agents, 2) * self.field_size
        self.ball_pos = np.array([self.field_size / 2, self.field_size / 2])
        self.ball_velocity = np.random.uniform(-self.ball_speed, self.ball_speed, size=2)
        return self._get_states()

    def _get_states(self):
        states = []
        for agent in self.agents:
            states.append(np.hstack([agent, self.ball_pos, self.ball_velocity]))
        return np.array(states)

    def step(self, actions):
        for i, action in enumerate(actions):
            if action == 0: self.agents[i][0] -= 1  # Left
            elif action == 1: self.agents[i][0] += 1  # Right
            elif action == 2: self.agents[i][1] -= 1  # Down
            elif action == 3: self.agents[i][1] += 1  # Up

        # Update ball position
        self.ball_pos += self.ball_velocity

        # Check for collisions
        rewards, done = [], False
        for agent in self.agents:
            distance = np.linalg.norm(self.ball_pos - agent)
            if distance < 0.5:  # Collision radius
                rewards.append(-1)
                done = True
            else:
                rewards.append(0.1)

        return self._get_states(), rewards, done, {}

env = DodgeballEnv(field_size=10, ball_speed=1, num_agents=5)
agent = PPOAgent(input_size=6, hidden_size=64, output_size=4)
ppo = PPO(agent)

for episode in range(1000):
    trajectories = simulate_episode(env, agent, num_agents=5)
    ppo.update(trajectories)
