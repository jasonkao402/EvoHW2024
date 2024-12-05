import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOAgent(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PPOAgent, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.policy = nn.Linear(hidden_size, output_size)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy_logits = self.policy(x)
        value = self.value(x)
        return policy_logits, value

class PPO:
    def __init__(self, agent, lr=3e-4, gamma=0.99, clip=0.2, entropy_coeff=0.01):
        self.agent = agent
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
        self.gamma = gamma
        self.clip = clip
        self.entropy_coeff = entropy_coeff

    def compute_returns_and_advantages(self, rewards, values, dones):
        returns, advantages = [], []
        gae = 0
        for step in reversed(range(len(rewards)-1)):
            td_error = rewards[step] + self.gamma * (1 - dones[step]) * values[step + 1] - values[step]
            gae = td_error + self.gamma * gae
            returns.insert(0, gae + values[step])
            advantages.insert(0, gae)
        return torch.tensor(returns), torch.tensor(advantages)

    def update(self, trajectories):
        states = torch.stack(trajectories["states"])
        actions = torch.stack(trajectories["actions"])
        log_probs = torch.stack(trajectories["log_probs"])
        returns = torch.tensor(trajectories["returns"])
        advantages = torch.tensor(trajectories["advantages"])
        
        # PPO optimization step
        for _ in range(4):  # Number of epochs
            logits, values = self.agent(states)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Compute ratios
            ratios = torch.exp(new_log_probs - log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = F.mse_loss(values.squeeze(-1), returns)

            # Total loss
            loss = actor_loss + 0.5 * critic_loss - self.entropy_coeff * entropy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
