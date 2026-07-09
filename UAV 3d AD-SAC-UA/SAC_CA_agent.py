import os
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
from torch.distributions import Normal


LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-6


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.memo_counter = 0

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.memo_counter += 1

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = map(np.array, zip(*batch))

        return (
            torch.FloatTensor(s),
            torch.FloatTensor(a),
            torch.FloatTensor(r).unsqueeze(1),
            torch.FloatTensor(s_),
            torch.FloatTensor(d).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super().__init__()
        self.max_action = max_action

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        x = normal.rsample()
        y = torch.tanh(x)

        action = y * self.max_action

        log_prob = normal.log_prob(x)
        log_prob -= torch.log(self.max_action * (1 - y.pow(2)) + EPS)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        deterministic_action = torch.tanh(mean) * self.max_action

        return action, log_prob, deterministic_action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)


class SACCAAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        memo_capacity=1_000_000,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        max_action=1.0,
        target_entropy=None,
        hidden_dim=256,
        alpha_min=0.01,
        alpha_max=1.0,
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        self.memory = ReplayBuffer(memo_capacity)

        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.target_entropy = -float(action_dim) if target_entropy is None else target_entropy

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def _clip_alpha(self):
        with torch.no_grad():
            alpha = self.log_alpha.exp().clamp(self.alpha_min, self.alpha_max)
            self.log_alpha.data.copy_(torch.log(alpha))

    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if deterministic:
                _, _, action = self.actor.sample(state)
            else:
                action, _, _ = self.actor.sample(state)

        return action.cpu().numpy()[0]

    def add_memo(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return {}

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        alpha_tensor = self.log_alpha.exp().detach()

        # ========== 1. 更新 Critic ==========
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)

            target_value = target_q - alpha_tensor * next_log_prob
            target_q_value = reward + (1 - done) * self.gamma * target_value

        current_q1, current_q2 = self.critic(state, action)

        critic_1_loss = nn.MSELoss()(current_q1, target_q_value)
        critic_2_loss = nn.MSELoss()(current_q2, target_q_value)
        critic_loss = critic_1_loss + critic_2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ========== 2. 更新 Actor ==========
        new_action, log_prob, _ = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, new_action)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.log_alpha.exp().detach() * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ========== 3. SAC-CA：带 clip 的自动熵系数更新 ==========
        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._clip_alpha()

        # ========== 4. 软更新 target critic ==========
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        with torch.no_grad():
            q_diff = torch.abs(current_q1 - current_q2).mean().item()
            bellman_target_mean = target_q_value.mean().item()
            bellman_target_std = target_q_value.std().item()
            bellman_target_var = target_q_value.var().item()

        return {
            "critic_1_loss": critic_1_loss.item(),
            "critic_2_loss": critic_2_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha,
            "q_diff": q_diff,
            "bellman_target_mean": bellman_target_mean,
            "bellman_target_std": bellman_target_std,
            "bellman_target_var": bellman_target_var,
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.log_alpha.data = checkpoint["log_alpha"].to(self.device)
        self._clip_alpha()