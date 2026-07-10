import os
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super().__init__()
        self.max_action = max_action

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
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


class DistributionalCritic(nn.Module):
    """
    DSAC critic:
    输出 Q_mean 和 sigma
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256, sigma_min=1.0):
        super().__init__()
        self.sigma_min = sigma_min

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.q_mean = nn.Linear(hidden_dim, 1)
        self.log_sigma = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.net(x)

        q = self.q_mean(x)
        sigma = F.softplus(self.log_sigma(x)) + self.sigma_min

        return q, sigma


class DSACAgent:
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
        sigma_min=1.0,
        clip_boundary=10.0,
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action
        self.clip_boundary = clip_boundary

        self.memory = ReplayBuffer(memo_capacity)

        self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(self.device)

        self.critic = DistributionalCritic(
            state_dim, action_dim, hidden_dim, sigma_min
        ).to(self.device)

        self.critic_target = DistributionalCritic(
            state_dim, action_dim, hidden_dim, sigma_min
        ).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        if target_entropy is None:
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = target_entropy

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

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

        # =====================
        # 1. 更新 DSAC Critic
        # =====================
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_state)

            next_q, next_sigma = self.critic_target(next_state, next_action)

            noise = torch.randn_like(next_q)
            next_z = next_q + next_sigma * noise

            target_z = reward + (1 - done) * self.gamma * (
                next_z - self.log_alpha.exp() * next_log_prob
            )

        current_q, current_sigma = self.critic(state, action)

        clipped_target_z = torch.clamp(
            target_z,
            current_q.detach() - self.clip_boundary,
            current_q.detach() + self.clip_boundary,
        )

        critic_loss = (
            0.5 * ((clipped_target_z - current_q) / current_sigma).pow(2)
            + torch.log(current_sigma)
        ).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        # =====================
        # 2. 更新 Actor
        # =====================
        new_action, log_prob, _ = self.actor.sample(state)
        q_new, _ = self.critic(state, new_action)

        actor_loss = (self.log_alpha.exp().detach() * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optimizer.step()

        # =====================
        # 3. 更新 alpha
        # =====================
        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # =====================
        # 4. 软更新 target critic
        # =====================
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha,
            "q_mean": current_q.mean().item(),
            "sigma": current_sigma.mean().item(),
            "target_z_mean": target_z.mean().item(),
            "target_z_std": target_z.std().item(),
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