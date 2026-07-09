import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using device: {device}')


class ReplayMemory:
    def __init__(self, memo_capacity, state_dim, action_dim):
        self.memo_size = int(memo_capacity)
        self.state_memo = np.zeros((self.memo_size, state_dim), dtype=np.float32)
        self.next_state_memo = np.zeros((self.memo_size, state_dim), dtype=np.float32)
        self.action_memo = np.zeros((self.memo_size, action_dim), dtype=np.float32)
        self.reward_memo = np.zeros(self.memo_size, dtype=np.float32)
        self.done_memo = np.zeros(self.memo_size, dtype=np.float32)  # 0/1 float
        self.memo_counter = 0

    def add_memory(self, state, action, reward, next_state, done):
        index = self.memo_counter % self.memo_size
        self.state_memo[index] = state
        self.next_state_memo[index] = next_state
        self.action_memo[index] = action
        self.reward_memo[index] = reward
        self.done_memo[index] = float(done)
        self.memo_counter += 1

    def sample_memory(self, batch_size):
        current_memo_size = min(self.memo_counter, self.memo_size)
        batch = np.random.choice(current_memo_size, batch_size, replace=False)
        return (self.state_memo[batch], self.action_memo[batch],
                self.reward_memo[batch], self.next_state_memo[batch],
                self.done_memo[batch])


class CriticNetwork(nn.Module):
    """Q网络: Q(s,a) -> scalar"""
    def __init__(self, state_dim, action_dim, fc1_dim=256, fc2_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)


class ActorNetwork(nn.Module):
    """策略网络: π(s) -> 高斯分布参数"""
    def __init__(self, state_dim, action_dim, fc1_dim=256, fc2_dim=256, max_action=1.0):
        super().__init__()
        self.max_action = max_action

        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.mu = nn.Linear(fc2_dim, action_dim)
        self.log_std = nn.Linear(fc2_dim, action_dim)

        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, state, deterministic=False):
        mu, log_std = self.forward(state)
        std = log_std.exp()

        if deterministic:
            action = torch.tanh(mu) * self.max_action
            return action, None

        normal = Normal(mu, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class SACAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 memo_capacity=1_000_000,
                 gamma=0.99,

                 # ===== 固定/初始参数 =====
                 tau=0.005,                 # init_tau：auto_tau=False时固定；auto_tau=True时作为初始/参考
                 batch_size=256,
                 lr=3e-4,
                 max_action=1.0,

                 # ===== 新增开关 =====
                 auto_delay=True,           # True: 自适应 policy delay；False: 固定 delay
                 auto_tau=True,             # True: 自适应 tau（仅当 auto_delay=True 才生效）；False: 固定 tau
                 uncertainty_alpha = True,  # True: 不确定性的alpha更新：False：基于目标熵的alpha更新

                 # ===== 目标熵 =====
                 target_entropy= None,

                 # ===== 固定 delay 的初始值 =====
                 init_policy_delay=4,

                 # ===== delay 范围（auto_delay=True时用）=====
                 min_policy_delay=1,
                 max_policy_delay=4,

                 # ===== tau 范围（auto_tau 生效时用）=====
                 tau_min=0.002,
                 tau_max=0.01):

        self.gamma = gamma
        self.batch_size = batch_size

        # ===== 新增开关保存 =====
        self.auto_delay = bool(auto_delay)
        self.auto_tau = bool(auto_tau)
        self.uncertainty_alpha = bool(uncertainty_alpha)

        # init/fixed 参数
        self.init_tau = float(tau)
        self.init_policy_delay = int(init_policy_delay)

        # Replay Buffer
        self.memory = ReplayMemory(memo_capacity, state_dim, action_dim)

        # Networks
        self.actor = ActorNetwork(state_dim, action_dim, max_action=max_action).to(device)
        self.critic_1 = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_2 = CriticNetwork(state_dim, action_dim).to(device)

        self.target_critic_1 = CriticNetwork(state_dim, action_dim).to(device)
        self.target_critic_2 = CriticNetwork(state_dim, action_dim).to(device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)

        # ===== 不确定性驱动 alpha 参数 =====
        self.alpha_min = 0.05
        self.alpha_max = 0.9

        self.ema_beta = 0.99
        self.uncertainty_ema = 0.0

        self.quantile_q = 0.1
        self.uncertainty_buffer = []
        self.uncertainty_buffer_size = int(1e6)

        self.alpha = self.alpha_max

        # ===== critic loss EMA（用于“抖动”检测）=====
        self.critic_loss_ema = 0.0
        self.critic_loss_beta = 0.99

        # ===== policy delay 参数 =====
        self.min_policy_delay = int(min_policy_delay)
        self.max_policy_delay = int(max_policy_delay)

        # 固定/初始 delay（auto_delay=False 时一直用它；auto_delay=True 时作为fallback）
        self.policy_delay = int(np.clip(self.init_policy_delay, self.min_policy_delay, self.max_policy_delay))
        self.actor_update_counter = 0

        # ===== tau 参数 =====
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)

        # 当前使用的 tau（最终会在 update() 里根据开关更新）
        self.current_tau = float(self.init_tau)

        # 用于日志/调试
        self.last_instability_target = 1.0
        self.last_instability_actor = 1.0

        if not uncertainty_alpha:
            if target_entropy is None:
                self.target_entropy = -action_dim  # 启发式目标熵
            else:
                self.target_entropy = target_entropy
            
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item() 

    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _ = self.actor.sample(state, deterministic)
        return action.cpu().numpy()[0]

    def add_memo(self, state, action, reward, next_state, done):
        self.memory.add_memory(state, action, reward, next_state, done)

    def _compute_instability(self, U_low, U_high):
        """
        instability ∈ [0,1]
        - 基于 q_diff 的“相对分位数位置”（norm_u）
        - 再融合 critic_loss_ema 的归一化
        """
        eps = 1e-6

        # 1) q_diff 归一化（分位数阈值）
        q_score = (self.uncertainty_ema - U_low) / (U_high - U_low + eps)
        q_score = float(np.clip(q_score, 0.0, 1.0))

        # 2) critic_loss_ema 归一化（log 压缩）
        loss_score = np.log1p(self.critic_loss_ema)
        loss_score = float(np.clip(loss_score / 2.0, 0.0, 1.0))

        # 3) 合成
        instability_target = 0.6 * q_score + 0.4 * loss_score
        instability_actor = q_score
        return float(np.clip(instability_target, 0.0, 1.0)), float(np.clip(instability_actor, 0.0, 1.0))

    def update(self):
        if self.memory.memo_counter < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.memory.sample_memory(self.batch_size)

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # ===== Critic Update =====
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1 = self.target_critic_1(next_states, next_actions)
            target_q2 = self.target_critic_2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * (target_q - self.alpha * next_log_probs)
        
        # ===== Bellman target statistics (for logging & analysis) =====
        with torch.no_grad():
            bellman_target_mean = target_q.mean().item()
            bellman_target_std = target_q.std().item()
            bellman_target_var = target_q.var().item()


        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)

        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # ===== critic loss EMA（用于“抖动”检测）=====
        critic_loss_scalar = 0.5 * (critic_1_loss.item() + critic_2_loss.item())
        self.critic_loss_ema = (self.critic_loss_beta * self.critic_loss_ema
                                + (1 - self.critic_loss_beta) * critic_loss_scalar)

        # ===== 双 Q 不确定性（Q1-Q2）=====
        with torch.no_grad():
            q_diff_mean = torch.abs(current_q1 - current_q2).mean().item()

        # EMA 平滑
        self.uncertainty_ema = (self.ema_beta * self.uncertainty_ema
                                + (1 - self.ema_beta) * q_diff_mean)

        # Buffer（用于分位数阈值）
        self.uncertainty_buffer.append(self.uncertainty_ema)
        if len(self.uncertainty_buffer) > self.uncertainty_buffer_size:
            self.uncertainty_buffer.pop(0)

        # Quantile thresholds
        if len(self.uncertainty_buffer) > 50:
            u = np.asarray(self.uncertainty_buffer, dtype=np.float32)
            U_low = float(np.quantile(u, self.quantile_q))
            U_high = float(np.quantile(u, 1 - self.quantile_q))
        else:
            U_low, U_high = 0.0, 1.0

        # ===== 根据开关决定  Clip-based alpha  或是原逻辑=====
        if self.uncertainty_alpha:
            eps = 1e-6
            norm_u = (self.uncertainty_ema - U_low) / (U_high - U_low + eps)
            norm_u = float(np.clip(norm_u, 0.0, 1.0))
            self.alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * norm_u
        else :
            new_actions, log_probs = self.actor.sample(states)
            alpha_loss = -(self.log_alpha * 
                          (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
    

        # ===== 根据开关决定 policy_delay & tau =====
        if self.auto_delay:
            # 自适应稳定器
            instability_target, instability_actor = self._compute_instability(U_low, U_high)
            self.last_instability_target = instability_target
            self.last_instability_actor = instability_actor

            # policy_delay：不稳定 → delay 大
            self.policy_delay = int(
                self.min_policy_delay + instability_actor * (self.max_policy_delay - self.min_policy_delay)
            )
            self.policy_delay = int(np.clip(self.policy_delay, self.min_policy_delay, self.max_policy_delay))

            # tau：只有 auto_delay=True 且 auto_tau=True 才生效
            if self.auto_tau:
                self.current_tau = self.tau_max - instability_actor * (self.tau_max - self.tau_min)
                self.current_tau = float(np.clip(self.current_tau, self.tau_min, self.tau_max))
            else:
                self.current_tau = float(self.init_tau)

        else:
            # 固定 delay
            self.policy_delay = int(np.clip(self.init_policy_delay, self.min_policy_delay, self.max_policy_delay))
            # 固定 tau（注意：auto_tau 即使 True 也不生效）
            self.current_tau = float(self.init_tau)

            # 日志值给个合理定义（固定模式下可看成“未知/不使用”，这里置 0）
            self.last_instability_target = 0.0
            self.last_instability_actor = 0.0

        # ===== Actor Update（按 policy_delay 执行）=====
        self.actor_update_counter += 1
        actor_loss_value = None

        if (self.actor_update_counter % self.policy_delay) == 0:
            new_actions, log_probs = self.actor.sample(states)
            q1_new = self.critic_1(states, new_actions)
            q2_new = self.critic_2(states, new_actions)
            q_new = torch.min(q1_new, q2_new)

            actor_loss = (self.alpha * log_probs - q_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_loss_value = actor_loss.item()

        # ===== Target Soft Update（用 current_tau）=====
        self._soft_update(self.critic_1, self.target_critic_1, self.current_tau)
        self._soft_update(self.critic_2, self.target_critic_2, self.current_tau)

        return {
            'critic_1_loss': critic_1_loss.item(),
            'critic_2_loss': critic_2_loss.item(),
            'actor_loss': actor_loss_value if actor_loss_value is not None else 0.0,

            'alpha': float(self.alpha),
            'uncertainty_ema': float(self.uncertainty_ema),
            'q_diff': float(q_diff_mean),

            # ===== Bellman target logging =====
            'bellman_target_mean': bellman_target_mean,
            'bellman_target_std': bellman_target_std,
            'bellman_target_var': bellman_target_var,

            # 额外建议记录（tensorboard）
            'policy_delay': int(self.policy_delay),
            'tau': float(self.current_tau),
            'instability_target': float(self.last_instability_target),
            'instability_actor': float(self.last_instability_actor),
            'critic_loss_ema': float(self.critic_loss_ema),

            # 开关状态也可以顺便回传便于确认
            'auto_delay': int(self.auto_delay),
            'auto_tau': int(self.auto_tau),
            'uncertainty_alpha': int(self.uncertainty_alpha)
        }

    def _soft_update(self, source, target, tau):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

    def save(self, filename):
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'target_critic_1': self.target_critic_1.state_dict(),
            'target_critic_2': self.target_critic_2.state_dict(),

            # 不确定性/alpha 状态
            'alpha': float(self.alpha),
            'uncertainty_ema': float(self.uncertainty_ema),
            'uncertainty_buffer': np.asarray(self.uncertainty_buffer, dtype=np.float32),

            # 稳定器状态（resume 更连续）
            'actor_update_counter': int(self.actor_update_counter),
            'critic_loss_ema': float(self.critic_loss_ema),

            # 保存开关与初始值（resume 一致性更好）
            'auto_delay': int(self.auto_delay),
            'auto_tau': int(self.auto_tau),
            'uncertainty_alpha': int(self.uncertainty_alpha),
            'init_policy_delay': int(self.init_policy_delay),
            'init_tau': float(self.init_tau),
        }
        torch.save(checkpoint, filename)

    def load(self, filename, map_location=None):
        checkpoint = torch.load(filename, map_location=map_location, weights_only=False)

        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.target_critic_1.load_state_dict(checkpoint['target_critic_1'])
        self.target_critic_2.load_state_dict(checkpoint['target_critic_2'])

        self.alpha = float(checkpoint.get('alpha', self.alpha_min))
        self.uncertainty_ema = float(checkpoint.get('uncertainty_ema', 0.0))
        ub = checkpoint.get('uncertainty_buffer', None)
        self.uncertainty_buffer = np.asarray(ub, dtype=np.float32).tolist() if ub is not None else []

        self.actor_update_counter = int(checkpoint.get('actor_update_counter', 0))
        self.critic_loss_ema = float(checkpoint.get('critic_loss_ema', 0.0))

        # 恢复开关/初始值（可选：你也可以选择不恢复，完全由当前构造参数决定）
        self.auto_delay = bool(checkpoint.get('auto_delay', int(self.auto_delay)))
        self.auto_tau = bool(checkpoint.get('auto_tau', int(self.auto_tau)))
        self.init_policy_delay = int(checkpoint.get('init_policy_delay', self.init_policy_delay))
        self.init_tau = float(checkpoint.get('init_tau', self.init_tau))

        # 载入后根据当前开关设置 current_tau/policy_delay 的初值
        if not self.auto_delay:
            self.policy_delay = int(np.clip(self.init_policy_delay, self.min_policy_delay, self.max_policy_delay))
            self.current_tau = float(self.init_tau)

        print(f"[Load] Model loaded from {filename}")
        print(f"       alpha={self.alpha:.4f}, uncertainty_ema={self.uncertainty_ema:.6f}, "
              f"buffer_size={len(self.uncertainty_buffer)}, actor_update_counter={self.actor_update_counter}")
        print(f"       auto_delay={self.auto_delay}, auto_tau={self.auto_tau}, "
              f"init_policy_delay={self.init_policy_delay}, init_tau={self.init_tau}")
