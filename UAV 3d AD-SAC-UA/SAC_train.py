import os
import numpy as np
from UAV_env import Drone3DEnv
from SAC_agent import SACAgent

import torch
from torch.utils.tensorboard import SummaryWriter


def train():
    # ========== 1. 创建环境和 Agent ==========
    env = Drone3DEnv(curriculum_level=4)  # 先用简单难度
    env.load_map("map/tmap_level4.pkl")
    agent = SACAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        memo_capacity=1_000_000,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        max_action=1.0,
        auto_delay=True,
        auto_tau=False,
        uncertainty_alpha=True,
        target_entropy=None, 
        init_policy_delay=1,
    )

    # ========== 2. 日志和模型保存目录 ==========
    log_dir = "./logs/drone_sac/AD_SAC_UA"
    ckpt_dir = "./checkpoints/drone_sac/AD_SAC_UA"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    # 一些训练超参数
    num_episodes = 3000 
    random_explore_steps = 1000   # 前多少步用随机动作
    save_interval = 500           # 每多少个 episode 存一次模型
    global_step = 0               # 用于记录 loss 的 step
    reward_history = []           # 用来算最近100回合平均奖励
    best_avg_reward = -1e9        # 用来保存最好的模型

    for episode in range(num_episodes):
        state, _  = env.reset()  # 你的 env.reset() 返回的是 obs（不是 obs, info）
        episode_reward = 0.0
        done = False
        last_info = {}
        losses = {}

        while not done:
            # ===== 2.1 行为策略：先随机探索，再用 SAC 策略 =====
            if agent.memory.memo_counter < random_explore_steps:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state, deterministic=False)

            # 与环境交互
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            last_info = info

            # 存入经验回放
            agent.add_memo(state, action, reward, next_state, done)

            # SAC 更新
            if agent.memory.memo_counter >= agent.batch_size:
                losses = agent.update()
                global_step += 1

                # ===== 2.2 TensorBoard 记录 loss 和 alpha =====
                if losses:  # update 里有可能返回 {}
                    writer.add_scalar("Loss/critic_1", losses['critic_1_loss'], global_step)
                    writer.add_scalar("Loss/critic_2", losses['critic_2_loss'], global_step)
                    writer.add_scalar("Loss/actor",    losses['actor_loss'],    global_step)
                    writer.add_scalar("Alpha/value",   losses['alpha'],         global_step)
                    writer.add_scalar("Uncertainty/ema",losses['uncertainty_ema'],global_step)
                    writer.add_scalar("Uncertainty/q1_q2_diff",losses['q_diff'],global_step)
                    writer.add_scalar("Stability/policy_delay", losses['policy_delay'], global_step)
                    writer.add_scalar("Stability/tau",          losses['tau'],          global_step)
                    writer.add_scalar("Stability/instability_target",  losses['instability_target'],  global_step)
                    writer.add_scalar("Stability/instability_actor",  losses['instability_actor'],  global_step)
                    writer.add_scalar("Stability/critic_loss_ema", losses['critic_loss_ema'], global_step)
                    writer.add_scalar("Target_Q/bellman_target_var", losses['bellman_target_var'], global_step)
                    writer.add_scalar("Target_Q/bellman_target_std", losses['bellman_target_std'], global_step)
                    writer.add_scalar("Target_Q/bellman_target_mean", losses['bellman_target_mean'], global_step)

            state = next_state
            episode_reward += reward

        # ===== 2.3 每个 episode 结束后记录奖励和距离 =====
        reward_history.append(episode_reward)
        avg_reward = np.mean(reward_history[-100:])  # 最近100回合平均

        distance_to_goal = last_info.get("distance_to_goal", -1.0)
        steps = last_info.get("steps", 0)
        hit_count= last_info.get("hit_count", 0)

        writer.add_scalar("Train/episode_reward", episode_reward, episode)
        writer.add_scalar("Train/avg_reward_100", avg_reward, episode)
        writer.add_scalar("Train/distance_to_goal", distance_to_goal, episode)
        writer.add_scalar("Train/steps", steps, episode)
        writer.add_scalar("Train/hit_count", hit_count, episode)

        # 控制台输出
        if episode % 100 == 0:
            print(
                f"Episode {episode}, "
                f"Reward: {episode_reward:.2f}, "
                f"AvgReward(100): {avg_reward:.2f}, "
                f"Distance: {distance_to_goal:.2f}, "
                f"Hit_count: {hit_count}, "
                f"Alpha: {agent.alpha:.3f}"
            )
            if losses:
                print("  Losses:", losses)

        # ===== 2.4 周期性保存模型，多版本 =====
        if (episode + 1) % save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"sac_ep{episode+1}.pth")
            agent.save(ckpt_path)
            print(f"[Checkpoint] Saved model to: {ckpt_path}")

        # ===== 2.5 保存“最佳模型”（按最近100回合平均回报） =====
        if avg_reward > best_avg_reward and episode > 50:  # 前50回合不算
            best_avg_reward = avg_reward
            best_ckpt_path = os.path.join(ckpt_dir, "sac_best.pth")
            agent.save(best_ckpt_path)
            print(f"[Best] New best model (avg_reward={best_avg_reward:.2f}) saved to: {best_ckpt_path}")

    writer.close()


if __name__ == "__main__":
    train()
