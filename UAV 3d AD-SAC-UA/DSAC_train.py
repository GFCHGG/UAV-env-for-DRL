import os
import numpy as np

from UAV_env import Drone3DEnv
from DSAC_agent import DSACAgent

from torch.utils.tensorboard import SummaryWriter


def train():
    # ========== 1. 创建环境和 Agent ==========
    env = Drone3DEnv(curriculum_level=4)
    env.load_map("map/tmap_level4.pkl")

    agent = DSACAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
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
    )

    # ========== 2. 日志和模型保存目录 ==========
    log_dir = "./logs/drone_sac/DSAC"
    ckpt_dir = "./checkpoints/drone_sac/DSAC"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)

    num_episodes = 3000
    random_explore_steps = 1000
    save_interval = 500
    global_step = 0
    reward_history = []
    best_avg_reward = -1e9

    for episode in range(num_episodes):
        state, _ = env.reset()

        episode_reward = 0.0
        done = False
        last_info = {}
        losses = {}

        while not done:
            # ===== 2.1 行为策略：先随机探索，再用 DSAC 策略 =====
            if agent.memory.memo_counter < random_explore_steps:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state, deterministic=False)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            last_info = info

            agent.add_memo(state, action, reward, next_state, done)

            if agent.memory.memo_counter >= agent.batch_size:
                losses = agent.update()
                global_step += 1

                if losses:
                    writer.add_scalar("Loss/critic", losses["critic_loss"], global_step)
                    writer.add_scalar("Loss/actor", losses["actor_loss"], global_step)
                    writer.add_scalar("Loss/alpha_loss", losses["alpha_loss"], global_step)

                    writer.add_scalar("Alpha/value", losses["alpha"], global_step)

                    writer.add_scalar("DSAC/q_mean", losses["q_mean"], global_step)
                    writer.add_scalar("DSAC/sigma", losses["sigma"], global_step)

                    writer.add_scalar("Target_Z/mean", losses["target_z_mean"], global_step)
                    writer.add_scalar("Target_Z/std", losses["target_z_std"], global_step)

            state = next_state
            episode_reward += reward

        # ===== 2.2 每个 episode 结束后记录 =====
        reward_history.append(episode_reward)
        avg_reward = np.mean(reward_history[-100:])

        distance_to_goal = last_info.get("distance_to_goal", -1.0)
        steps = last_info.get("steps", 0)
        hit_count = last_info.get("hit_count", 0)
        success = last_info.get("success", False)

        writer.add_scalar("Train/episode_reward", episode_reward, episode)
        writer.add_scalar("Train/avg_reward_100", avg_reward, episode)
        writer.add_scalar("Train/distance_to_goal", distance_to_goal, episode)
        writer.add_scalar("Train/steps", steps, episode)
        writer.add_scalar("Train/hit_count", hit_count, episode)
        writer.add_scalar("Train/success", float(success), episode)

        if episode % 100 == 0:
            print(
                f"Episode {episode}, "
                f"Reward: {episode_reward:.2f}, "
                f"AvgReward(100): {avg_reward:.2f}, "
                f"Distance: {distance_to_goal:.2f}, "
                f"Hit_count: {hit_count}, "
                f"Success: {success}, "
                f"Alpha: {agent.alpha:.3f}"
            )

            if losses:
                print("  Losses:", losses)

        # ===== 2.3 周期性保存模型 =====
        if (episode + 1) % save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"sac_ep{episode + 1}.pth")
            agent.save(ckpt_path)
            print(f"[Checkpoint] Saved model to: {ckpt_path}")

        # ===== 2.4 保存最佳模型 =====
        if avg_reward > best_avg_reward and episode > 50:
            best_avg_reward = avg_reward
            best_ckpt_path = os.path.join(ckpt_dir, "sac_best.pth")
            agent.save(best_ckpt_path)
            print(
                f"[Best] New best model "
                f"(avg_reward={best_avg_reward:.2f}) saved to: {best_ckpt_path}"
            )

    writer.close()


if __name__ == "__main__":
    train()