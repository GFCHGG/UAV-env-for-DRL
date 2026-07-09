import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 只要导入一下就能用 3D 投影

from UAV_env import Drone3DEnv
from SAC_agent import SACAgent


def eval_and_plot(agent, env, max_steps=300, render=False, save_fig_path=None):
    """
    使用训练好的 agent 在环境中跑一条轨迹，并画出 3D 路径。
    """
    state = env.reset()
    done = False
    total_reward = 0.0

    # 记录轨迹（完整 episode）
    traj = []
    traj.append(env.drone_pos.copy())

    step = 0
    last_info = {}

    while not done and step < max_steps:
        # 评估时用确定性策略
        action = agent.get_action(state, deterministic=True)

        next_state, reward, done, info = env.step(action)
        last_info = info
        total_reward += reward

        traj.append(env.drone_pos.copy())
        
        state = next_state
        step += 1

        if render:
            env.render(mode='human')

    print(f"[Eval] Steps: {step}, Total Reward: {total_reward:.2f},"
          f"Distance_to_goal: {last_info.get('distance_to_goal', -1):.2f}, "
          f"Success: {last_info.get('success', False)}"
          f"Hitcount: {last_info.get('hit_count')}")

    traj = np.array(traj)
    goal = env.goal_pos.copy()

    # ====== 画 3D 轨迹图 ======
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制障碍物
    env._draw_obstacles(ax=ax)

    # 轨迹线
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=2, label="UAV Path")

    # 起点 & 终点
    ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], s=60, c='blue', marker='o', label="Start")
    ax.scatter(goal[0], goal[1], goal[2], s=80, c='green', marker='*', label="Goal")
    ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], s=60, c='red', marker='^', label="End")

    ax.set_xlim(0, env.space_size[0])
    ax.set_ylim(0, env.space_size[1])
    ax.set_zlim(0, env.space_size[2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("UAV Trajectory (Eval)")
    ax.legend()

    if save_fig_path is not None:
        plt.savefig(save_fig_path, dpi=200)
        print(f"Trajectory figure saved to: {save_fig_path}")

    plt.show()

    return traj


if __name__ == "__main__":
    # 1. 创建环境（可以用跟训练时同样的 curriculum_level，也可以用更高难度测试泛化）
    env = Drone3DEnv(curriculum_level=4)

    # 2. 创建 agent 并加载“最优模型”
    agent = SACAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        memo_capacity=1_000_000,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        max_action=1.0,
        auto_delay=False,
        auto_tau=False,
        uncertainty_alpha=False,
        target_entropy=None, 
        init_policy_delay=1,
    )

    ckpt_path = "./checkpoints/drone_sac/sac_best.pth"  # 对应训练脚本里保存的路径
    agent.load(ckpt_path)
    print(f"Loaded model from {ckpt_path}")

    # 3. 评估并可视化（render=True 可以同时看 matplotlib 里的 3D 场景）
    eval_and_plot(agent, env, max_steps=300, render=False,
                  save_fig_path="./uav_trajectory_eval.png")

    env.close()
