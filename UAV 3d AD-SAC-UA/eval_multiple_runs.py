import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # 只要导入一下就能用 3D 投影

from UAV_env import Drone3DEnv
from SAC_agent import SACAgent
from DSAC_agent import DSACAgent
from SAC_CA_agent import SACCAAgent


def evaluate_multiple_runs(agent, env, runs=300, max_steps=300,csv_path="eval_results.csv"):

    success_count = 0
    total_collisions = 0
    total_step = 0

    records = []

    for i in range(runs):
        
        env = Drone3DEnv(curriculum_level=4)

        state,info = env.reset()
        done = False
        step = 0
        episode_collisions = 0
        success = False
        reward = 0

        while not done and step < max_steps:
            action = agent.get_action(state, deterministic=True)

            # ✔ 正确解包 5 个返回值
            next_state, reward,  terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            episode_collisions = info['hit_count']
            success = info['success']

            state = next_state
            step += 1

        if success:
            success_count += 1
            total_collisions += episode_collisions
            total_step += step 

        records.append({"run": i + 1,"success": success,"collisions": episode_collisions,"steps": step,"final_reward": reward,})   
        print(f"[Run {i+1}/{runs}] Success: {success}, Collisions: {episode_collisions}, Reward:{reward}, step:{step}")

    success_rate = success_count / runs * 100
    
    if success_count > 0:
        avg_collision = total_collisions / success_count
        avg_step = total_step / success_count
    else:
        avg_collision = 0
        avg_step = 0

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Total Runs: {runs}")
    print(f"Successes: {success_count}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Collisions per Episode: {avg_collision:.2f}")
    print(f"Average steps per Episode: {avg_step:.2f}")
    print("====================================================")

    return success_rate, avg_collision

if __name__ == "__main__":
    env = Drone3DEnv(curriculum_level=4)
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

    # agent = SACCAAgent(
    #     state_dim=env.observation_space.shape[0],
    #     action_dim=env.action_space.shape[0],
    #     memo_capacity=1_000_000,
    #     lr=3e-4,
    #     gamma=0.99,
    #     tau=0.005,
    #     batch_size=256,
    #     max_action=1.0,
    #     target_entropy=None,
    #     hidden_dim=256,
    #     alpha_min=0.01,
    #     alpha_max=1.0,
    # )

    # agent = DSACAgent(
    #     state_dim=env.observation_space.shape[0],
    #     action_dim=env.action_space.shape[0],
    #     memo_capacity=1_000_000,
    #     lr=3e-4,
    #     gamma=0.99,
    #     tau=0.005,
    #     batch_size=256,
    #     max_action=1.0,
    #     target_entropy=None,
    #     hidden_dim=256,
    #     sigma_min=1.0,
    #     clip_boundary=10.0,
    # )

    ckpt_path = "./checkpoints/drone_sac/SD_SAC_UA/sac_ep3000.pth"
    agent.load(ckpt_path)
    print(f"Loaded model from {ckpt_path}")

    # ★ 新增：批量评估 300 次
    evaluate_multiple_runs(
        agent,
        env,
        runs=300,
        max_steps=300,
        csv_path="./data1/SD_SAC_UA/eval_300_runs5.csv"
    )

    env.close()