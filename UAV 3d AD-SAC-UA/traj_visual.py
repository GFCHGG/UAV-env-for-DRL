import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyvista as pv

from UAV_env import Drone3DEnv
from SAC_agent import SACAgent

from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib.lines import Line2D
# ===========================
# 量化指标计算函数
# ===========================
def calculate_path_length(traj: np.ndarray) -> float:
    """计算总路径长度"""
    return np.sum(np.linalg.norm(traj[1:] - traj[:-1], axis=1))

def calculate_smoothness(traj: np.ndarray) -> float:
    """计算路径弯曲程度（转角总和）"""
    smoothness = 0.0
    for i in range(1, len(traj)-1):
        v1 = traj[i] - traj[i-1]
        v2 = traj[i+1] - traj[i]
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 > 1e-6 and norm2 > 1e-6:
            cos_theta = np.clip(np.dot(v1, v2)/(norm1*norm2), -1.0, 1.0)
            smoothness += np.arccos(cos_theta)
    return smoothness

def calculate_redundancy(traj: np.ndarray, start: np.ndarray, goal: np.ndarray) -> float:
    """计算路径冗余率：实际路径长度 / 最短直线距离"""
    straight_dist = np.linalg.norm(goal - start)
    actual_length = calculate_path_length(traj)
    return actual_length / straight_dist


# ===========================
# 单模型轨迹采集
# ===========================
def rollout_one_model(agent, env, max_steps=300):
    """在当前 env 中用确定性策略跑一条轨迹"""
    state, _ = env.reset()
    traj = [env.drone_pos.copy()]

    done = False
    step = 0
    last_info = {}

    while not done and step < max_steps:
        action = agent.get_action(state, deterministic=True)
        next_state, reward,terminated, truncated, info = env.step(action)
        done=terminated or truncated
        last_info = info

        traj.append(env.drone_pos.copy())
        state = next_state
        step += 1

    return np.array(traj), last_info


# ===========================
# 多模型可视化 + 量化指标
# ===========================
def visualize_multiple_models(
    ckpt_paths,
    model_names,
    curriculum_level=3,
    max_steps=300,
    save_fig_path=None,
    save_topdown_fig_path=None,
):
    assert len(ckpt_paths) == len(model_names), "模型路径和名称数量必须一致"

    env = Drone3DEnv(curriculum_level=curriculum_level)

    # =======================
    # 3D轨迹图
    # =======================
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    env.reset()
    env._draw_obstacles(ax=ax)

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(ckpt_paths))]

    legend_handles = []
    legend_labels = []

    all_trajs = []
    traj_metrics = []

    for i, (ckpt, name) in enumerate(zip(ckpt_paths, model_names)):
        agent = SACAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            memo_capacity=1_000_000,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            batch_size=256,
            max_action=1.0,
        )
        agent.load(ckpt)

        traj, info = rollout_one_model(agent, env, max_steps)
        all_trajs.append(traj)

        
        # =======================
        # 计算量化指标
        # =======================
        path_len = calculate_path_length(traj)
        smooth = calculate_smoothness(traj)
        redundancy = calculate_redundancy(traj, traj[0], env.goal_pos)

        traj_metrics.append({
            "model": name,
            "success": info['success'],
            "path_length": path_len,
            "smoothness": smooth,
            "redundancy": redundancy,
            "collisions": info['hit_count'],
            "steps": info['steps'],
        })

        # =======================
        # 3D轨迹可视化
        # =======================
        #line, = ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],linewidth=2,color=colors[i])
        line = Line3D(traj[:, 0],traj[:, 1],traj[:, 2],linewidth=2.5,color=colors[i],alpha=1.0)
    
        ax.add_line(line)
        legend_handles.append(Line2D([0],[0],color=colors[i],lw=2.5))
        #legend_handles.append(line)
        legend_labels.append(name)

        if i == 0:
            ax.scatter(
                traj[0, 0], traj[0, 1], traj[0, 2],
                s=60, c='blue', marker='o', label="Start"
            )

        ax.scatter(
            traj[-1, 0], traj[-1, 1], traj[-1, 2],
            s=50, marker='^', color=colors[i]
        )
    
    visualize_multiple_models_pyvista(
            env=env,

            trajectories=all_trajs,

            model_names=model_names,

            screenshot_path="picture/pyvista_result.tif"
        )

    visualize_topdown_pyvista(
            env=env,

            trajectories=all_trajs,

            model_names=model_names,

            screenshot_path="picture/pyvista_topdown.tif"
        )

    # 绘制目标点
    goal = env.goal_pos.copy()
    ax.scatter(goal[0], goal[1], goal[2],
               s=100, c='green', marker='*', label="Goal")

    ax.set_xlim(0, env.space_size[0])
    ax.set_ylim(0, env.space_size[1])
    ax.set_zlim(0, env.space_size[2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("UAV Trajectories (3D View)", pad=6)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=len(model_names),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02)
    )
    plt.subplots_adjust(bottom=0.08)

    if save_fig_path is not None:
        plt.savefig(save_fig_path, dpi=300, bbox_inches="tight")
        print(f"Saved 3D figure to: {save_fig_path}")

    plt.show()

    # =======================
    # 俯视图
    # =======================
    if save_topdown_fig_path is not None:
        fig2 = plt.figure(figsize=(9, 9))
        ax2 = fig2.add_subplot(111, projection="3d")
        env._draw_obstacles(ax=ax2)

        for i, traj in enumerate(all_trajs):
            #ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2],linewidth=2.5,color=colors[i],label=model_names[i])
            line = Line3D(traj[:, 0],traj[:, 1],traj[:, 2],linewidth=2.5,color=colors[i])
            ax2.add_line(line)
            
            if i == 0:
                ax2.scatter(traj[0, 0], traj[0, 1], traj[0, 2],
                            s=60, c="blue", marker="o")

            ax2.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2],
                        s=50, marker="^", color=colors[i])

        ax2.scatter(goal[0], goal[1], goal[2],
                    s=100, c="green", marker="*", label="Goal")

        ax2.view_init(elev=90, azim=-90)
        ax2.set_proj_type("ortho")

        ax2.set_xlim(0, env.space_size[0])
        ax2.set_ylim(0, env.space_size[1])
        ax2.set_zlim(0, env.space_size[2])
        ax2.set_box_aspect((env.space_size[0], env.space_size[1], 1.0))

        ax2.set_zticks([])
        ax2.set_zlabel("")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_title("UAV Trajectories (Top-down View)", pad=6)

        fig2.legend(
            loc="lower center",
            ncol=len(model_names),
            frameon=False,
            bbox_to_anchor=(0.5, -0.02)
        )
        plt.subplots_adjust(bottom=0.08)

        plt.savefig(save_topdown_fig_path, dpi=300, bbox_inches="tight")
        print(f"Saved top-down figure to: {save_topdown_fig_path}")
        plt.show()

    # =======================
    # 输出量化指标表格
    # =======================
    df_metrics = pd.DataFrame(traj_metrics)
    print("\n=== Trajectory Quantitative Metrics ===")
    print(df_metrics)
    df_metrics.to_csv("trajectory_metrics.csv", index=False)
    print("Saved metrics table to: trajectory_metrics.csv")

    env.close()

def visualize_multiple_models_pyvista(
    env,
    trajectories,
    model_names,
    screenshot_path=None
):

    import pyvista as pv

    plotter = pv.Plotter(
        window_size=(1600, 1200),
        off_screen=True
    )

    plotter.set_background("white")

    # =========================
    # 绘制建筑
    # =========================

    for cube, color in env.obstacles:

        min_x, max_x, min_y, max_y, min_z, max_z = cube

        box = pv.Box(bounds=(
            min_x, max_x,
            min_y, max_y,
            min_z, max_z
        ))

        #plotter.add_mesh(
                #box,

                #color='gray',

                #opacity=0.85,

                #show_edges=True,

                #edge_color='black',
                
                #line_width=1.5,

                #smooth_shading=False
            #)
        gray_value = np.random.uniform(0.15, 0.35)

        plotter.add_mesh(
            box,

            color=(gray_value, gray_value, gray_value),

            opacity=1.0,

            show_edges=True,

            edge_color='lightgray',

            line_width=0.8,

            smooth_shading=True,

            pbr=True,

            metallic=0.05,

            roughness=0.8
        )

    # ====================================
    # 绘制禁飞区
    # ====================================

    for nfz in env.no_fly_zones:

        min_x, max_x, min_y, max_y, min_z, max_z = nfz

        nfz_box = pv.Box(bounds=(
            min_x, max_x,
            min_y, max_y,
            min_z, max_z
        ))

        plotter.add_mesh(
            nfz_box,

            color=(1.0, 0.5, 0.5),

            opacity=0.25,

            show_edges=True,

            edge_color='red',

            line_width=2.0,

            smooth_shading=False
        )

    # =========================
    # 轨迹颜色
    # =========================

    colors = [
        'red',
        'blue',
        'green',
        'purple',
        'orange'
    ]

    # =========================
    # 绘制轨迹
    # =========================

    for i, traj in enumerate(trajectories):

        line = pv.lines_from_points(
            traj
        )

        plotter.add_mesh(
            line,

            color=colors[i % len(colors)],

            line_width=5,

            render_lines_as_tubes=True,

            label=model_names[i]
        )
        

        # 起点
        start_sphere = pv.Sphere(
            radius=0.3,
            center=traj[0]
        )

        plotter.add_mesh(
            start_sphere,

            color=colors[i % len(colors)]
        )

        # 终点
        end_sphere = pv.Sphere(
            radius=0.4,
            center=traj[-1]
        )

        plotter.add_mesh(
            end_sphere,

            color=colors[i % len(colors)]
        )

    # =========================
    # 目标点
    # =========================

    goal = pv.Sphere(
        radius=0.6,
        center=env.goal_pos
    )

    plotter.add_mesh(
        goal,

        color='limegreen'
    )

    # =========================
    # 坐标轴
    # =========================

    plotter.show_axes()

    plotter.show_grid(
            color='gray',

            xtitle='X',
            ytitle='Y',
            ztitle='Z',

            font_size=14,

            grid='back',

            location='outer',

            ticks='outside'
        )

   

    plotter.add_legend(
        labels=[
            ['—— SAC', 'red'],
            ['—— SAC-CA', 'blue'],
            ['—— DSAC', 'green'],
            ['—— AD-SAC-UA', 'purple']
        ],

        bcolor='white',

        border=True,

        size=(0.18, 0.15),

        loc='upper left'
    )

    #plotter.add_legend(
            #bcolor='white',
            #border=True,
            #size=(0.18, 0.18)
        #)

    # plotter.enable_parallel_projection()  # 可选：不要透视失真（更论文风）

    # 1. 拉伸 Z 轴（视觉上更高）
    # z_scale = 1.8
    # plotter.set_scale(xscale=1.0, yscale=1.0, zscale=z_scale, reset_camera=False)

    cx = env.space_size[0] / 2
    cy = env.space_size[1] / 2
    cz = env.space_size[2] / 2

    plotter.camera_position = [
        # 相机位置（侧上方）
        (cx + 0.7 * env.space_size[0],
        cy - 0.9 * env.space_size[1],
        cz + 2.2 * env.space_size[2]),

        # 看向中心
        (cx, cy, cz),

        # “向上方向”（关键）
        (0, 0, 1)
    ]
    # =========================
    # 保存
    # =========================

    if screenshot_path is not None:

        plotter.show(auto_close=False)
        plotter.render()
        plotter.reset_camera()
        plotter.camera.zoom(1.2)
        plotter.screenshot(screenshot_path)
        print(f"Saved screenshot to: {screenshot_path}")
        plotter.close()

    else:

        plotter.show()
    
def visualize_topdown_pyvista(
        env,
        trajectories,
        model_names,
        screenshot_path=None
    ):
        plotter = pv.Plotter(
            window_size=(1400, 1400),
            off_screen=True
        )

        plotter.set_background("white")

        # ====================================
        # 地面
        # ====================================

        ground = pv.Plane(
            center=(
                env.space_size[0] / 2,
                env.space_size[1] / 2,
                0
            ),

            direction=(0, 0, 1),

            i_size=env.space_size[0],

            j_size=env.space_size[1]
        )

        plotter.add_mesh(
            ground,

            color='whitesmoke',

            opacity=1.0
        )   

        # ====================================
        # 建筑物（仅底面投影）
        # ====================================

        for cube, color in env.obstacles:

            min_x, max_x, min_y, max_y, min_z, max_z = cube

            box = pv.Box(bounds=(
                min_x, max_x,
                min_y, max_y,
                0, 0.5
            ))

            gray_value = np.random.uniform(
                0.2,
                0.4
            )

            plotter.add_mesh(
                box,

                color=(gray_value, gray_value, gray_value),

                opacity=1.0,

                show_edges=True,

                edge_color='black',

                line_width=1.0
            )

        # ====================================
        # 禁飞区（Top-down）
        # ====================================

        for nfz in env.no_fly_zones:

            min_x, max_x, min_y, max_y, min_z, max_z = nfz

            # 二维投影
            nfz_box = pv.Box(bounds=(
                min_x, max_x,
                min_y, max_y,
                0,
                0.2
            ))

            plotter.add_mesh(
                nfz_box,

                color=(1.0, 0.6, 0.6),

                opacity=0.22,

                show_edges=True,

                edge_color='red',

                line_width=2.0,

                smooth_shading=False
            )

        # ====================================
        # 路径颜色
        # ====================================

        colors = [
            'red',
            'blue',
            'green',
            'purple',
            'orange'
        ]

        # ====================================
        # 绘制轨迹
        # ====================================

        for i, traj in enumerate(trajectories):

            # 强制压到 z=1 平面
            traj_2d = traj.copy()

            traj_2d[:, 2] = 1.0

            line = pv.lines_from_points(
                traj_2d
            )

            plotter.add_mesh(
                line,

                color=colors[i % len(colors)],

                line_width=6,

                render_lines_as_tubes=True,

                label=model_names[i]
            )

            # 起点
            start = pv.Sphere(
                radius=0.3,
                center=traj_2d[0]
            )

            plotter.add_mesh(
                start,

                color=colors[i % len(colors)]
            )

            # 终点  
            end = pv.Sphere(
                radius=0.4,
                center=traj_2d[-1]
            )

            plotter.add_mesh(
                end,

                color=colors[i % len(colors)]
            )

        # ====================================
        # 目标点
        # ====================================

        goal_center = env.goal_pos.copy()

        goal_center[2] = 1.2

        goal = pv.Sphere(
            radius=0.6,
            center=goal_center
        )

        plotter.add_mesh(
            goal,

            color='limegreen'
        )

        # ====================================
        # 网格
        # ====================================

        plotter.show_grid(
            color='gray',

                xtitle='X',
            ytitle='Y',
            ztitle='',

            font_size=14,

            grid='back',

            location='outer',

            ticks='outside'
        )

        # ====================================
        # Legend
        # ====================================

        #plotter.add_legend(
            #bcolor='white',
            #border=True,
            #size=(0.18, 0.18)
        #)

        plotter.add_legend(
            labels=[
                ['—— SAC', 'red'],
                ['—— SAC-CA', 'blue'],
                ['—— DSAC', 'green'],
                ['—— AD-SAC-UA', 'purple']
            ],

            bcolor='white',

            border=True,

            size=(0.18, 0.15),

            loc='upper left'
        )

        # ====================================
        # 顶视角（关键）
        # ====================================

        cx = env.space_size[0] / 2
        cy = env.space_size[1] / 2
        cz = env.space_size[2]

        plotter.camera_position = [
            (cx, cy, cz + 200),   # 从Z轴上方看下来（高度拉大）
            (cx, cy, 0),
            (0, 1, 0)
        ]

        # 正交投影（关键）
        plotter.enable_parallel_projection()

        # ====================================
        # 保存
        # ====================================

        if screenshot_path is not None:

            plotter.show(auto_close=False)
            plotter.render()
            plotter.reset_camera()
            plotter.camera.zoom(1.25)
            plotter.screenshot(
                screenshot_path
            )

            print(
                f"Saved topdown view to: {screenshot_path}"
            )

            plotter.close()

        else:

            plotter.show()

def render_city_pyvista(
        env,
        window_size=(1200, 1200),
        screenshot_path=None
    ):

    """
    仅渲染城市环境
    """

    # ====================================
    # Plotter
    # ====================================

    plotter = pv.Plotter(
        window_size=window_size,
        off_screen=True
    )

    plotter.set_background("white")

    # ====================================
    # 地面
    # ====================================

    ground = pv.Plane(

        center=(
            env.space_size[0] / 2,
            env.space_size[1] / 2,
            0
        ),

        direction=(0, 0, 1),

        i_size=env.space_size[0],

        j_size=env.space_size[1]
    )

    plotter.add_mesh(

        ground,

        color='whitesmoke',

        opacity=1.0
    )

    # ====================================
    # 建筑物
    # ====================================

    for cube, color in env.obstacles:

        min_x, max_x, min_y, max_y, min_z, max_z = cube

        box = pv.Box(bounds=(
            min_x, max_x,
            min_y, max_y,
            min_z, max_z
        ))

        gray_value = np.random.uniform(
            0.15,
            0.35
        )

        plotter.add_mesh(

            box,

            color=(
                gray_value,
                gray_value,
                gray_value
            ),

            opacity=1.0,

            show_edges=True,

            edge_color='lightgray',

            line_width=0.8,

            smooth_shading=True,

            pbr=True,

            metallic=0.05,

            roughness=0.8
        )

    # ====================================
    # 禁飞区
    # ====================================

    for nfz in env.no_fly_zones:

        min_x, max_x, min_y, max_y, min_z, max_z = nfz

        nfz_box = pv.Box(bounds=(

            min_x, max_x,

            min_y, max_y,

            min_z, max_z
        ))

        plotter.add_mesh(

            nfz_box,

            color=(1.0, 0.6, 0.6),

            opacity=0.20,

            show_edges=True,

            edge_color='red',

            line_width=2.0,

            smooth_shading=False
        )

        # NFZ 标签
        center = (
            (min_x + max_x) / 2,
            (min_y + max_y) / 2,
            env.space_size[2] * 0.9
        )

        plotter.add_point_labels(

            [center],

            ['NFZ'],

            font_size=18,

            text_color='darkred',

            point_size=0,

            shape_opacity=0
        )

    # ====================================
    # 起点
    # ====================================

    start = pv.Sphere(

        radius=0.4,

        center=env.start_pos
    )

    plotter.add_mesh(

        start,

        color='dodgerblue'
    )

    # ====================================
    # 目标点
    # ====================================

    goal = pv.Sphere(

        radius=0.5,

        center=env.goal_pos
    )

    plotter.add_mesh(

        goal,

        color='limegreen'
    )

    # ====================================
    # 坐标轴
    # ====================================

    plotter.show_axes()

    plotter.show_grid(

        color='gray',

        xtitle='X',
        ytitle='Y',
        ztitle='Z',

        font_size=14,

        grid='back',

        location='outer',

        ticks='outside'
    )

    # ====================================
    # 相机
    # ====================================

    plotter.camera_position = [

        (
            env.space_size[0] * 2.0,
            -env.space_size[1] * 2.2,
            env.space_size[2] * 1.8
        ),

        (
            env.space_size[0] / 2,
            env.space_size[1] / 2,
            env.space_size[2] / 2
        ),

        (0, 0, 1)
    ]


    # ====================================
    # 标题
    # ====================================

    plotter.add_text(

        "Urban Environment",

        position='upper_edge',

        font_size=20,

        color='black'
    )

    # ====================================
    # 保存
    # ====================================

    if screenshot_path is not None:

        plotter.screenshot(
            screenshot_path
        )

        print(
            f"Saved city environment to: {screenshot_path}"
        )

        plotter.close()

    else:

        plotter.show()

# ===========================
# 主函数
# ===========================

if __name__ == "__main__":
    ckpt_paths = [
        "./testmodels/AD-SAC-UA.pth",
        "./testmodels/SAC.pth",
        "./testmodels/SAC-UA.pth",
        "./testmodels/SD-SAC-UA.pth"
    ]

    model_names = [
        "AD-SAC-UA",
        "SAC",
        "DSAC",
        "SAC-CA",
    ]

    visualize_multiple_models(
        ckpt_paths=ckpt_paths,
        model_names=model_names,
        curriculum_level=4,
        max_steps=300,
        save_fig_path="./picture/traj_3d_view.tif",
        save_topdown_fig_path="./picture/traj_topdown_view.tif",
    )