import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import gymnasium as gym
from gymnasium import spaces  
from typing import Tuple, List, Dict, Any, Optional
from collections import deque
import random
from mpl_toolkits.mplot3d.art3d import Line3D
import pickle

class Drone3DEnv(gym.Env):
    """
    3D无人机路径规划环境
    无人机需要从起点导航到终点，避开三维障碍物（从底部开始的长方体）
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self,
                 space_size: Tuple[int, int, int] = (50, 50, 20),
                 max_steps: int = 300,
                 num_obstacles: int = 10,
                 min_building_height: float = 2.0,
                 use_global_goal: bool = True,
                 collision_radius: float = 0.5,
                 max_velocity: float = 1.0,
                 curriculum_level: int = 0,
                 **kwargs):
        super().__init__(**kwargs)
        
        # 环境参数
        self.space_size = space_size
        self.max_steps = max_steps
        self.num_obstacles = num_obstacles
        self.min_building_height = min_building_height
        self.use_global_goal = use_global_goal
        self.collision_radius = collision_radius
        self.max_velocity = max_velocity
        self.curriculum_level = curriculum_level  # 新增参数
        self.no_fly_zones = []

        # 动作空间: [vx, vy, vz] 三维速度向量 (连续动作)
        self.action_space = spaces.Box(
            low=-max_velocity, 
            high=max_velocity, 
            shape=(3,), 
            dtype=np.float32
        )
        
        # 观察空间
        if use_global_goal:
            # [drone_x, drone_y, drone_z, goal_x, goal_y, goal_z, vx, vy, vz] + 局部观测
            obs_dim = 6 + 26  # 9个全局状态 + 26个局部观测
        else:
            obs_dim = 3 + 26  # 3个速度状态 + 26个局部观测
            
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # 环境状态
        self.drone_pos = None
        self.drone_velocity = None
        self.goal_pos = None
        self.obstacles = None  # 只保留长方体障碍物
        self.steps = 0
        self.space = None
        self.start_pos = None
        self.hit_count = 0
      
        # 渲染相关
        self.fig = None
        self.ax = None
        self.drone_marker = None
        self.goal_marker = None
        self.trajectory = None
        
        # 初始化空间
        self.space = np.zeros(self.space_size, dtype=int)
        
        
        
        # 生成障碍物（在初始化时只生成一次）
        self._generate_obstacles()
        
        #生成禁飞区
        self._generate_no_fly_zones()

        # 放置起点和终点
        self._place_start_goal()

        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.drone_pos = self.start_pos
        self.drone_velocity = np.zeros(3, dtype=np.float32)
        self.steps = 0
        self.hit_count = 0
        #生成禁飞区
        self._generate_no_fly_zones()
        self.prev_distance = np.linalg.norm(self.start_pos - self.goal_pos)
        self.trajectory = deque(maxlen=200)
        self.trajectory.append(self.drone_pos.copy())
        return self._get_observation(), {}
    
    def _generate_obstacles(self):
        """生成不会相互重叠的障碍物"""

        # curriculum 配置（保持你原来的逻辑）
        if self.curriculum_level == 0:
            self.num_obstacles = 10
            self.min_building_height = 8.0
        elif self.curriculum_level == 1:
            self.num_obstacles = 15
            self.min_building_height = 8.0
        elif self.curriculum_level == 2:
            self.num_obstacles = 20
            self.min_building_height = 10.0
        elif self.curriculum_level == 3:
            self.num_obstacles = 30
            self.min_building_height = 12.0
        elif self.curriculum_level == 4:
            self.num_obstacles = 35
            self.min_building_height = 14.0
        elif self.curriculum_level == 5:
            self.num_obstacles = 40
            self.min_building_height = 14.0
        elif self.curriculum_level == 6:
            self.num_obstacles = 45
            self.min_building_height = 14.0
        elif self.curriculum_level == 7:
            self.num_obstacles = 50
            self.min_building_height = 14.0

        self.obstacles = []

        def is_overlap(c1, c2) -> bool:
            """检查两个长方体是否重叠（AABB 判断）"""
            min_x1, max_x1, min_y1, max_y1, min_z1, max_z1 = c1
            min_x2, max_x2, min_y2, max_y2, min_z2, max_z2 = c2

            # 三维全部轴向都有重叠才算重叠
            overlap_x = not (max_x1 <= min_x2 or max_x2 <= min_x1)
            overlap_y = not (max_y1 <= min_y2 or max_y2 <= min_y1)
            overlap_z = not (max_z1 <= min_z2 or max_z2 <= min_z1)

            return overlap_x and overlap_y and overlap_z

        # 开始生成
        for _ in range(self.num_obstacles):
            for attempt in range(100):   # 最多尝试 100 次
                base_x = random.uniform(2, self.space_size[0] - 4)
                base_y = random.uniform(2, self.space_size[1] - 4)
                width  = random.uniform(3.0, 4.0)
                length = random.uniform(4.0, 5.0)

                end_x = min(base_x + width, self.space_size[0] - 1)
                end_y = min(base_y + length, self.space_size[1] - 1)
                height = random.uniform(self.min_building_height, self.space_size[2] * 0.9)

                new_cube = [base_x, end_x, base_y, end_y, 0.0, height]

                # 检查与已有障碍物是否重叠
                conflict = False
                for cube, _color in self.obstacles:
                    if is_overlap(new_cube, cube):
                        conflict = True
                        break

                if not conflict:
                    # 找到合法 cube，加入
                    color = (0.0, 0.0, 0.0)
                    self.obstacles.append((new_cube, color))
                    break

            else:
                print("警告：多次尝试后仍未找到不重叠障碍物，跳过生成一个。")

    def _generate_no_fly_zones(self):

        """
        生成多个互不重叠的禁飞区
        """

        self.no_fly_zones = []

        num_nfzs = 2

        # =========================
        # overlap 检测
        # =========================

        def is_overlap(rect1, rect2):

            min_x1, max_x1, min_y1, max_y1, _, _ = rect1
            min_x2, max_x2, min_y2, max_y2, _, _ = rect2

            # 安全间隔（推荐）
            clearance = 1.0

            overlap_x = not (
                max_x1 + clearance <= min_x2 or
                max_x2 + clearance <= min_x1
            )

            overlap_y = not (
                max_y1 + clearance <= min_y2 or
                max_y2 + clearance <= min_y1
            )

            return overlap_x and overlap_y

        # =========================
        # 开始生成 NFZ
        # =========================

        for i in range(num_nfzs):

            success = False

            attempt = 0

            while not success:

                attempt += 1

                # 防止死循环
                if attempt > 50000:

                    print(
                        f"Warning: Failed to generate NFZ {i}"
                    )

                    break

                # =========================
                # 随机尺寸
                # =========================

                width = random.uniform(6.0, 8.0)

                length = random.uniform(6.0, 8.0)

                # =========================
                # 限制生成区域
                # =========================

                margin_start = 5.0
                margin_goal = 5.0

                min_x = random.uniform(
                    margin_start,
                    self.space_size[0] - width - margin_goal
                )

                min_y = random.uniform(
                    margin_start,
                    self.space_size[1] - length - margin_goal
                )

                max_x = min_x + width
                max_y = min_y + length

                new_nfz = [
                    min_x,
                    max_x,
                    min_y,
                    max_y,
                    0.0,
                    self.space_size[2]
                ]

                # =========================
                # 检查 overlap
                # =========================

                overlap = False

                for existing_nfz in self.no_fly_zones:

                    if is_overlap(
                        new_nfz,
                        existing_nfz
                    ):

                        overlap = True
                        break

                # =========================
                # 如果重叠
                # 继续重新生成
                # =========================

                if overlap:
                    continue

                # =========================
                # 合法 NFZ
                # =========================

                self.no_fly_zones.append(
                    new_nfz
                )

                success = True
    def _place_start_goal(self):
        #固定起点和终点，高度 10，避开地面障碍
        self.start_pos = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        self.goal_pos  = np.array([48.0, 48.0, 18.0], dtype=np.float32)
        self.drone_pos = self.start_pos

        # 可选：简单检查是否落在障碍内（高度 10 基本不会碰）
        while self._check_collision(self.start_pos):
            self.start_pos[0] += 0.5   # 微调 x 直到安全
        while self._check_collision(self.goal_pos) or np.linalg.norm(self.start_pos - self.goal_pos) < 10:
            self.goal_pos[0] -= 0.5    # 微调 x 直到安全

    def _check_collision(self, position: np.ndarray) -> bool:
        """检查位置是否与障碍物碰撞"""
        # 检查边界碰撞
        if (position[0] < 0 or position[0] >= self.space_size[0] or
            position[1] < 0 or position[1] >= self.space_size[1] or
            position[2] < 0 or position[2] >= self.space_size[2]):
            return True
        
        # 检查长方体障碍物碰撞
        for cube_info in self.obstacles:
            cube, color = cube_info
            min_x, max_x, min_y, max_y, min_z, max_z = cube
            if (min_x - self.collision_radius <= position[0] <= max_x + self.collision_radius and
                min_y - self.collision_radius <= position[1] <= max_y + self.collision_radius and
                min_z - self.collision_radius <= position[2] <= max_z + self.collision_radius):
                return True
        
        # ======================
        # 禁飞区碰撞
        # ======================

        for nfz in self.no_fly_zones:

            min_x, max_x, min_y, max_y, min_z, max_z = nfz

            if (
                min_x <= position[0] <= max_x and
                min_y <= position[1] <= max_y
            ):
                return True

        return False
    
    def step(self, action: np.ndarray):
        """执行一步"""
        self.steps += 1
        old_pos = self.drone_pos.copy()
        
        # 更新速度
        self.drone_velocity = 0.7 * self.drone_velocity + 0.3 * np.clip(
            action, -self.max_velocity, self.max_velocity
        )
        
        new_pos = self.drone_pos + self.drone_velocity
        collision = self._check_collision(new_pos)
        
        # 改进的碰撞处理
        if collision:
            self.drone_pos = old_pos
            self.drone_velocity *= 0.1  # 衰减而非反向
            self.hit_count += 1
        else:
            self.drone_pos = new_pos
    
        # 简化的奖励计算
        distance_to_goal = np.linalg.norm(self.drone_pos - self.goal_pos)
        prev_distance = getattr(self, 'prev_distance', distance_to_goal)
    
        reward = 0.0
    
        # 1. 距离改进奖励 (主要驱动力)
        reward += (prev_distance - distance_to_goal) * 10.0
        
        # 2. 碰撞惩罚 (明确且恒定)
        if collision:
            reward -= 10.0
        
        # 3. 到达目标奖励 (降低数值)
        if distance_to_goal < 1.0:  # 收紧阈值
            reward += 100.0
    
        # 4. 时间惩罚 (鼓励快速到达)
        reward -= 2
        
        # 5. 避障奖励 (仅在接近障碍时)
        local_obs = self._get_local_observation()
        min_distance = np.min(local_obs)
        if min_distance < 0.1:  # 极度危险
            reward -= 3.0
        elif min_distance < 0.2:  # 危险
            reward -= 1.0
        elif min_distance < 0.3:  # 接近但安全
            reward -= 0.5  # 奖励保持安全距离
    
        self.prev_distance = distance_to_goal
        self.trajectory.append(self.drone_pos.copy())
    
        info = {
        'distance_to_goal': distance_to_goal,
        'steps': self.steps,
        'collision': collision,
        'hit_count': self.hit_count,
        'success': distance_to_goal < 1.0
        }

        terminated = distance_to_goal < 1.0
        
        truncated = self.steps >= self.max_steps

        return self._get_observation(), reward, terminated, truncated, info


    
    def _get_observation(self):

        goal_relative = (
            self.goal_pos - self.drone_pos
        ) / np.array(self.space_size)

        velocity_norm = (
            self.drone_velocity / self.max_velocity
        )
        
        local_obs = self._get_local_observation()

        obs = np.concatenate([
            goal_relative,
            velocity_norm,
            local_obs
        ])

        return obs.astype(np.float32)
    
    def _get_local_observation(self) -> np.ndarray:
        """获取局部距离观测（26个方向）"""
        local_obs = []
        
        # 26个方向: 三维网格的邻域 (-1, 0, 1) 在每个维度
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue  # 跳过自身方向
                    
                    direction = np.array([dx, dy, dz], dtype=np.float32)
                    direction = direction / np.linalg.norm(direction)  # 归一化
                    
                    distance = self._ray_cast(self.drone_pos, direction)
                    local_obs.append(distance)
        
        return np.array(local_obs, dtype=np.float32)
    
    def _ray_cast(self, position: np.ndarray, direction: np.ndarray) -> float:
        """射线投射，检测障碍物距离"""
        max_distance = 8.0
        step_size = 0.1
        distance = 0
        
        while distance < max_distance:
            distance += step_size
            test_point = position + direction * distance
            
            if self._check_collision(test_point):
                return distance / max_distance  # 归一化距离
        
        return 1.0  # 最大距离
    
    def export_map(self):

        return {
            "obstacles": self.obstacles,
            "no_fly_zones": self.no_fly_zones,
            "start_pos": self.start_pos.copy(),
            "goal_pos": self.goal_pos.copy(),
            "space_size": self.space_size,
            "curriculum_level": self.curriculum_level
        }

    def import_map(self, map_data):

        self.obstacles = map_data["obstacles"]

        self.no_fly_zones = map_data["no_fly_zones"]

        self.start_pos = map_data["start_pos"].copy()

        self.goal_pos = map_data["goal_pos"].copy()

        self.space_size = map_data["space_size"]

        self.curriculum_level = map_data["curriculum_level"]

    def save_map(self, filepath):

        map_data = self.export_map()

        with open(filepath, "wb") as f:

            pickle.dump(map_data, f)

        print(f"Map saved to {filepath}")

    def load_map(self, filepath):

        with open(filepath, "rb") as f:

            map_data = pickle.load(f)

        self.import_map(map_data)

        print(f"Map loaded from {filepath}")    

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """渲染3D环境"""
        if self.fig is None:
            self.fig = plt.figure(figsize=(12, 10))
            self.ax = self.fig.add_subplot(111, projection='3d')
            
        self.ax.clear()
        
        # 设置坐标轴范围
        self.ax.set_xlim(0, self.space_size[0])
        self.ax.set_ylim(0, self.space_size[1])
        self.ax.set_zlim(0, self.space_size[2])
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        # 绘制障碍物
        self._draw_obstacles()
        
        # 绘制无人机
        self.ax.scatter(*self.drone_pos, color='blue', s=100, label='Drone')
        
        # 绘制目标
        self.ax.scatter(*self.goal_pos, color='green', s=100, marker='*', label='Goal')
        
        # 绘制轨迹
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            line = Line3D(traj[:, 0],traj[:, 1],traj[:, 2],color='red',linewidth=3,alpha=1.0)


            self.ax.add_line(line)
            #self.ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r-', alpha=0.6, linewidth=2, label='Trajectory')
        
        # 绘制速度向量
        velocity_scale = 2.0
        # 计算速度向量终点（用于后续可视化扩展，当前未使用）
        velocity_end = self.drone_pos + self.drone_velocity * velocity_scale
        self.ax.quiver(*self.drone_pos, *(self.drone_velocity * velocity_scale), 
                      color='orange', arrow_length_ratio=0.2, linewidth=2, label='Velocity')
        
        self.ax.set_title(f'3D Drone Path Planning - Steps: {self.steps}')
        self.ax.legend()
        
        if mode == 'human':
            plt.pause(0.01)
        elif mode == 'rgb_array':
            self.fig.canvas.draw()
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return img
    
    def _draw_obstacles(self, ax=None):
        """绘制3D障碍物（只有长方体）"""
        if ax is None:
            ax = self.ax
        # 绘制长方体障碍物
        for cube, color in self.obstacles:
            min_x, max_x, min_y, max_y, min_z, max_z = cube
            
            # 定义立方体的8个顶点
            vertices = [
                [min_x, min_y, min_z],
                [max_x, min_y, min_z],
                [max_x, max_y, min_z],
                [min_x, max_y, min_z],
                [min_x, min_y, max_z],
                [max_x, min_y, max_z],
                [max_x, max_y, max_z],
                [min_x, max_y, max_z]
            ]
            
            # 定义立方体的6个面
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
                [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
                [vertices[0], vertices[3], vertices[7], vertices[4]]   # 左面
            ]
            
            
            
            # 绘制立方体
            #ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors='white', alpha=0.8, linewidths=1))
            poly = Poly3DCollection(faces,facecolors='black',edgecolors='white',alpha=0.8,linewidths=1,zsort='average')
            ax.add_collection3d(poly)
            
            # 为了更好地显示底部连接地面，添加底面边框
            ax.plot([min_x, max_x, max_x, min_x, min_x],
                         [min_y, min_y, max_y, max_y, min_y],
                         [min_z, min_z, min_z, min_z, min_z],
                         'k-', linewidth=1.5)
    
    
    def close(self):
        """关闭环境"""
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None