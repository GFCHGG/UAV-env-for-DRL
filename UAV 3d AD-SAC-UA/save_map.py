import os
import pickle
import numpy as np
from UAV_env import Drone3DEnv
import matplotlib.pyplot as plt
import pyvista as pv

# ==========================
# 1. 创建 map 文件夹
# ==========================
map_dir = "./map"
os.makedirs(map_dir, exist_ok=True)

# ==========================
# 2. 创建环境
# ==========================
env = Drone3DEnv(curriculum_level=4)  # 可以调整难度

# ==========================
# 4. 渲染并展示环境
# ==========================
# 使用 matplotlib 渲染初始状态
env.render(mode='human')  # 打开可交互窗口

# ==========================
# 4. 保存地图
# ==========================
map_data = env.export_map()  # 导出障碍物、起点、终点等

map_path = os.path.join(map_dir, "map_level9.pkl")
with open(map_path, "wb") as f:
    pickle.dump(map_data, f)

print(f"地图已保存到: {map_path}")

# 保存图片
image_path = os.path.join(
    map_dir,
    "map_level9.png"
)

env.fig.savefig(
    image_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"环境截图已保存到: {image_path}")

# 保持窗口
plt.show(block=True)

# ==========================
# 5. 测试加载地图
# ==========================
env2 = Drone3DEnv(curriculum_level=4)
with open(map_path, "rb") as f:
    loaded_map = pickle.load(f)

env2.import_map(loaded_map)

# 再渲染一次，确保加载正确
env2.render(mode='human')