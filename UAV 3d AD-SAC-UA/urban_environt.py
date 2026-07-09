from UAV_env import Drone3DEnv
import pyvista as pv
import numpy as np

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
            env.space_size[0] * 1.6,
            -env.space_size[1] * 1.9,
            env.space_size[2] * 4
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
if __name__ == "__main__":

    env = Drone3DEnv(curriculum_level=4)
    env.load_map("map/map_level9.pkl")

    render_city_pyvista(
                env=env,
                screenshot_path="map/map_level9.tif"
            )