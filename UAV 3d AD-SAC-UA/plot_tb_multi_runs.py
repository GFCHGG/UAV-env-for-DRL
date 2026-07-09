import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# =========================
# 读取 scalar
# =========================
def load_scalar_from_run(run_dir, tag):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()

    if tag not in ea.Tags()["scalars"]:
        raise ValueError(
            f"[ERROR] Tag '{tag}' not found in {run_dir}\n"
            f"Available tags: {ea.Tags()['scalars']}"
        )

    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


# =========================
# 平滑（EMA，推荐）
# =========================
# smoother
def smooth_curve(x):
    x = np.array(x)

    ema_alpha = 0.02
    window = 30

    ema = np.zeros_like(x)
    ema[0] = x[0]

    for i in range(1, len(x)):
        ema[i] = ema_alpha * x[i] + (1 - ema_alpha) * ema[i-1]

    ema = np.convolve(ema, np.ones(window)/window, mode='same')
    return ema

# =========================
# ⭐统一绘图函数（核心）
# =========================
def plot_line(ax, x, y, color, lw=2, use_glow=False, label=None):
    if use_glow:
        ax.plot(x, y,
                color=color,
                linewidth=lw * 4,
                alpha=0.06,
                zorder=1)

        ax.plot(x, y,
                color=color,
                linewidth=lw * 2.5,
                alpha=0.12,
                zorder=2)

        ax.plot(x, y,
                color=color,
                linewidth=lw,
                alpha=1.0,
                label=label,
                zorder=3)
    else:
        ax.plot(x, y,
                color=color,
                linewidth=lw,
                label=label,
                zorder=3)


# =========================
# 主绘图函数
# =========================
def plot_multi_runs(
    root_log_dir,
    tag,
    smooth_window=1,
    save_path=None,
    use_glow=False,   # ⭐全局开关
):
    run_dirs = [
        os.path.join(root_log_dir, d)
        for d in os.listdir(root_log_dir)
        if os.path.isdir(os.path.join(root_log_dir, d))
    ]

    assert len(run_dirs) > 0, "No run directories found!"

    use_broken_axis = (tag == "Train/avg_reward_100")

    if use_broken_axis:
        fig, (ax_high, ax_low) = plt.subplots(
            2, 1,
            sharex=True,
            figsize=(9, 7),
            gridspec_kw={"height_ratios": [4, 1]}
        )
    else:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # =========================
    # 画每个 run
    # =========================
    plt.style.use("seaborn-v0_8-white")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    run_dirs = sorted(run_dirs)

    for i,run_dir in enumerate(run_dirs):
        run_name = os.path.basename(run_dir)
        color = colors[i % len(colors)]

        try:
            steps, values = load_scalar_from_run(run_dir, tag)
        except Exception as e:
            print(e)
            continue

        # EMA smoothing
        raw_values = values
        smooth_values = smooth_curve(values)

        if use_broken_axis:
            # ===== low axis =====
            if use_glow:
                ax_low.plot(steps, raw_values, color=color, alpha=0.08, linewidth=2)
                ax_low.plot(steps, smooth_values, color=color, linewidth=2)
            else:
                ax_low.plot(steps, raw_values, color=color, linewidth=1.2, label=run_name)

            # ===== high axis =====
            if use_glow:
                ax_high.plot(steps, raw_values, color=color, alpha=0.08, linewidth=2)
                ax_high.plot(steps, smooth_values, color=color, linewidth=2)
            else:
                ax_high.plot(steps, raw_values, color=color, linewidth=1.2, label=run_name)
        else:
                if use_glow:
                    ax.plot(steps, raw_values, color=color, alpha=0.15, linewidth=8,zorder=1)
                    ax.plot(steps, smooth_values, color=color, alpha=0.8,linewidth=1, label=run_name,zorder=3)
                else:
                    ax.plot(steps, raw_values, color=color, linewidth=1.2, label=run_name)

    # =========================
    # 坐标设置
    # =========================
    if use_broken_axis:
        ax_low.set_ylim(-2000, 0)
        ax_high.set_ylim(0, 900)

        ax_low.spines["top"].set_visible(False)
        ax_low.spines["right"].set_visible(False)
        ax_high.spines["bottom"].set_visible(False)
        ax_high.spines["right"].set_visible(False)
        ax_high.spines["top"].set_visible(False)

        ax_low.tick_params(labeltop=False)
        ax_high.tick_params(labelbottom=True)

        d = 0.008
        kwargs = dict(color="k", clip_on=False)

        ax_low.plot((-d, +d), (1 - d, 1 + d),
                    transform=ax_low.transAxes, **kwargs)
        ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d),
                    transform=ax_low.transAxes, **kwargs)

        ax_high.plot((-d, +d), (-d, +d),
                     transform=ax_high.transAxes, **kwargs)
        ax_high.plot((1 - d, 1 + d), (-d, +d),
                     transform=ax_high.transAxes, **kwargs)

        ax_high.set_ylabel("Avg Reward")
        ax_low.set_xlabel("Training Steps")

        ax_high.grid(True, alpha=0.2, linestyle='--', linewidth=0.7)
        ax_low.grid(True, alpha=0.2, linestyle='--', linewidth=0.7)
        ax_high.legend()

    else:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='in')
        ax.set_xlabel("Training Steps",fontsize=12)
        ax.set_ylabel("Delay Update Steps",fontsize=12)
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.7)
        ax.legend()

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[Saved] {save_path}")

    plt.show()
    plt.close()


# =========================
# main
# =========================
if __name__ == "__main__":
    # plot_multi_runs(
    #     root_log_dir="./ulogs",
    #     tag="Train/avg_reward_100",
    #     smooth_window=10,
    #     save_path="./picture/fig_avg_reward.tif",
    #     use_glow=False   # 开启虚影+主线
    # )
    # plot_multi_runs(
    #     root_log_dir="./ulogs",
    #     tag="Alpha/value",
    #     smooth_window=5,
    #     save_path="./picture/fig_alpha.tif",
    #     use_glow=False
    #     )
    # plot_multi_runs(
    #     root_log_dir="./ulogs",
    #     tag="Uncertainty/ema",
    #     smooth_window=5,
    #     save_path="./picture/fig_uncertainty.tif",
    #     use_glow=False
    # )
    # plot_multi_runs(
    #     root_log_dir="./ulogs",
    #     tag="Loss/actor",
    #     smooth_window=5,
    #     save_path="./picture/fig_actor_loss.tif",
    #     use_glow=False
    # )
    # plot_multi_runs(
    #     root_log_dir="./ulogs",
    #     tag="Stability/critic_loss_ema",
    #     smooth_window=5,
    #     save_path="./picture/fig_critic_loss_ema.tif",
    #     use_glow=False
    # )
    plot_multi_runs(
        root_log_dir="./ulogs",
        tag="Stability/policy_delay",
        smooth_window=5,
        save_path="./picture/fig_policy_delay.tif",
        use_glow=False
    )
    # plot_multi_runs(
    #     root_log_dir="./ulogs",
    #     tag="Stability/tau",
    #     smooth_window=5,
    #     save_path="./picture/fig_tau.tif",
    #     use_glow=False
    # )
    # plot_multi_runs(
    #     root_log_dir="./ulogs",
    #     tag="Target_Q/bellman_target_var",
    #     smooth_window=5,
    #     save_path="./picture/fig_bellman_target_var.tif",
    #     use_glow=False
    # )
    # plot_multi_runs(
    #     root_log_dir="./ulogs",
    #     tag="Target_Q/bellman_target_std",
    #     smooth_window=5,
    #     save_path="./picture/fig_bellman_target_std.tif",
    #     use_glow=False
    # )
    # plot_multi_runs(
    #     root_log_dir="./ulogs",
    #     tag="Target_Q/bellman_target_mean",
    #     smooth_window=5,
    #     setave_path="./picture/fig_bellman_target_mean.tif",
    #     use_glow=False
    # )