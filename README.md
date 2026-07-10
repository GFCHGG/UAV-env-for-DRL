# AD-SAC-UA: Uncertainty-Aware Adaptive-Delay Soft Actor-Critic for 3D UAV Navigation
[![DOI](https://zenodo.org/badge/1294808699.svg)](https://doi.org/10.5281/zenodo.21289289)
Official implementation of **AD-SAC-UA**, a Soft Actor-Critic-based reinforcement learning method for autonomous UAV navigation in a three-dimensional urban environment with obstacles and no-fly zones.

The repository contains the proposed method, two comparison methods, a custom Gymnasium environment, training scripts, evaluation utilities, TensorBoard plotting tools, trajectory visualization code, and a pretrained AD-SAC-UA checkpoint.

> **Paper status:** Replace the citation placeholder in the [Citation](#citation) section with the final bibliographic information after publication.

## Highlights

- Continuous-control 3D UAV navigation with a 3-dimensional action space.
- A 32-dimensional observation composed of relative-goal information, UAV velocity, and 26-direction local obstacle sensing.
- Uncertainty-aware entropy-coefficient adaptation.
- Adaptive actor update delay based on training instability.
- Optional adaptive target-network update coefficient.
- Curriculum-style urban environments with configurable obstacle density.
- Training, repeated evaluation, TensorBoard analysis, and 3D trajectory visualization.
- CPU and CUDA execution through PyTorch.

## Implemented Methods

| Method | Script | Description |
|---|---|---|
| **AD-SAC-UA** | `train/SAC_train.py` | Proposed SAC variant with adaptive policy delay and uncertainty-aware entropy coefficient. |
| **DSAC** | `train/DSAC_train.py` | Distributional Soft Actor-Critic baseline. |
| **SAC-CA** | `train/SAC_CA_train.py` | SAC baseline with clipped/adaptive entropy coefficient. |

The behavior of the main SAC implementation can be controlled in `agent/SAC_agent.py` using:

- `auto_delay`: enables adaptive actor-update delay.
- `auto_tau`: enables adaptive target-network soft-update coefficient when adaptive delay is active.
- `uncertainty_alpha`: uses uncertainty-aware entropy-coefficient adaptation instead of the conventional target-entropy update.
- `init_policy_delay`: sets the fixed delay or initial adaptive delay.

## Repository Structure

```text
.
├── agent/
│   ├── SAC_agent.py              # AD-SAC-UA and configurable SAC variants
│   ├── DSAC_agent.py             # Distributional SAC agent
│   └── SAC_CA_agent.py           # SAC-CA agent
├── env/
│   ├── UAV_env.py                # Custom Gymnasium 3D UAV environment
│   ├── save_map.py               # Map generation and serialization utility
│   └── urban_environt.py         # Additional urban-environment visualization
├── train/
│   ├── SAC_train.py              # AD-SAC-UA training
│   ├── DSAC_train.py             # DSAC training
│   └── SAC_CA_train.py           # SAC-CA training
├── test and visual/
│   ├── eval_multiple_runs.py     # Repeated quantitative evaluation
│   ├── plot_tb_multi_runs.py     # TensorBoard curve plotting
│   └── traj_visual.py            # Matplotlib/PyVista trajectory visualization
├── testmodels/
│   └── AD-SAC-UA.pth             # Pretrained checkpoint used by demo.py
├── demo.py                       # Single-trajectory inference demo
├── requirements.txt
└── README.md
```

Directories such as `logs/`, `checkpoints/`, `data1/`, `picture/`, `map/`, and `ulogs/` are used for generated outputs and may initially be empty.

## Environment

The default environment is a bounded 3D urban flight space:

- Space size: `50 × 50 × 20`
- Maximum episode length: `300` steps
- Continuous action: UAV velocity command `[vx, vy, vz]`
- Action range: `[-1, 1]` for each axis by default
- Observation dimension: `32`
- Local perception: normalized ray-casting distances in 26 neighboring 3D directions
- Success condition: distance to the goal is less than `1.0`
- Curriculum levels: `0` to `7`, with increasing obstacle density

The environment follows the Gymnasium API:

```python
state, info = env.reset()
next_state, reward, terminated, truncated, info = env.step(action)
```

## Installation

### Recommended Setup

Python **3.10 or 3.11** is recommended.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>

python -m venv .venv
```

Activate the virtual environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a specific CUDA version, install the matching PyTorch build according to the official PyTorch installation instructions before installing the remaining packages.

### Headless Linux Servers

PyVista rendering may require an X server or off-screen rendering configuration. Training and non-PyVista evaluation can still be run without opening the PyVista visualizer.

## Quick Start

Run the pretrained AD-SAC-UA model:

```bash
python demo.py
```

The script loads:

```text
testmodels/AD-SAC-UA.pth
```

and saves the trajectory figure as:

```text
uav_trajectory_eval.png
```

The demo creates a new curriculum-level-4 environment at runtime. Because the environment is randomly generated, the displayed map can differ between runs.

## Preparing a Fixed Training Map

The training scripts currently load the following file:

```text
map/tmap_level4.pkl
```

A serialized map is not included in the current source archive. Generate one before training.

Open `env/save_map.py` and change:

```python
map_path = os.path.join(map_dir, "map_level9.pkl")
```

to:

```python
map_path = os.path.join(map_dir, "tmap_level4.pkl")
```

Then run:

```bash
python env/save_map.py
```

This creates a fixed map that can be loaded by all three training scripts. The map contains obstacle geometry, no-fly zones, start and goal positions, environment size, and curriculum level.

For exact reproducibility, generate the map once, commit or archive the resulting `.pkl` file, and use the same file for every compared method.

## Training

All commands below should be executed from the repository root.

### Train AD-SAC-UA

```bash
python train/SAC_train.py
```

Default configuration:

```text
Episodes:                 3000
Replay-buffer capacity:   1,000,000
Batch size:               256
Learning rate:            3e-4
Discount factor:          0.99
Initial/fixed tau:        0.005
Random exploration:       1000 environment steps
Checkpoint interval:      500 episodes
Adaptive policy delay:    enabled
Adaptive tau:             disabled
Uncertainty-aware alpha:  enabled
```

Outputs:

```text
logs/drone_sac/AD_SAC_UA/
checkpoints/drone_sac/AD_SAC_UA/
```

### Train DSAC

```bash
python train/DSAC_train.py
```

Outputs:

```text
logs/drone_sac/DSAC/
checkpoints/drone_sac/DSAC/
```

### Train SAC-CA

```bash
python train/SAC_CA_train.py
```

Outputs:

```text
logs/drone_sac/SAC_CA/
checkpoints/drone_sac/SAC_CA/
```

### Monitor Training

```bash
tensorboard --logdir logs
```

Then open the local address printed by TensorBoard in a browser.

Important logged quantities include episode reward, 100-episode average reward, distance to goal, collision count, actor and critic losses, entropy coefficient, uncertainty, policy delay, target-update coefficient, and Bellman-target statistics.

## Evaluation

### Single-Trajectory Evaluation

```bash
python demo.py
```

To evaluate another checkpoint, edit `ckpt_path` in `demo.py` and make sure the selected agent configuration matches the configuration used when the checkpoint was saved.

### Repeated Quantitative Evaluation

```bash
python "test and visual/eval_multiple_runs.py"
```

Before running, edit these fields in the script:

- Agent class and its constructor settings.
- `ckpt_path`.
- Number of runs.
- Output CSV path.

The evaluator reports and stores:

- Success rate
- Collision count
- Episode steps
- Final reward

By default, each run creates a new random curriculum-level-4 environment. For controlled comparisons on identical maps, modify the evaluator to call `env.load_map(...)` after constructing the environment or reuse a fixed set of serialized maps.

## Visualization

### TensorBoard Curves

Edit the active `plot_multi_runs(...)` call at the bottom of:

```text
test and visual/plot_tb_multi_runs.py
```

Then run:

```bash
python "test and visual/plot_tb_multi_runs.py"
```

### Multi-Model Trajectories

```bash
python "test and visual/traj_visual.py"
```

The script supports Matplotlib and PyVista visualizations, including 3D and top-down views. Update `ckpt_paths` and `model_names` in the script before execution. The current archive contains only `testmodels/AD-SAC-UA.pth`; additional comparison checkpoints must be supplied separately.

## Using the Environment in New Code

```python
from env.UAV_env import Drone3DEnv


env = Drone3DEnv(curriculum_level=4)
state, info = env.reset(seed=42)

done = False
while not done:
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

Load a fixed map with:

```python
env = Drone3DEnv(curriculum_level=4)
env.load_map("map/tmap_level4.pkl")
```

## Reproducibility Notes

The current code does not globally seed Python `random`, NumPy, and PyTorch in the training scripts. For repeatable experiments, add a helper similar to:

```python
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)
```

Also pass the seed to `env.reset(seed=42)` and preserve the exact serialized maps used in the experiments. For publication results, report the seeds, hardware, dependency versions, number of runs, and model-selection rule.

## Checkpoint Compatibility

A checkpoint must be loaded with the corresponding agent class and compatible network/configuration parameters. In particular, the pretrained `AD-SAC-UA.pth` checkpoint is intended for `SACAgent` with the configuration used in `demo.py`.

PyTorch checkpoints use Python serialization. Only load checkpoint files obtained from trusted sources.

## Known Limitations of the Current Release

- The fixed map expected by the training scripts is not included and must be generated.
- Only the AD-SAC-UA pretrained checkpoint is included in `testmodels/`.
- Several experiment paths are configured directly inside scripts rather than through command-line arguments.
- The training scripts do not yet provide centralized random-seed control.
- PyVista visualization may require additional system-level display configuration on remote servers.

## Citation

## License

This project is licensed under the MIT License. For the full license terms, please refer to the LICENSE file in the project root directory.

## Acknowledgements

This implementation uses PyTorch, Gymnasium, NumPy, Matplotlib, TensorBoard, pandas, and PyVista.
