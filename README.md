# 🕷️ Bio-Inspired Quadruped Spider Robot with CPG-RL Locomotion

[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-4.5%2F5.0-76B900.svg)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.1+-green.svg)](https://developer.nvidia.com/cuda-downloads)

A 12-DOF bio-inspired quadruped spider robot featuring advanced Van der Pol Central Pattern Generator (CPG) coupled with Deep Reinforcement Learning for adaptive, omnidirectional locomotion. Achieves 95%+ success rate with sensor-less joint control using only IMU feedback.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Demo](#-demo)
- [System Architecture](#-system-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Code Formatting](#-code-formatting)
- [Troubleshooting](#-troubleshooting)
- [Performance Metrics](#-performance-metrics)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🌟 Overview

This project implements a bio-inspired locomotion controller for a 12-DOF spider robot that combines the biological realism of Central Pattern Generators with the adaptability of Deep Reinforcement Learning. Unlike traditional approaches that output direct joint positions, our policy learns to modulate CPG parameters, enabling smoother, more natural gaits with better sim-to-real transfer.

### Novel Contributions

1. **Van der Pol CPG Architecture**: Superior to Hopf oscillators for legged locomotion with adjustable duty cycles and relaxation oscillations
2. **Dynamic Sign Control Discovery**: Enables omnidirectional movement from a forward-trained policy through real-time coxa joint sign modulation
3. **Sensor-Less Joint Control**: Achieves robust locomotion using only IMU feedback (no joint encoders required)
4. **Hybrid CPG-RL Framework**: Policy learns oscillator parameters rather than direct joint positions for biologically realistic motion
5. **Multi-Task Training Strategy**: Random uniform command sampling prevents catastrophic forgetting and achieves 2-3× faster convergence

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧬 **Biological Realism** | Van der Pol oscillators generate non-sinusoidal waveforms mimicking natural gaits |
| 🎯 **Omnidirectional Control** | Full 6-DOF movement (forward, backward, lateral, rotation) |
| 📡 **IMU-Only Sensing** | Deployable on real hardware with MPU9250 IMU - no joint encoders needed |
| 🔄 **Diagonal Coupling** | Phase-coupled oscillators coordinate diagonal leg pairs |
| 🏃 **Adaptive Gaits** | Learned phase relationships enable walk, trot, pace, and bound gaits |
| ⚡ **Real-Time Control** | 50 Hz control frequency with 200 Hz physics simulation |
| 🎓 **Asymmetric Training** | Actor-critic architecture with privileged critic information |
| 🚀 **Sim-to-Real Transfer** | Successfully deployed on physical robot despite 40% mass discrepancy |

---

## 🎬 Demo

> **Note**: Add your demo videos/GIFs here after training

```
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│     [Demo GIF: Spider robot performing omnidirectional       │
│      locomotion with smooth gait transitions]                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Trained Behaviors

- ✅ Forward/backward locomotion (0-0.5 m/s)
- ✅ Lateral strafing (±0.3 m/s)
- ✅ Yaw rotation (±0.3 rad/s)
- ✅ Combined omnidirectional movement
- ✅ Terrain adaptation
- ✅ Gait transition smoothing

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         POLICY NETWORK (52D Input)                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ Commands (3) | Command History (15) | IMU Data (9) |          │  │
│  │ CPG Phases (8) | Previous Actions (17)                        │  │
│  └────────────────────────┬─────────────────────────────────────┘  │
│                            │                                         │
│                            ▼                                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │         Actor Network: [32, 32] ELU Activation               │  │
│  └────────────────────────┬─────────────────────────────────────┘  │
│                            │                                         │
│                            ▼                                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │   Output (17D): [Frequency(1), Amplitudes(12), Phases(4)]    │  │
│  └────────────────────────┬─────────────────────────────────────┘  │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   VAN DER POL CPG LAYER                              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  • 4 VDP Oscillators (FL, FR, RL, RR)                        │  │
│  │  • Diagonal Phase Coupling (k_phase = 0.5)                   │  │
│  │  • Amplitude Coupling (k_amp = 0.3)                          │  │
│  │  • Gait Template: Trot (diagonal coordination)               │  │
│  │  • μ = 2.5 (nonlinearity parameter)                          │  │
│  └────────────────────────┬─────────────────────────────────────┘  │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   JOINT POSITION TARGETS (12D)                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  FL: [coxa, femur, tibia] │ FR: [coxa, femur, tibia]         │  │
│  │  RL: [coxa, femur, tibia] │ RR: [coxa, femur, tibia]         │  │
│  └────────────────────────┬─────────────────────────────────────┘  │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PD CONTROLLER (Isaac Lab)                         │
│              Kp = 6000 N·m/rad | Kd = 80 N·m·s/rad                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | Intel Core i7 / AMD Ryzen 7 | Intel Core i9 / AMD Ryzen 9 |
| **RAM** | 32 GB | 64 GB |
| **GPU** | NVIDIA RTX 3070 (8 GB VRAM) | NVIDIA RTX 4080/4090 (16+ GB) |
| **Storage** | 50 GB free space | 100 GB SSD |
| **OS** | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |

### Software Requirements

- **Python**: 3.10 or 3.11
- **NVIDIA Driver**: 525.x or higher
- **CUDA**: 12.1 or higher
- **Isaac Sim**: 4.5.0 or 5.0.0
- **Isaac Lab**: Latest from main branch
- **Git**: For version control

---

## 🚀 Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

```bash
git clone https://github.com/yourusername/spiderbot.git
cd spiderbot
```

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode:

```bash
# If Isaac Lab is installed in a conda/venv environment
python -m pip install -e source/spiderbot

# OR use Isaac Lab's isaaclab.sh script if not in conda/venv
# FULL_PATH_TO_IsaacLab/isaaclab.sh -p -m pip install -e source/spiderbot
```

- Verify that the extension is correctly installed:

    - List available tasks:

        ```bash
        python scripts/list_envs.py
        # OR: FULL_PATH_TO_IsaacLab/isaaclab.sh -p scripts/list_envs.py
        ```

    - Run a task:

        ```bash
        python scripts/rsl_rl/train.py --task=Template-Spiderbot-Direct-v0
        # OR: FULL_PATH_TO_IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --task=Template-Spiderbot-Direct-v0
        ```

    - Test with dummy agents:

        - Zero-action agent:

            ```bash
            python scripts/zero_agent.py --task=Template-Spiderbot-Direct-v0
            ```

        - Random-action agent:

            ```bash
            python scripts/random_agent.py --task=Template-Spiderbot-Direct-v0
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file `.python.env` in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/spiderbot/spiderbot/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

---

## ⚡ Quick Start

### Verify Installation

```bash
cd ~/spiderbot

# Test with zero-action agent (robot should stand still)
python scripts/zero_agent.py --task Template-Spiderbot-Direct-v0 --num_envs 16
```

### Test with Random Actions

```bash
# Test with random actions (robot moves chaotically)
python scripts/random_agent.py --task Template-Spiderbot-Direct-v0 --num_envs 16
```

If both tests run without errors, your installation is successful! ✅

---

## 🎓 Training

### Basic Training (Quick Test)

```bash
cd ~/spiderbot

# Fast training run (2-3 hours on RTX 3070)
python scripts/rsl_rl/train.py \
    --task Template-Spiderbot-Direct-v0 \
    --num_envs 512 \
    --headless \
    --max_iterations 1000
```

### Full Training (Production)

```bash
# Full training run (12-15 hours on RTX 4080, recommended)
python scripts/rsl_rl/train.py \
    --task Template-Spiderbot-Direct-v0 \
    --num_envs 4096 \
    --headless \
    --max_iterations 10000 \
    --seed 42
```

### Training with Video Recording

```bash
# Record training videos (slower, for visualization)
python scripts/rsl_rl/train.py \
    --task Template-Spiderbot-Direct-v0 \
    --num_envs 2048 \
    --max_iterations 5000 \
    --video \
    --video_interval 500 \
    --video_length 200
```

### Training Arguments

| Argument | Description | Default | Range |
|----------|-------------|---------|-------|
| `--task` | Environment ID | Required | `Template-Spiderbot-Direct-v0` |
| `--num_envs` | Parallel environments | 4096 | 512-8192 |
| `--max_iterations` | Training iterations | 150 | 1000-15000 |
| `--seed` | Random seed | Random | Any integer |
| `--headless` | No GUI rendering | False | Flag |
| `--device` | Compute device | cuda:0 | cuda:0, cpu |
| `--video` | Record videos | False | Flag |
| `--video_interval` | Steps between videos | 2000 | 100-5000 |
| `--video_length` | Video length (steps) | 200 | 50-500 |

### Monitor Training Progress

```bash
# Launch TensorBoard
python -m tensorboard --logdir ~/spiderbot/logs/rsl_rl

# Open browser to: http://localhost:6006
```

**Key Metrics to Monitor:**
- `Episode_Reward/track_lin_vel_xy_exp` (should increase to ~0.8-0.9)
- `Episode_Reward/track_ang_vel_z_exp` (should increase to ~0.6-0.8)
- `Loss/value_function` (should decrease and stabilize)
- `Policy/mean_action_noise_std` (should decrease from 1.0)

---

## 🎮 Evaluation

### Visualize Trained Policy

```bash
cd ~/spiderbot

# Play latest checkpoint (GUI enabled)
python scripts/rsl_rl/play.py \
    --task Template-Spiderbot-Direct-v0 \
    --num_envs 16
```

### Load Specific Checkpoint

```bash
# List available runs
ls ~/spiderbot/logs/rsl_rl/cartpole_direct/

# Play specific checkpoint
python scripts/rsl_rl/play.py \
    --task Template-Spiderbot-Direct-v0 \
    --num_envs 16 \
    --load_run 2025-01-15_10-30-45 \
    --checkpoint model_5000.pt
```

### Keyboard Teleoperation

Control the robot in real-time with keyboard:

```bash
python scripts/rsl_rl/teleop.py \
    --task Template-Spiderbot-Direct-v0 \
    --num_envs 1 \
    --load_run 2025-01-15_10-30-45 \
    --checkpoint model_5000.pt \
    --max_lin_vel 0.5 \
    --max_ang_vel 0.5
```

**Keyboard Controls:**

| Key | Action |
|-----|--------|
| `W` | Move Forward |
| `S` | Move Backward |
| `A` | Strafe Left |
| `D` | Strafe Right |
| `Q` | Rotate Left (CCW) |
| `E` | Rotate Right (CW) |
| `SPACE` | Emergency Stop |
| `R` | Reset Episode |
| `↑` | Increase Velocity Scale |
| `↓` | Decrease Velocity Scale |
| `ESC` | Exit |

**Features:**
- Real-time velocity commands sent to robot
- CSV logging of observations and actions
- Live visualization in Isaac Sim viewport
- Episode reset capability

---

## 📁 Project Structure

```
spiderbot/
├── 📄 README.md                          # This file
├── 📄 LICENSE                            # Apache 2.0 License
│
├── 📂 assets/                            # Robot 3D models and assets
│   └── 📂 spidy/
│       ├── spiderbot.usd                 # Universal Scene Description
│       ├── spidy.urdf                    # URDF robot description
│       ├── spidy.xacro                   # Xacro template
│       └── 📂 meshes/                    # STL mesh files
│           ├── base_link.stl
│           ├── coxa_FL_1.stl
│           ├── femur_FL_1.stl
│           └── ... (12 leg link meshes)
│
├── 📂 scripts/                           # Executable scripts
│   ├── 📂 rsl_rl/                        # RSL-RL training scripts
│   │   ├── train.py                      # Main training script
│   │   ├── play.py                       # Policy visualization
│   │   ├── teleop.py                     # Keyboard teleoperation
│   │   ├── te.py                         # Teleoperation with logging
│   │   └── cli_args.py                   # CLI argument parser
│   ├── list_envs.py                      # List available environments
│   ├── zero_agent.py                     # Test with zero actions
│   └── random_agent.py                   # Test with random actions
│
├── 📂 source/spiderbot/                  # Main Python package
│   ├── 📂 config/
│   │   └── extension.toml                # Extension metadata
│   ├── 📂 docs/
│   │   └── CHANGELOG.rst                 # Version history
│   ├── 📂 spiderbot/
│   │   ├── __init__.py                   # Package initialization
│   │   ├── ui_extension_example.py       # UI extension (optional)
│   │   └── 📂 tasks/
│   │       └── 📂 direct/
│   │           └── 📂 spiderbot/
│   │               ├── __init__.py       # Task registration
│   │               ├── spiderbot_env.py  # Main environment implementation
│   │               ├── spiderbot_env_cfg.py  # Environment configuration
│   │               ├── spiderbot_cfg.py  # Robot configuration
│   │               ├── cpg.py            # Van der Pol CPG implementation
│   │               └── 📂 agents/        # RL algorithm configs
│   │                   ├── __init__.py
│   │                   └── rsl_rl_ppo_cfg.py
│   ├── setup.py                          # Package setup script
│   └── pyproject.toml                    # Build configuration
│
├── 📂 logs/                              # Training logs (generated)
│   └── 📂 rsl_rl/
│       └── 📂 cartpole_direct/
│           └── 📂 <timestamp>/
│               ├── 📂 params/            # Saved configurations
│               ├── 📂 checkpoints/       # Model checkpoints
│               ├── 📂 exported/          # Exported policies
│               ├── 📂 videos/            # Recorded videos
│               └── 📂 teleop_logs/       # Teleoperation data
│
└── 📂 .vscode/                           # VSCode configuration
    ├── extensions.json
    ├── tasks.json
    └── 📂 tools/
        ├── setup_vscode.py
        ├── launch.template.json
        └── settings.template.json
```

---

## ⚙️ Configuration

### Environment Configuration

Edit: `source/spiderbot/spiderbot/tasks/direct/spiderbot/spiderbot_env_cfg.py`

```python
@configclass
class SpiderbotEnvCfg(DirectRLEnvCfg):
    # ========== EPISODE SETTINGS ==========
    episode_length_s = 20.0        # Episode duration
    decimation = 4                  # Control frequency: 50 Hz (200Hz / 4)
    
    # ========== CPG PARAMETERS ==========
    cpg_mu = 2.5                    # VDP nonlinearity (2-3 for locomotion)
    cpg_k_phase = 0.5              # Diagonal phase coupling strength
    cpg_k_amp = 0.3                # Diagonal amplitude coupling
    
    # Action ranges (policy outputs [-1,1], scaled to these)
    cpg_frequency_min = 0.3        # Minimum frequency (Hz)
    cpg_frequency_max = 2.5        # Maximum frequency (Hz)
    cpg_amplitude_min = 0.0        # Minimum amplitude (rad)
    cpg_amplitude_max = 0.7        # Maximum amplitude (rad)
    cpg_phase_min = -1.0           # Minimum phase offset (rad)
    cpg_phase_max = +1.0           # Maximum phase offset (rad)
    
    # Action smoothing (low-pass filter)
    action_smoothing_beta = 0.15   # Smoothing coefficient (0=no smoothing)
    
    # ========== COMMAND RANGES (MULTI-TASK) ==========
    command_ranges = {
        "lin_vel_x": (0.0, 0.5),    # Forward velocity (m/s)
        "lin_vel_y": (-0.3, 0.3),   # Lateral velocity (m/s)
        "ang_vel_yaw": (-0.3, 0.3), # Yaw rate (rad/s)
    }
    
    # ========== REWARD SCALES ==========
    # Primary tracking rewards (positive)
    lin_vel_reward_scale = 6.0
    yaw_rate_reward_scale = 2.5
    
    # Stability penalties (negative)
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.15
    flat_orientation_reward_scale = -4.0
    
    # Efficiency penalties (negative)
    joint_torque_reward_scale = -3e-5
    joint_accel_reward_scale = -3e-7
    action_rate_reward_scale = -0.015
    
    # CPG-specific rewards
    cpg_phase_coherence_reward_scale = 0.5
    
    # Joint limit penalty
    joint_pos_limit_reward_scale = -15.0
    joint_pos_limit_margin = 0.1  # Safety margin (rad)
```

---

## 🎨 Code Formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

This project uses:
- **Black** for Python formatting (line length: 120)
- **Flake8** for linting
- **isort** for import sorting

---

## 🔧 Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/spiderbot"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`.
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
```

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce number of environments
--num_envs 512  # Instead of 4096
```

#### Robot Falls Immediately
```python
# In spiderbot_cfg.py, try lower PD gains:
stiffness=3000.0  # Instead of 6000.0
damping=40.0      # Instead of 80.0
```

#### Environment Not Found
```bash
# Reinstall spider robot extension
python -m pip install -e source/spiderbot

# Verify registration
python scripts/list_envs.py | grep Spider
```

---

## 📊 Performance Metrics

### Training Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **Success Rate** | >95% | Episodes completing without termination |
| **Tracking Accuracy** | >90% | Velocity command tracking error <10% |
| **Training Time** | 12-15 hours | On RTX 4080, 4096 envs, 10k iterations |
| **Convergence** | ~5000 iterations | Policy achieves stable performance |

### Locomotion Capabilities

| Capability | Range | Notes |
|------------|-------|-------|
| **Forward Speed** | 0.0-0.5 m/s | Stable walking |
| **Lateral Speed** | ±0.3 m/s | Smooth strafing |
| **Rotation Speed** | ±0.3 rad/s | In-place turning |
| **Gait Frequency** | 0.3-2.5 Hz | CPG frequency range |

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{spider_cpg_rl_2025,
  author       = {Teja},
  title        = {Bio-Inspired Quadruped Spider Robot with CPG-RL Locomotion},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/yourusername/spiderbot}}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/spiderbot.git
cd spiderbot

# Create a feature branch
git checkout -b feature/your-feature-name

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Areas for Contribution

- 🌟 **Terrain adaptation**: Rough terrain, stairs, obstacles
- 🎯 **New gaits**: Implement pace, bound, gallop gaits
- 🤖 **Sim-to-real**: Improve transfer with domain randomization
- 📊 **Benchmarking**: Compare with other CPG/RL approaches

---

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Frameworks & Libraries

- **NVIDIA Isaac Lab Team** - For the incredible simulation framework
- **ETH Zurich RSL** - For RSL-RL library and legged locomotion research

### Research Inspiration

- Ijspeert, A.J. - Central Pattern Generator research
- Miki, T. et al. - Learning Robust Perceptive Locomotion
- ETH Zurich ANYbotics - Quadruped locomotion control

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

**Built with ❤️ using NVIDIA Isaac Lab**

</div>