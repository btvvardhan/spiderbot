"""Environment Configuration for VDP-CPG Spider Robot with IMU-Aware Actor

Key improvements over previous CPG-RL:
1. Van der Pol oscillators (better biological realism, duty cycle control)
2. Actor now includes IMU data (deployable with real IMU)
3. Richer CPG phase information (8D instead of 2D)
4. Multi-gait capability through learned phase relationships
5. Better reward structure for dynamic locomotion

Observation Design Philosophy:
- Actor sees ONLY what's available on real robot: commands, IMU, CPG phases
- No joint encoders needed (sensor-less joint control)
- Critic uses full privileged state for better training
"""

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

from .spiderbot_cfg import SPIDERBOT_CFG


@configclass
class EventCfg:
    """Domain randomization for robust training."""
    
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.3),
            "dynamic_friction_range": (0.6, 1.1),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.3, 0.5),
            "operation": "add",
        },
    )
    
    joint_stiffness = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.25),
            "damping_distribution_params": (0.75, 1.25),
            "operation": "scale",
        },
    )


@configclass
class SpiderbotEnvCfg(DirectRLEnvCfg):
    """VDP-CPG environment with IMU-aware actor."""
    
    # Episode settings
    episode_length_s = 20.0
    decimation = 4  # 50 Hz control
    
    # ========== ACTION SPACE ==========
    # Policy outputs CPG parameters:
    #   1: frequency (shared across legs)
    #   12: amplitudes (per joint)
    #   4: leg phase offsets (FL, FR, RL, RR)
    # Total: 17 dimensions
    action_space = 17
    action_scale = 1.0
    
    # ========== OBSERVATION SPACES ==========
    # Actor (deployable with real IMU):
    #   - commands: 3
    #   - command history: 3 * 5 = 15
    #   - IMU linear velocity: 3 (body frame)
    #   - IMU angular velocity: 3 (body frame)
    #   - projected gravity: 3 (body frame)
    #   - CPG phase features: 8 (sin/cos for 4 legs)
    #   - previous actions: 17
    # Total: 52 dimensions (IMU-aware!)
    cmd_history_len = 5
    observation_space = 52
    
    # Critic (privileged, training only):
    #   - actor obs: 52
    #   - joint positions (relative): 12
    #   - joint velocities: 12
    # Total: 76 dimensions
    state_space = 76

    # ========== SIMULATION ==========
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,  # 200 Hz physics
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
    )
    
    # ========== TERRAIN ==========
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=0,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # ========== SENSORS ==========
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=5,
        update_period=0.005,
        track_air_time=True,
    )

    # ========== SCENE ==========
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.0,
        replicate_physics=True
    )

    # ========== EVENTS ==========
    events: EventCfg = EventCfg()

    # ========== ROBOT ==========
    robot = SPIDERBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # ========== VDP-CPG PARAMETERS ==========
    # VDP oscillator parameters
    cpg_mu = 2.5  # Nonlinearity (2-3 good for locomotion)
    cpg_k_phase = 0.5  # Diagonal phase coupling (0-1)
    cpg_k_amp = 0.3    # Diagonal amplitude coupling (0-1)
    
    # CPG action ranges (policy outputs in [-1,1], scaled to these)
    cpg_frequency_min = 0.3
    cpg_frequency_max = 2.5
    cpg_amplitude_min = 0.0
    cpg_amplitude_max = 0.7
    cpg_phase_min = -1.0  # Phase offset range
    cpg_phase_max = +1.0
    
    # Action smoothing (low-pass filter)
    action_smoothing_beta = 0.15  # Slightly more smoothing than before
    
    # ========== COMMAND RANGES ==========
    # Multi-task training (sample all directions)
    command_ranges = {
        "lin_vel_x": (0.0, 0.5),    # m/s
        "lin_vel_y": (-0.3, 0.3),    # m/s
        "ang_vel_yaw": (-0.3, 0.3),  # rad/s
    }

    # ========== REWARD SCALES ==========
    # Primary tracking rewards
    lin_vel_reward_scale = 6.0       # Track linear velocity (increased)
    yaw_rate_reward_scale = 2.5      # Track yaw rate
    
    # Stability penalties
    z_vel_reward_scale = -2.0        # Minimize vertical motion
    ang_vel_reward_scale = -0.15     # Minimize roll/pitch rates
    flat_orientation_reward_scale = -4.0  # Keep body level
    
    # Efficiency penalties
    joint_torque_reward_scale = -3e-5    # Minimize effort
    joint_accel_reward_scale = -3e-7     # Smooth motion
    action_rate_reward_scale = -0.015    # Action smoothness
    
    # CPG-specific rewards
    cpg_phase_coherence_reward_scale = 0.5  # Reward coherent phase relationships
    
    # Joint limits
    joint_pos_limit_reward_scale = -15.0
    joint_pos_limit_margin = 0.1  # rad