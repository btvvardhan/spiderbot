"""Optimized configuration for Spider Bot with IK and asymmetric actor-critic."""

import math
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
    """Domain randomization events."""
    
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.8, 1.2),  # Less variation for stability
            "operation": "scale",
        },
    )
    
    joint_stiffness = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

@configclass
class SpiderbotEnvCfg(DirectRLEnvCfg):
    """Asymmetric actor-critic configuration."""
    
    # Episode settings
    episode_length_s = 20.0
    decimation = 4  # 50Hz control after decimation (200Hz sim / 4)
    
    # === Observation/Action Spaces ===
    # Actor only sees command history
    obs_cmd_hist_len = 10
    observation_space = 3 * obs_cmd_hist_len  # 30 dims
    
    # Critic sees everything for better value estimation
    # cmd_history(30) + base_vel(6) + gravity(3) + joint_pos(12) + joint_vel(12) + phases(4) + ik_targets(12)
    state_space = 30 + 6 + 3 + 12 + 12 + 4 + 12  # 79 dims
    
    # Actions: freq(1) + amp_params(12) + phase_offsets(4)
    action_space = 17
    action_scale = 1.0
    
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 200.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
    )
    
    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    
    # Contact sensor
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*tibia.*",  # Only on feet
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )
    
    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
    )
    
    # Domain randomization
    events: EventCfg = EventCfg()
    
    # Robot configuration
    robot = SPIDERBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    # CPG parameters (tuned for IK integration)
    cpg_k_phase = 0.7  # Stronger phase coupling for diagonal pairs
    cpg_k_amp = 0.5    # Moderate amplitude coupling
    
    # Phase action parameters
    phase_range_rad = math.pi * 0.5  # Limit phase adjustments
    phase_beta = 0.3  # Smoothing factor
    
    # === Reward Weights (optimized for straight-line tracking) ===
    # Velocity tracking (main objectives)
    w_forward = 3.0      # Strong forward velocity tracking
    w_lateral = 1.0      # Lateral velocity tracking
    w_yaw = 1.5          # Yaw rate tracking
    
    # Stability penalties
    z_vel_reward_scale = -8.0        # Strong penalty for hopping
    ang_vel_reward_scale = -0.1      # Penalty for wobbling
    flat_orientation_reward_scale = -10.0  # Keep robot level
    
    # Efficiency
    joint_torque_reward_scale = -1e-5
    joint_accel_reward_scale = -1e-6
    action_rate_reward_scale = -0.02
    
    # Gait quality
    phase_regularity_scale = 0.5
    
    # Command sampling (biased toward forward motion)
    cmd_vx_range = [0.0, 0.4]    # Forward only
    cmd_vy_range = [-0.2, 0.2]   # Limited lateral
    cmd_vyaw_range = [-0.5, 0.5] # Moderate turning
    
    # How often to change commands
    command_change_interval_s = 3.0
    
    # Termination thresholds
    termination_height = 0.08
    termination_angle = 0.5  # cos(60 degrees)
    
    # Logging
    log_interval_steps = 100
    save_interval_steps = 1000