"""Configuration for Model-Based Spider Bot with IK + RL residuals."""

import math
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
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
            "mass_distribution_params": (-0.2, 0.2),
            "operation": "add",
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
            "distribution": "uniform",
        },
    )

@configclass
class SpiderbotEnvCfg(DirectRLEnvCfg):
    """Model-based configuration with fixed gait + RL residuals."""
    
    # Episode settings
    episode_length_s = 20.0
    decimation = 4  # 50Hz control (200Hz sim / 4)
    
    # Command history length
    obs_cmd_hist_len = 10
    
    # Action and observation spaces
    action_space = 12  # 12 joint angle residuals
    observation_space = 30  # 3 * obs_cmd_hist_len (command history - actor)
    state_space = 65  # Full state for critic: actor(30) + vel(3) + ang_vel(3) + gravity(3) + joint_pos(12) + joint_vel(12) + phase(2)
    
    # Gait parameters
    base_gait_frequency = 1.5  # Fixed trot frequency (Hz)
    max_residual_rad = 0.15  # Limit RL corrections to ±0.15 rad
    
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
        prim_path="/World/envs/env_.*/Robot/.*",
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
    
    # Reward weights (tuned for model-based approach)
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -1e-5
    action_rate_reward_scale = -0.01
    flat_orientation_reward_scale = -5.0
    residual_magnitude_scale = -0.05  # Encourage small corrections