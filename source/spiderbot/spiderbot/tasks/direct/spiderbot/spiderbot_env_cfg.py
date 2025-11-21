"""Optimized configuration for Spider Bot with IK and asymmetric actor-critic."""

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
    """Asymmetric actor-critic configuration."""
    
    # Episode settings
    episode_length_s = 20.0
    decimation = 4  # 50Hz control after decimation (200Hz sim / 4)
    
    # Command history length
    obs_cmd_hist_len = 10
    
    # Action and observation spaces
    action_space = 17  # 1 freq + 12 amps + 4 phase offsets
    observation_space = 30  # 3 * obs_cmd_hist_len (command history only - actor)
    state_space = 0  # Critic uses same observations as actor (set to 0 for symmetric)
    
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
    
    # CPG parameters (tuned for IK integration)
    cpg_k_phase = 0.7  # Stronger phase coupling for diagonal pairs
    cpg_k_amp = 0.5    # Moderate amplitude coupling
    
    # Phase action parameters
    phase_range_rad = math.pi * 0.5  # Limit phase adjustments
    phase_beta = 0.3  # Smoothing factor
    
    # Reward weights
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -1e-5
    action_rate_reward_scale = -0.01
    flat_orientation_reward_scale = -5.0