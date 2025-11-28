"""Environment configuration for Spider Bot CPG-RL training."""

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
    """Random events during training."""
    
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    
    # Asymmetric mass randomization (payloads typically added, not removed)
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.3, 0.5),  # -0.3kg to +0.5kg
            "operation": "add",
        },
    )


@configclass
class SpiderbotEnvCfg(DirectRLEnvCfg):
    """Configuration for 3kg Spider Bot CPG-RL environment."""
    
    # Environment settings
    episode_length_s = 20.0
    decimation = 4
    action_scale = 1.0
    action_space = 17
    
    # Asymmetric actor-critic observations
    cmd_history_len = 5
    observation_space = 37  # Actor (sensor-less, deployable)
    state_space = 70        # Critic (privileged, training only)

    # Simulation
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
    
    # Terrain
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

    # Contact sensors
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=5,
        update_period=0.005,
        track_air_time=True,
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.5,  # Slightly more space for 3kg robot
        replicate_physics=True
    )

    # Events
    events: EventCfg = EventCfg()

    # Robot
    robot = SPIDERBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # CPG parameter ranges - CONSERVATIVE for initial training
    # These will produce slow, stable gaits suitable for 3kg robot
    cpg_frequency_min = 0.8    # ~0.8 Hz = slow walk
    cpg_frequency_max = 1.8    # ~1.8 Hz = moderate trot
    cpg_amplitude_min = 0.05   # Small movements
    cpg_amplitude_max = 0.25   # Conservative max (your legs are long!)
    cpg_phase_min = -0.2       # Tight phase coupling
    cpg_phase_max = +0.2

    # Reward scales - PHASE 1 (walking)
    lin_vel_reward_scale = 15.0
    lin_vel_exp_scale = 0.25
    yaw_rate_reward_scale = 0.5
    yaw_rate_exp_scale = 0.25
    z_vel_reward_scale = -5.0
    ang_vel_reward_scale = -0.2
    joint_torque_reward_scale = -3e-5
    joint_accel_reward_scale = -2e-6
    action_rate_reward_scale = -0.003
    flat_orientation_reward_scale = -12.0
    alive_reward_scale = 1.5