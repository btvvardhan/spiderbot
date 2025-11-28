"""Environment configuration for 3kg Spider Bot WALK gait training."""

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
            "static_friction_range": (0.6, 1.4),  # Wider for walk stability
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.4, 0.6),  # -400g to +600g
            "operation": "add",
        },
    )


@configclass
class SpiderbotEnvCfg(DirectRLEnvCfg):
    """Configuration for 3kg Spider Bot WALK gait training."""
    
    # Environment settings
    episode_length_s = 25.0  # Longer episodes for walk
    decimation = 4
    action_scale = 1.0
    action_space = 17
    
    # Asymmetric actor-critic
    cmd_history_len = 5
    observation_space = 37  # Actor
    state_space = 70        # Critic

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
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
        history_length=6,  # Longer for walk cycle detection
        update_period=0.005,
        track_air_time=True,
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True
    )

    # Events
    events: EventCfg = EventCfg()

    # Robot
    robot = SPIDERBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # CPG parameter ranges - WALK GAIT (slower, more controlled)
    cpg_frequency_min = 0.4    # ~0.4 Hz = very slow walk
    cpg_frequency_max = 1.2    # ~1.2 Hz = moderate walk
    cpg_amplitude_min = 0.03   # Small movements
    cpg_amplitude_max = 0.18   # Conservative max for walk
    cpg_phase_min = -0.15      # Tighter coupling for coordination
    cpg_phase_max = +0.15

    # Reward scales (not directly used, kept for reference)
    lin_vel_reward_scale = 12.0
    lin_vel_exp_scale = 0.20
    yaw_rate_reward_scale = 0.8
    yaw_rate_exp_scale = 0.25
    z_vel_reward_scale = -6.0
    ang_vel_reward_scale = -0.3
    joint_torque_reward_scale = -4e-5
    joint_accel_reward_scale = -2e-6
    action_rate_reward_scale = -0.003
    flat_orientation_reward_scale = -12.0
    alive_reward_scale = 2.0