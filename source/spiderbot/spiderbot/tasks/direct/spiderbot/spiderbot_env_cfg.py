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

# ✅ Import your robot config
from .spiderbot_cfg import SPIDERBOT_CFG


@configclass
class EventCfg:
    """Random events during training."""
    
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.2),
            "dynamic_friction_range": (0.8, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "add",
        },
    )


@configclass
class SpiderbotEnvCfg(DirectRLEnvCfg):
    """Configuration for Spider Bot CPG-RL environment."""
    
    # Environment settings
    episode_length_s = 20.0
    decimation = 4
    
    # Action/observation spaces
    action_space = 17      # 1 freq + 12 amp + 4 phase
    observation_space = 53
    state_space = 0
    action_scale = 1.0

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=0,  # ✅ Enable collisions (not -1)
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,
            dynamic_friction=1.0,
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
        num_envs=200,
        env_spacing=2.0,
        replicate_physics=True
    )

    # Events
    events: EventCfg = EventCfg()

    # ✅ Robot - Clean and simple!
    robot = SPIDERBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # CPG parameters
    cpg_frequency_min = 0.0
    cpg_frequency_max = 1.0
    cpg_amplitude_min = 0.0
    cpg_amplitude_max = 0.3
    cpg_phase_min = -0.5
    cpg_phase_max = +0.5

    # Reward scales
    lin_vel_reward_scale = 5.0
    yaw_rate_reward_scale = 1.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -1e-5
    joint_accel_reward_scale = -5e-7
    action_rate_reward_scale = -0.01
    flat_orientation_reward_scale = -5.0


# spiderbot_env_cfg.py (add near bottom)

# @configclass
# class CurriculumCfg:
#     # thresholds are in [0,1] because we divide by each term's scale to recover the raw exp(.) avg
#     forward_success_threshold: float = 0.70
#     yaw_success_threshold: float = 0.65
#     lateral_success_threshold: float = 0.60
#     # min episodes before considering promotion
#     min_stage_episodes: int = 150
#     # rehearsal (mixing) probabilities per stage (index by stage, fallback to last)
#     mix_prev_probs: tuple[float, ...] = (0.0, 0.30, 0.30, 0.25)

#     # command ranges per stage
#     vx_ranges = ((0.15, 0.30), (0.10, 0.25), (0.08, 0.25), (-0.50, 0.50))
#     vy_ranges = ((0.00, 0.00), (0.00, 0.00), (-0.15, 0.15), (-0.20, 0.20))
#     yaw_ranges = ((0.00, 0.00), (-0.30, 0.30), (-0.25, 0.25), (-0.70, 0.70))

# @configclass
# class SpiderbotEnvCfg(DirectRLEnvCfg):
#     ...
#     curriculum: CurriculumCfg = CurriculumCfg()
