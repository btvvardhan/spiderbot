"""Environment configuration for Spider Bot CPG-RL training (omni-directional, command-only obs)."""

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
    # Timing
    episode_length_s = 20.0
    decimation = 4

    # Spaces (command-only observation)
    action_space = 13                         # 1 frequency + 12 amplitudes
    obs_cmd_hist_len = 10
    observation_space = 3 * obs_cmd_hist_len  # (vx, vy, yaw) * H

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
        collision_group=0,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Contact sensor (for terminations only; not observed)
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
        replicate_physics=True,
    )

    # DR
    events: EventCfg = EventCfg()

    # Robot and CPG
    robot = SPIDERBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cpg_frequency_min = 0.0
    cpg_frequency_max = 3.0
    cpg_amplitude_min = 0.0
    cpg_amplitude_max = 0.8
    cpg_k_phase = 0.5
    cpg_k_amp = 0.5

    # Commands (ranges for sampler or your own remap)
    cmd_vx_min = -0.40; cmd_vx_max =  0.40
    cmd_vy_min = -0.40; cmd_vy_max =  0.40
    cmd_yaw_min = -0.60; cmd_yaw_max = 0.60
    command_change_interval_s = 2.0

    # >>> Frequency floor (prevents the CPG from stalling when |v*|>0)
    freq_floor_enable = True
    freq_floor_idle = 0.6   # Hz when command is tiny but non-zero
    freq_floor_run  = 1.8   # Hz at full command (|v*| == max)
    w_freq_bonus = 0.2      # small positive reward to use frequency when |v*|>0

    # Reward weights (projection tracker)
    w_align = 6.0
    w_speed_err = 4.0
    w_lat = 2.0
    w_yaw_err = 2.0

    # Stabilizers
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -1e-5
    joint_accel_reward_scale = -5e-7
    action_rate_reward_scale = -0.01
    flat_orientation_reward_scale = -5.0

    # Logging controls
    log_every_steps = 100
    log_to_extras = True
