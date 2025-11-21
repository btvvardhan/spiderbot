# spiderbot_env_cfg.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
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
    """Configuration for randomization/materials."""

    # Default material on all bodies
    physics_material_all = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    # Make feet high-friction to discourage slipping and belly-sledding
    physics_material_feet = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["fl_tibia_link", "fr_tibia_link", "rl_tibia_link", "rr_tibia_link"],
            ),
            "static_friction_range": (1.6, 2.0),
            "dynamic_friction_range": (1.2, 1.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    # Make base low-friction so sliding on the base is unattractive
    physics_material_base = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "static_friction_range": (0.05, 0.10),
            "dynamic_friction_range": (0.03, 0.08),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    # Optional: add base mass variation
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (1.0, 3.0),
            "operation": "add",
        },
    )


@configclass
class SpiderbotEnvCfg(DirectRLEnvCfg):
    # ======== Env timing / spaces ========
    episode_length_s = 20.0
    decimation = 4
    action_scale = 1.0
    action_space = 12
    state_space = 0

    # ======== Commands-only observation ========
    cmd_history_len = 9
    use_phase = True
    use_commands_only_obs = True
    observation_space = 3 * (cmd_history_len + 1) + (2 if use_phase else 0)  # 32

    # Phase dynamics
    phase_base_hz = 1.5
    phase_k_v = 1.0

    # ======== Omni-direction commands ========
    cmd_vx_range = (-0.40, 0.60)   # m/s
    cmd_vy_range = (-0.40, 0.40)   # m/s
    cmd_yaw_range = (-1.00, 1.00)  # rad/s
    cmd_hold_time_s = 1.0          # resample every second

    # ======== Robust termination thresholds ========
    max_tilt_angle_deg = 45.0
    min_base_height = 0.12
    base_contact_force_thresh = 80.0     # N
    joint_pos_limit_rad = 3.14           # ~180 degrees, sanity check

    # ======== Reward shaping (anti-belly-sledding) ========
    base_contact_penalty_scale = -0.05   # multiplied by base contact force (N)
    base_height_target = 0.22            # m
    base_height_low_penalty_scale = -10.0
    foot_contact_force_thresh = 5.0      # N for "in contact"
    stance_contact_reward_scale = 1.0

    # ======== Simulation ========
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Contact sensor on the articulated robot under /World/envs/.../Robot
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=5,
        update_period=0.005,  # Matches sim dt = 1/200
        track_air_time=True,
    )

    # ======== Scene / events ========
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=200, env_spacing=2.0, replicate_physics=True)
    events: EventCfg = EventCfg()

    # ======== Robot ========
    robot: ArticulationCfg = SPIDERBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # ======== Reward scales (existing terms) ========
    lin_vel_reward_scale = 5.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -1e-5
    joint_accel_reward_scale = -1e-7
    action_rate_reward_scale = -0.01
    flat_orientation_reward_scale = -10.0  # stronger upright bias