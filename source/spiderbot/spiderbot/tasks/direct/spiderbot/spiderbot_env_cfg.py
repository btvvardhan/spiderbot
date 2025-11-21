"""Environment configuration for Spider Bot CPG-RL training (minimal, omni-directional)."""

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

# Robot config (standing-zero pose preserved)
from .spiderbot_cfg import SPIDERBOT_CFG  # keeps neutral stance at zero.  # noqa: E402


@configclass
class EventCfg:
    """Light randomization to improve robustness."""
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
    """Minimal CPG-RL env for omni-directional locomotion.

    Observations: stacked command history only (vx, vy, yaw).
    Actions: [1 freq, 12 amplitudes]; zero action => standing pose.
    """

    # Timing
    episode_length_s = 20.0
    decimation = 4

    # Action/observation spaces
    #   - 1 frequency + 12 joint amplitudes (no explicit per-leg phase action)
    action_space = 13

    #   - Only velocity commands and their history: 3 * H
    obs_cmd_hist_len = 10
    observation_space = 3 * obs_cmd_hist_len

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

    # Contact sensor (for safe terminations; not exposed to policy)
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

    # Domain randomization
    events: EventCfg = EventCfg()

    # Robot
    robot = SPIDERBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # CPG parameters and coupling (can be tuned or disabled)
    cpg_frequency_min = 0.0
    cpg_frequency_max = 3.0
    cpg_amplitude_min = 0.0
    cpg_amplitude_max = 0.6
    cpg_k_phase = 0.5     # 0 => no diagonal phase coupling; 1 => lock diagonals
    cpg_k_amp = 0.5       # 0 => no amplitude tie; 1 => lock diagonals

    # Command sampling (body-frame desired velocities)
    cmd_vx_min = -0.40; cmd_vx_max =  0.40
    cmd_vy_min = -0.40; cmd_vy_max =  0.40
    cmd_yaw_min = -0.60; cmd_yaw_max = 0.60
    command_change_interval_s = 2.0   # piecewise constant targets

    # Reward scales (unchanged core tracking, simple regularizers)
    lin_vel_reward_scale = 5.0
    yaw_rate_reward_scale = 1.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -1e-5
    joint_accel_reward_scale = -5e-7
    action_rate_reward_scale = -0.01
    flat_orientation_reward_scale = -5.0
