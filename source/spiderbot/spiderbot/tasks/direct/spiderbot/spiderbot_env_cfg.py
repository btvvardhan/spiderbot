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
from isaaclab.actuators import ImplicitActuatorCfg


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    
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
    episode_length_s = 20.0
    decimation = 4
    
    action_space = 17
    
    action_scale = 1.0
    
    observation_space = 53
    state_space = 0

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

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Spider/.*",
        history_length=5,
        update_period=0.005,
        track_air_time=True,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=200,
        env_spacing=2.0,
        replicate_physics=True
    )

    events: EventCfg = EventCfg()

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Spider",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/teja/spiderbot/assets/spiderbot/spiderbot.usd",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0
            ),
            activate_contact_sensors=True,
        ),

        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    r"Revolute_110", r"Revolute_111", r"Revolute_112",
                    r"Revolute_113", r"Revolute_114", r"Revolute_115",
                    r"Revolute_116", r"Revolute_117", r"Revolute_118",
                    r"Revolute_119", r"Revolute_120", r"Revolute_121",
                ],
                stiffness=150.0,
                damping=30.0,
                effort_limit_sim=60.0,
                velocity_limit_sim=8.0,
            )
        },

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.15),
            rot=(1.0, 0.0, 0.0, 0.0),

            joint_pos={
                "Revolute_110": 0.6,
                "Revolute_111": 0.5,
                "Revolute_112": -0.2,
                
                "Revolute_113": -0.6,
                "Revolute_114": 0.5,
                "Revolute_115": -0.2,
                
                "Revolute_116": 0.6,
                "Revolute_117": 0.5,
                "Revolute_118": -0.2,

                "Revolute_119": -0.6,
                "Revolute_120": 0.5,
                "Revolute_121": -0.2,
            },
            joint_vel={".*": 0.0},
        ),
    )
        
    # In spiderbot_env_cfg.py
    cpg_frequency_min = 0.0  # Was 3.0 - allow slower
    cpg_frequency_max = 3.0  # Was 4.0 - allow faster

    cpg_amplitude_min = 0.0  # Was 0.3 - allow smaller steps
    cpg_amplitude_max = 0.5  # Was 0.4 - allow bigger steps

    cpg_phase_min = -0.5  # Reduced from -3.14
    cpg_phase_max = 0.5   # Reduced from 3.14




    # reward scales
    lin_vel_reward_scale = 5.0  # Increased for more movement reward
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -1e-5  # Reduced punishment
    joint_accel_reward_scale = -1e-7  # Reduced punishment
    action_rate_reward_scale = -0.01
    flat_orientation_reward_scale = -5.0
    max_tilt_angle_deg = 45.0