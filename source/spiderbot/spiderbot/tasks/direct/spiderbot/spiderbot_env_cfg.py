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
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (0.8, 0.8),
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
        prim_path="/World/envs/env_.*/Spider/spiderbot/.*",
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
        prim_path="/World/envs/env_.*/Spider",      # container prim name in each env
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/teja/spiderbot/assets/spidy/spiderbot.usd",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8
            ,
            #activate_contact_sensors=True
        ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0
            ),
            activate_contact_sensors=True,
        ),

        # ✅ Use your 12 actual joint names (3 × 4 legs)
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    "fl_coxa_joint",  "fl_femur_joint",  "fl_tibia_joint",
                    "fr_coxa_joint",  "fr_femur_joint",  "fr_tibia_joint",
                    "rl_coxa_joint",  "rl_femur_joint",  "rl_tibia_joint",
                    "rr_coxa_joint",  "rr_femur_joint",  "rr_tibia_joint",
                ],
                # sane PD for sim (you can tune later)
                stiffness=100.0,
                damping=4.0,
                effort_limit_sim=40.0,     # raise to 60 if it sags; lower if too “strong”
                velocity_limit_sim=8.0,
            )
        },

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.15),
            rot=(1.0, 0.0, 0.0, 0.0),

            # ✅ Same angles you had before, mapped to your joint names
            joint_pos={
                "fl_coxa_joint":  +0.0,   "fl_femur_joint": 0.0, "fl_tibia_joint": -0.0,
                "fr_coxa_joint":  -0.0,   "fr_femur_joint": 0.0, "fr_tibia_joint": -0.0,
                "rl_coxa_joint":  +0.0,   "rl_femur_joint": 0.0, "rl_tibia_joint": -0.0,
                "rr_coxa_joint":  -0.0,   "rr_femur_joint": 0.0, "rr_tibia_joint": -0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )        






        
        # Allow stillness (no forced motion at zero command)
    cpg_frequency_min = 0.0     # was 1.0
    cpg_frequency_max = 2.0

    cpg_amplitude_min = 0.0     # was 0.15
    cpg_amplitude_max = 0.3

    # Allow exploring different gait families (trot/pace/turn)
    cpg_phase_min = -0.5     # was -0.5
    cpg_phase_max = +0.5     # was +0.5

    # Reward scales
    lin_vel_reward_scale = 5.0
    yaw_rate_reward_scale = 1.5   # was 0.5  (care more about yaw control)
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -1e-5
    joint_accel_reward_scale = -5e-7
    action_rate_reward_scale = -0.01
    flat_orientation_reward_scale = -5.0