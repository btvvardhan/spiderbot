"""Configuration for the Spider Bot robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass


SPIDERBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/teja/spiderbot/assets/spidy/spiderbot.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "fl_coxa_joint":  0.0, "fl_femur_joint":  -0.6, "fl_tibia_joint":  0.9,
            "fr_coxa_joint":  0.0, "fr_femur_joint":  -0.6, "fr_tibia_joint":  0.9,
            "rl_coxa_joint":  0.0, "rl_femur_joint":  -0.6, "rl_tibia_joint":  0.9,
            "rr_coxa_joint":  0.0, "rr_femur_joint":  -0.6, "rr_tibia_joint":  0.9,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "fl_coxa_joint", "fl_femur_joint", "fl_tibia_joint",
                "fr_coxa_joint", "fr_femur_joint", "fr_tibia_joint",
                "rl_coxa_joint", "rl_femur_joint", "rl_tibia_joint",
                "rr_coxa_joint", "rr_femur_joint", "rr_tibia_joint",
            ],
            stiffness=6000.0,   # ↓ was 10000.0
            damping=80.0,       # ↓ was 100.0
            effort_limit_sim=100.0,
            velocity_limit_sim=5.0,
        ),
    },
)
