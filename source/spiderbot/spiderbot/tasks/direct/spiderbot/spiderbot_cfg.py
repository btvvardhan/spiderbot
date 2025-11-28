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
    
    # Initial state - OPTIMIZED FOR YOUR DIMENSIONS
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.25),  # Spawn at 25cm, will settle to ~13cm
        rot=(1.0, 0.0, 0.0, 0.0),
        
        # STABLE CROUCHED STANCE
        # Calculated for: coxa=55mm, femur=80mm, tibia=150mm
        # Target standing height: ~130mm
        joint_pos={
            # Front Left
            "fl_coxa_joint":   0.0,    # Neutral (forward-back alignment)
            "fl_femur_joint":  -0.5,    # Lift femur ~34°
            "fl_tibia_joint": -0.5,    # Bend tibia ~57° downward
            
            # Front Right
            "fr_coxa_joint":   0.0,
            "fr_femur_joint":  -0.5,
            "fr_tibia_joint": 0.5,
            
            # Rear Left
            "rl_coxa_joint":   0.0,
            "rl_femur_joint":  -0.5,
            "rl_tibia_joint": 0.5,
            
            # Rear Right
            "rr_coxa_joint":   0.0,
            "rr_femur_joint":  -0.5,
            "rr_tibia_joint": 0.5,
        },
        joint_vel={".*": 0.0},
    ),
    
    # Actuators - Tuned PD gains for 3kg robot
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "fl_coxa_joint",  "fl_femur_joint",  "fl_tibia_joint",
                "fr_coxa_joint",  "fr_femur_joint",  "fr_tibia_joint",
                "rl_coxa_joint",  "rl_femur_joint",  "rl_tibia_joint",
                "rr_coxa_joint",  "rr_femur_joint",  "rr_tibia_joint",
            ],
            
            # PD gains optimized for DS3225 servos + 3kg load
            stiffness=2500.0,   # Moderate stiffness
            damping=40.0,       # Light damping for responsiveness
            
            effort_limit_sim=100.0,   # DS3225 spec: ~20kg.cm ≈ 2Nm per joint
            velocity_limit_sim=10.0,  # ~5 rad/s is realistic for servos
        ),
    },
)