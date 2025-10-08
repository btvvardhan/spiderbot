# spiderbot_env_cfg.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for Spider Robot with CPG-RL
===========================================

KEY CHANGES FROM PURE RL:
--------------------------
1. action_space: 12 → 17
   - Was: 12 direct joint position offsets
   - Now: 1 frequency + 12 amplitudes + 4 leg phase offsets
   
2. action_scale: Changed to 1.0
   - With CPGs, actions are parameters (not positions), so we don't need aggressive scaling
   
3. New reward terms for CPG
   - Gait symmetry
   - Phase coordination
   - Smooth frequency changes

4. Lower stiffness: 1500 → 100
   - Bio-inspired: compliant legs like real spiders
   - High stiffness fights natural dynamics
"""

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
    """Configuration for domain randomization."""
    
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
    """Configuration for Spider CPG-RL environment."""
    
    # ============================================
    # ENVIRONMENT SETTINGS
    # ============================================
    episode_length_s = 20.0
    decimation = 4  # Run 4 physics steps per policy step
    
    # ============================================
    # CPG ACTION SPACE (CHANGED FROM 12 → 17)
    # ============================================
    # Action breakdown:
    # [0]: frequency (rad/s) - shared by all oscillators
    # [1-12]: amplitudes (rad) - one per joint
    # [13-16]: leg phase offsets (rad) - one per leg
    action_space = 17  # ← CHANGED: was 12 (direct joint control)
    
    # Action scaling: converts NN output [-1, 1] to meaningful ranges
    # Frequency: map [-1, 1] → [0.5, 3.0] rad/s (roughly 0.08-0.5 Hz stepping)
    # Amplitude: map [-1, 1] → [-0.5, 0.5] rad (±28 degrees)
    # Phase: map [-1, 1] → [-π, π] rad
    # We'll handle this scaling in the environment
    action_scale = 1.0  # Keep at 1.0 since we do custom scaling per parameter type
    
    # Observation space (unchanged)
    observation_space = 53
    state_space = 0

    # ============================================
    # SIMULATION
    # ============================================
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,  # 200 Hz physics (5ms per step)
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
        update_period=0.005,  # 5ms (matches sim dt)
        track_air_time=True,
    )

    # ============================================
    # SCENE
    # ============================================
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=200, 
        env_spacing=2.0, 
        replicate_physics=True
    )

    # ============================================
    # EVENTS (Domain Randomization)
    # ============================================
    events: EventCfg = EventCfg()

    # ============================================
    # ROBOT CONFIGURATION
    # ============================================
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

        # ============================================
        # ACTUATORS (BIO-INSPIRED: COMPLIANT!)
        # ============================================
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[
                    # Leg 1 (Front-Left): Hip, Knee, Ankle
                    r"Revolute_110", r"Revolute_111", r"Revolute_112",
                    # Leg 2 (Front-Right): Hip, Knee, Ankle
                    r"Revolute_113", r"Revolute_114", r"Revolute_115",
                    # Leg 3 (Back-Left): Hip, Knee, Ankle
                    r"Revolute_116", r"Revolute_117", r"Revolute_118",
                    # Leg 4 (Back-Right): Hip, Knee, Ankle
                    r"Revolute_119", r"Revolute_120", r"Revolute_121",
                ],
                # ✅ BIO-INSPIRED: Low stiffness for compliance
                # Real spider legs are flexible, not rigid
                # This allows natural dynamics and energy storage
                stiffness=100.0,  # ← CHANGED from 1500! Much more compliant
                damping=15.0,     # ← CHANGED from 30! Allows oscillations
                effort_limit_sim=40.0,
                velocity_limit_sim=8.0,
            )
        },

        # Initial state
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.22),   # Spawn slightly above ground
            rot=(1.0, 0.0, 0.0, 0.0),  # Quaternion (w, x, y, z)
            joint_pos={".*": 0.0},  # All joints at default position
            joint_vel={".*": 0.0},
        ),
    )

    # ============================================
    # REWARD SCALES
    # ============================================
    # Existing rewards (for velocity tracking and stability)
    lin_vel_reward_scale = 5.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    flat_orientation_reward_scale = -5.0
    
    # ============================================
    # NEW: CPG-SPECIFIC REWARDS
    # ============================================
    # Reward smooth frequency changes (avoid jerky gait transitions)
    frequency_change_reward_scale = -0.1
    
    # Reward gait symmetry (left-right balance)
    gait_symmetry_reward_scale = 0.5
    
    # Reward proper contact timing (feet should touch ground at right phase)
    contact_timing_reward_scale = 1.0
    
    # Penalize foot slipping (feet should be stationary during stance)
    foot_slip_reward_scale = -0.5
    
    # ============================================
    # CPG PARAMETER RANGES (for action scaling)
    # ============================================
    # These define the valid ranges for CPG parameters
    # RL output [-1, 1] will be mapped to these ranges
    
    # Frequency range (rad/s)
    # 1.0 rad/s ≈ 0.16 Hz ≈ very slow walk
    # 6.0 rad/s ≈ 1.0 Hz ≈ fast trot
    cpg_frequency_min = 1.0
    cpg_frequency_max = 6.0
    
    # Amplitude range (radians)
    # ±0.5 rad ≈ ±28 degrees (reasonable joint movement)
    cpg_amplitude_min = 0.0
    cpg_amplitude_max = 0.5
    
    # Phase offset range (radians)
    # ±π allows full phase shift range
    cpg_phase_min = -3.14159
    cpg_phase_max = 3.14159