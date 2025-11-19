# spiderbot_env_cfg_.py
""" environment configuration with bias mitigation and better curriculum."""

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
    """Configuration for randomization."""

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
    
    # ✅ NEW: Add joint friction randomization
    joint_friction = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),  # Use this instead of stiffness_range
            "damping_distribution_params": (0.75, 1.5),    # Use this instead of damping_range
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )


@configclass
class SpiderbotEnvCfg(DirectRLEnvCfg):
    """ configuration for Spider Bot with bias mitigation."""
    
    # Environment settings
    episode_length_s = 20.0
    decimation = 4
    
    # Action/observation spaces
    action_space = 17       # 1 freq + 12 amp + 4 phase
    observation_space = 54  # +1 for curriculum level
    state_space = 0
    action_scale = 1.0
    
    # ✅ NEW: Training improvements
    add_action_noise = True  # Add small noise to break symmetry
    use_adaptive_smoothing = True  # Curriculum-based smoothing

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
        # ✅ : Better physics simulation
        physx=sim_utils.PhysxCfg(
            solver_type=1,  # TGS solver for better stability
            enable_stabilization=True,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.04,
            friction_correlation_distance=0.025,
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

    # Robot
    robot = SPIDERBOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # ✅ : CPG parameters with better ranges
    cpg_frequency_min = 0.5   # Higher minimum for stability
    cpg_frequency_max = 2.5   # Lower maximum to prevent instability
    cpg_amplitude_min = 0.05  # Small minimum to maintain motion
    cpg_amplitude_max = 0.5   # Reduced max for control
    cpg_phase_min = -2.0      # Reduced range for easier learning
    cpg_phase_max = +2.0

    # ✅ : Reward scales with better balance
    # Primary tracking rewards (increased importance)
    lin_vel_x_reward_scale = 8.0        # Increased for stronger forward motion incentive
    lin_vel_y_reward_scale = 6.0        # Slightly less than forward
    yaw_rate_reward_scale = 3.0         # Doubled for better yaw tracking
    
    # Drift penalties (stronger to prevent unwanted motion)
    lateral_drift_penalty_scale = -10.0  # Stronger lateral drift penalty
    yaw_drift_penalty_scale = -5.0       # Stronger yaw drift penalty
    
    # Stability rewards
    z_vel_reward_scale = -3.0           # Increased for better vertical stability
    ang_vel_reward_scale = -0.1         # Stronger roll/pitch penalty
    flat_orientation_reward_scale = -10.0  # Strong upright incentive
    
    # Efficiency penalties (reduced to allow more exploration)
    joint_torque_reward_scale = -5e-6    # Reduced by half
    joint_accel_reward_scale = -2.5e-7   # Reduced by half
    action_rate_reward_scale = -0.005    # Reduced for more dynamic motion
    
    # ✅ NEW: Symmetry rewards
    yaw_symmetry_reward_scale = 2.0      # Reward balanced yaw behavior
    leg_symmetry_reward_scale = -0.05    # Penalize asymmetric gaits

    # ✅ NEW: Curriculum parameters
    curriculum_success_threshold = 0.7   # Success rate to advance level
    curriculum_window_size = 100         # Episodes to average over
    curriculum_max_level = 3             # Maximum difficulty level


@configclass
class TrainingConfig:
    """Training hyperparameters for  convergence."""
    
    # PPO parameters
    learning_rate = 3e-4
    lr_schedule = "adaptive"  # Reduce LR when plateauing
    
    # Batch sizes
    num_envs = 2048          # More envs for better statistics
    minibatch_size = 64
    
    # Training length
    max_iterations = 5000
    
    # Entropy regularization (important for exploration)
    entropy_coef = 0.01      # Increased for more exploration
    
    # Gradient clipping
    max_grad_norm = 1.0
    
    # Value function
    vf_coef = 0.5
    vf_clip_param = 10.0
    
    # PPO clip
    clip_param = 0.2
    
    # GAE
    gamma = 0.99
    gae_lambda = 0.95
    
    # Network architecture
    hidden_dims = [256, 256, 128]  # Deeper network
    activation = "elu"              # ELU often works better than ReLU
    
    # Normalization
    normalize_obs = True
    normalize_rewards = True
    
    # ✅ NEW: Symmetry-specific training
    use_symmetric_sampling = True   # Sample equal left/right turns
    augment_with_mirror = True      # Add mirrored experiences