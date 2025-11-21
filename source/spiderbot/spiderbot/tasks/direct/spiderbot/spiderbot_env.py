"""Model-Based Spider Bot environment with IK + RL residuals."""

import torch
import numpy as np
from typing import Dict
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_rotate_inverse

from .spider_ik import SpiderIK

class SpiderbotEnv(DirectRLEnv):
    """
    Model-based approach: IK generates nominal trajectories, RL learns residual corrections.
    
    Key features:
    - Fixed gait frequency (1.5 Hz trot)
    - IK handles all kinematics  
    - RL learns 12 small joint angle corrections
    - Actor observes only command history (sensor-less)
    - Critic observes full state for better value estimation
    """
    
    cfg: "SpiderbotEnvCfg"
    
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Initialize IK solver
        self.ik_solver = SpiderIK(device=self.device)
        
        # Gait state (managed by environment, not RL)
        self.gait_time = torch.zeros(self.num_envs, device=self.device)
        self.base_frequency = cfg.base_gait_frequency  # Fixed frequency (e.g., 1.5 Hz)
        
        # Command history buffer for actor observations
        self.cmd_history = torch.zeros(
            self.num_envs, cfg.obs_cmd_hist_len, 3, device=self.device
        )
        
        # Current commands
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.command_timer = torch.zeros(self.num_envs, device=self.device)
        
        # Store IK targets for observations
        self.ik_targets = torch.zeros(self.num_envs, 12, device=self.device)
        
        # Action smoothing
        self.previous_actions = torch.zeros(self.num_envs, cfg.action_space, device=self.device)
        
        # Tracking metrics
        self.episode_sums = {
            "forward_distance": torch.zeros(self.num_envs, device=self.device),
            "lateral_drift": torch.zeros(self.num_envs, device=self.device),
        }
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Process actions as residual corrections to IK-generated trajectories.
        
        Actions: (N, 12) - small joint angle corrections in radians
        """
        # Update gait time
        self.gait_time += self.physics_dt
        
        # Compute gait phases for trot gait
        # Diagonal pairs (FL-RR and FR-RL) move together
        base_phase = (2.0 * np.pi * self.base_frequency * self.gait_time) % (2.0 * np.pi)
        
        gait_phases = torch.stack([
            base_phase,  # FL: 0°
            (base_phase + np.pi) % (2.0 * np.pi),  # FR: 180° out of phase
            (base_phase + np.pi) % (2.0 * np.pi),  # RL: 180° (with FR)
            base_phase,  # RR: 0° (with FL)
        ], dim=1)  # (N, 4)
        
        # IK generates nominal joint targets based on commands and gait phase
        self.ik_targets = self.ik_solver.compute_all_legs_ik(
            gait_phases,
            self.commands[:, 0],  # vx
            self.commands[:, 1],  # vy
            self.commands[:, 2],  # vyaw
        )
        
        # RL learns small residual corrections to IK solution
        # Limit corrections to prevent breaking kinematics
        residuals = torch.tanh(actions) * self.cfg.max_residual_rad
        
        # Combine IK baseline with learned corrections
        joint_targets = self.ik_targets + residuals
        
        # Safety: clamp to joint limits
        joint_targets = torch.clamp(joint_targets, -1.5, 1.5)
        
        # Send targets to robot
        self.robot.set_joint_position_target(joint_targets, joint_ids=self._robot_joint_ids)
        
        # Store for action rate penalty
        self.previous_actions = actions.clone()
    
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """
        Asymmetric observations:
        - Actor: Command history only (for sensor-less deployment)
        - Critic: Full state for better value estimation
        """
        # Update command history (rolling buffer)
        self.cmd_history[:, 1:] = self.cmd_history[:, :-1].clone()
        self.cmd_history[:, 0] = self.commands
        
        # Actor observations: flattened command history
        actor_obs = self.cmd_history.view(self.num_envs, -1)
        
        # Critic observations: full state
        base_quat = self.robot.data.root_quat_w
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        
        # Gravity vector in body frame (orientation sensing)
        gravity_vec = quat_rotate_inverse(
            base_quat, 
            torch.tensor([0, 0, -1], device=self.device).repeat(self.num_envs, 1)
        )
        
        # Gait phase info (normalized to [0, 1])
        phase_normalized = (self.gait_time * self.base_frequency) % 1.0
        phase_info = torch.stack([
            torch.sin(2 * np.pi * phase_normalized),
            torch.cos(2 * np.pi * phase_normalized)
        ], dim=1)  # (N, 2)
        
        critic_obs = torch.cat([
            actor_obs,  # Command history
            base_lin_vel,  # (3)
            base_ang_vel,  # (3)
            gravity_vec,  # (3)
            joint_pos,  # (12)
            joint_vel,  # (12)
            phase_info,  # (2)
        ], dim=-1)
        
        return {"policy": actor_obs, "critic": critic_obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """
        Reward function optimized for straight-line tracking and stability.
        """
        # Get velocities
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        base_quat = self.robot.data.root_quat_w
        
        # === PRIMARY: Velocity Tracking ===
        vx_error = torch.abs(base_lin_vel[:, 0] - self.commands[:, 0])
        vy_error = torch.abs(base_lin_vel[:, 1] - self.commands[:, 1])
        vyaw_error = torch.abs(base_ang_vel[:, 2] - self.commands[:, 2])
        
        # Exponential rewards for smooth gradients
        velocity_reward = (
            torch.exp(-2.0 * vx_error) * 2.0 +  # Forward motion (highest weight)
            torch.exp(-3.0 * vy_error) * 1.0 +  # Lateral (penalize drift heavily)
            torch.exp(-2.0 * vyaw_error) * 1.0  # Yaw tracking
        )
        
        # === STABILITY PENALTIES ===
        
        # Vertical velocity (penalize hopping)
        z_vel_penalty = torch.square(base_lin_vel[:, 2]) * self.cfg.z_vel_reward_scale
        
        # Angular velocity about roll/pitch (penalize wobbling)
        ang_vel_penalty = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1) * self.cfg.ang_vel_reward_scale
        
        # Orientation (keep robot level)
        gravity_vec = quat_rotate_inverse(
            base_quat,
            torch.tensor([0, 0, -1], device=self.device).repeat(self.num_envs, 1)
        )
        orientation_penalty = (1.0 - gravity_vec[:, 2]) * self.cfg.flat_orientation_reward_scale
        
        # === EFFICIENCY ===
        
        # Energy: penalize large torques
        joint_torques = self.robot.data.applied_torque
        torque_penalty = torch.sum(torch.square(joint_torques), dim=1) * self.cfg.joint_torque_reward_scale
        
        # Action smoothness: penalize rapid changes
        action_rate = torch.sum(torch.square(self.actions - self.previous_actions), dim=1)
        action_penalty = action_rate * self.cfg.action_rate_reward_scale
        
        # === RESIDUAL PENALTY ===
        # Encourage RL to use small corrections (trust IK baseline)
        residual_magnitude = torch.sum(torch.square(self.actions), dim=1)
        residual_penalty = residual_magnitude * self.cfg.residual_magnitude_scale
        
        # === TOTAL REWARD ===
        total_reward = (
            velocity_reward +
            z_vel_penalty +
            ang_vel_penalty +
            orientation_penalty +
            torque_penalty +
            action_penalty +
            residual_penalty
        )
        
        # Logging
        self.extras["velocity_reward"] = velocity_reward.mean()
        self.extras["vx_error"] = vx_error.mean()
        self.extras["vy_error"] = vy_error.mean()
        self.extras["orientation_penalty"] = orientation_penalty.mean()
        self.extras["residual_magnitude"] = torch.sqrt(residual_magnitude).mean()
        
        return total_reward
    
    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """Reset environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        super()._reset_idx(env_ids)
        
        # Reset gait time
        self.gait_time[env_ids] = torch.rand(len(env_ids), device=self.device) * (2.0 * np.pi / self.base_frequency)
        
        # Sample new commands
        self._sample_commands(env_ids)
        
        # Reset histories
        self.cmd_history[env_ids] = 0
        self.ik_targets[env_ids] = 0
        self.previous_actions[env_ids] = 0
        
        # Reset tracking
        for key in self.episode_sums:
            self.episode_sums[key][env_ids] = 0
    
    def _sample_commands(self, env_ids: torch.Tensor) -> None:
        """
        Sample new velocity commands with emphasis on straight-line forward motion.
        """
        N = len(env_ids)
        
        # Command distribution: 70% straight forward, 20% turning, 10% complex
        motion_type = torch.rand(N, device=self.device)
        
        vx = torch.zeros(N, device=self.device)
        vy = torch.zeros(N, device=self.device)
        vyaw = torch.zeros(N, device=self.device)
        
        # Straight forward (70%)
        straight_mask = motion_type < 0.7
        vx[straight_mask] = torch.rand(straight_mask.sum(), device=self.device) * 0.25 + 0.1  # [0.1, 0.35] m/s
        
        # Forward with turning (20%)
        turn_mask = (motion_type >= 0.7) & (motion_type < 0.9)
        vx[turn_mask] = torch.rand(turn_mask.sum(), device=self.device) * 0.2 + 0.05  # [0.05, 0.25] m/s
        vyaw[turn_mask] = (torch.rand(turn_mask.sum(), device=self.device) - 0.5) * 0.8  # [-0.4, 0.4] rad/s
        
        # Complex motion (10%)
        complex_mask = motion_type >= 0.9
        vx[complex_mask] = torch.rand(complex_mask.sum(), device=self.device) * 0.2
        vy[complex_mask] = (torch.rand(complex_mask.sum(), device=self.device) - 0.5) * 0.2
        vyaw[complex_mask] = (torch.rand(complex_mask.sum(), device=self.device) - 0.5) * 0.4
        
        self.commands[env_ids, 0] = vx
        self.commands[env_ids, 1] = vy
        self.commands[env_ids, 2] = vyaw
        
        # Reset command timer
        self.command_timer[env_ids] = 0
    
    def _check_termination(self) -> torch.Tensor:
        """Check for early termination conditions."""
        # Robot fallen (base too low or flipped)
        base_pos = self.robot.data.root_pos_w
        base_quat = self.robot.data.root_quat_w
        
        gravity_vec = quat_rotate_inverse(
            base_quat,
            torch.tensor([0, 0, -1], device=self.device).repeat(self.num_envs, 1)
        )
        
        # Termination conditions
        fallen = (base_pos[:, 2] < 0.08) | (gravity_vec[:, 2] < 0.5)
        
        # Joint limits
        joint_pos = self.robot.data.joint_pos
        joint_limit_exceeded = (torch.abs(joint_pos) > 2.0).any(dim=1)
        
        return fallen | joint_limit_exceeded