"""Optimized Spider Bot environment with IK and  actor-critic for sim-to-real."""

import torch
import numpy as np
from typing import Dict
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import (
    quat_rotate_inverse,
    quat_from_euler_xyz,
    euler_xyz_from_quat,
)
from .spider_ik import SpiderIK
from .cpg import SpiderCPG_Advanced

class SpiderbotEnv(DirectRLEnv):
    """
     actor-critic environment for spider robot.
    - Actor: Only observes command history (sensor-less for real robot)
    - Critic: Observes full state for better value estimation
    - Uses IK for stable target generation with CPG modulation
    """
    
    cfg: "SpiderbotEnvCfg"
    
    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Initialize IK solver
        self.ik_solver = SpiderIK(device=self.device)
        
        # Initialize CPG
        self.cpg = SpiderCPG_Advanced(
            num_envs=self.num_envs,
            dt=self.physics_dt,
            device=self.device,
            k_phase=cfg.cpg_k_phase,
            k_amp=cfg.cpg_k_amp
        )
        
        # Command history buffer for actor observations
        self.cmd_history = torch.zeros(
            self.num_envs, cfg.obs_cmd_hist_len, 3, device=self.device
        )
        
        # Current commands
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.command_timer = torch.zeros(self.num_envs, device=self.device)
        
        # IK targets and CPG states
        self.ik_targets = torch.zeros(self.num_envs, 12, device=self.device)
        self.cpg_phases = torch.zeros(self.num_envs, 4, device=self.device)
        
        # Action smoothing for phase commands
        self.phase_commands = torch.zeros(self.num_envs, 4, device=self.device)
        
        # Tracking metrics
        self.episode_sums = {
            "forward_distance": torch.zeros(self.num_envs, device=self.device),
            "lateral_distance": torch.zeros(self.num_envs, device=self.device),
            "yaw_distance": torch.zeros(self.num_envs, device=self.device),
        }
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions through IK+CPG hybrid control."""
        
        # Parse actions: [freq(1), amp_params(12), phase_offsets(4)]
        frequency = actions[:, 0:1]
        amplitude_params = actions[:, 1:13]
        phase_offsets = actions[:, 13:17]
        
        # Scale actions to appropriate ranges
        frequency = 0.5 + 2.5 * torch.sigmoid(frequency)  # [0.5, 3.0] Hz
        amplitude_params = torch.sigmoid(amplitude_params) * 0.3  # [0, 0.3] rad
        phase_offsets = torch.tanh(phase_offsets) * self.cfg.phase_range_rad
        
        # Smooth phase transitions to prevent jerky movements
        self.phase_commands = (
            self.cfg.phase_beta * phase_offsets + 
            (1 - self.cfg.phase_beta) * self.phase_commands
        )
        
        # Get IK targets based on current commands and gait phases
        self.cpg_phases = (self.cpg_phases + 2 * np.pi * frequency * self.physics_dt) % (2 * np.pi)
        
        # Add phase offsets to base phases
        adjusted_phases = self.cpg_phases + self.phase_commands
        
        # Compute IK targets for nominal trajectory
        self.ik_targets = self.ik_solver.compute_all_legs_ik(
            adjusted_phases,
            self.commands[:, 0],  # vx
            self.commands[:, 1],  # vy
            self.commands[:, 2],  # vyaw
        )
        
        # Apply CPG modulation on top of IK targets
        cpg_deltas = self.cpg.compute_joint_targets(
            frequency, amplitude_params, self.phase_commands
        )
        
        # Combine IK base trajectory with CPG modulation
        joint_targets = self.ik_targets + cpg_deltas
        
        # Apply safety limits
        joint_targets = torch.clamp(joint_targets, -1.5, 1.5)
        
        # Set joint position targets
        self.robot.set_joint_position_target(joint_targets, joint_ids=self._robot_joint_ids)
    
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """
         observations:
        - Actor: Only command history (for sensor-less deployment)
        - Critic: Full state information
        """
        
        # Update command history (shift and add new)
        self.cmd_history[:, 1:] = self.cmd_history[:, :-1].clone()
        self.cmd_history[:, 0] = self.commands
        
        # Actor observations: just flattened command history
        actor_obs = self.cmd_history.view(self.num_envs, -1)
        
        # Critic observations: full state for better value estimation
        base_quat = self.robot.data.root_quat_w
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel
        
        # Orientation in gravity frame
        gravity_vec = quat_rotate_inverse(base_quat, torch.tensor([0, 0, -1], device=self.device))
        
        critic_obs = torch.cat([
            actor_obs,  # Include actor obs
            base_lin_vel,
            base_ang_vel,
            gravity_vec,
            joint_pos,
            joint_vel,
            self.cpg_phases,
            self.ik_targets,
        ], dim=-1)
        
        return {"policy": actor_obs, "critic": critic_obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """Optimized reward function for straight-line tracking and stability."""
        
        # Get base velocities
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        
        # Primary: Velocity tracking with emphasis on forward motion
        vx_error = torch.abs(base_lin_vel[:, 0] - self.commands[:, 0])
        vy_error = torch.abs(base_lin_vel[:, 1] - self.commands[:, 1])
        vyaw_error = torch.abs(base_ang_vel[:, 2] - self.commands[:, 2])
        
        # Use exponential decay for smoother gradient
        velocity_reward = (
            torch.exp(-2.0 * vx_error) * 2.0 +  # Strong forward tracking
            torch.exp(-3.0 * vy_error) * 1.0 +  # Lateral tracking
            torch.exp(-2.0 * vyaw_error) * 1.0  # Yaw tracking
        )
        
        # Penalize vertical motion (hopping)
        z_vel_penalty = torch.abs(base_lin_vel[:, 2]) * self.cfg.z_vel_reward_scale
        
        # Penalize excessive angular velocity (wobbling)
        ang_vel_penalty = torch.norm(base_ang_vel[:, :2], dim=1) * self.cfg.ang_vel_reward_scale
        
        # Orientation penalty (keep robot level)
        base_quat = self.robot.data.root_quat_w
        gravity_vec = quat_rotate_inverse(base_quat, torch.tensor([0, 0, -1], device=self.device))
        orientation_penalty = (1 - gravity_vec[:, 2]) * self.cfg.flat_orientation_reward_scale
        
        # Energy efficiency
        joint_torques = self.robot.data.applied_torque
        torque_penalty = torch.sum(torch.abs(joint_torques), dim=1) * self.cfg.joint_torque_reward_scale
        
        # Action smoothness
        if hasattr(self, 'previous_actions'):
            action_rate = torch.norm(self.actions - self.previous_actions, dim=1)
            action_penalty = action_rate * self.cfg.action_rate_reward_scale
        else:
            action_penalty = 0
        
        self.previous_actions = self.actions.clone()
        
        # Gait regularity bonus (reward consistent phase relationships)
        phase_diff = self.cpg_phases[:, [0, 3]] - self.cpg_phases[:, [1, 2]]  # Diagonal pairs
        phase_regularity = torch.exp(-torch.var(phase_diff, dim=1)) * 0.5
        
        # Combine all rewards
        total_reward = (
            velocity_reward +
            z_vel_penalty +
            ang_vel_penalty +
            orientation_penalty +
            torque_penalty +
            action_penalty +
            phase_regularity
        )
        
        # Store metrics for logging
        self.extras["velocity_reward"] = velocity_reward.mean()
        self.extras["z_vel_penalty"] = z_vel_penalty.mean()
        self.extras["orientation_penalty"] = orientation_penalty.mean()
        
        return total_reward
    
    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """Reset environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        super()._reset_idx(env_ids)
        
        # Reset CPG
        self.cpg.reset(env_ids)
        self.cpg_phases[env_ids] = 0
        
        # Reset commands
        self._sample_commands(env_ids)
        
        # Reset histories
        self.cmd_history[env_ids] = 0
        self.phase_commands[env_ids] = 0
        self.ik_targets[env_ids] = 0
        
        # Reset tracking
        for key in self.episode_sums:
            self.episode_sums[key][env_ids] = 0
    
    def _sample_commands(self, env_ids: torch.Tensor) -> None:
        """Sample new velocity commands with bias toward forward motion."""
        N = len(env_ids)
        
        # 70% straight forward, 20% turning, 10% lateral
        motion_type = torch.rand(N, device=self.device)
        
        # Initialize commands
        vx = torch.zeros(N, device=self.device)
        vy = torch.zeros(N, device=self.device)
        vyaw = torch.zeros(N, device=self.device)
        
        # Straight forward (70%)
        straight_mask = motion_type < 0.7
        vx[straight_mask] = torch.rand(straight_mask.sum(), device=self.device) * 0.3 + 0.1
        
        # Turning (20%)
        turn_mask = (motion_type >= 0.7) & (motion_type < 0.9)
        vx[turn_mask] = torch.rand(turn_mask.sum(), device=self.device) * 0.2
        vyaw[turn_mask] = (torch.rand(turn_mask.sum(), device=self.device) - 0.5) * 0.6
        
        # Lateral/complex (10%)
        complex_mask = motion_type >= 0.9
        vx[complex_mask] = torch.rand(complex_mask.sum(), device=self.device) * 0.2
        vy[complex_mask] = (torch.rand(complex_mask.sum(), device=self.device) - 0.5) * 0.2
        vyaw[complex_mask] = (torch.rand(complex_mask.sum(), device=self.device) - 0.5) * 0.3
        
        self.commands[env_ids, 0] = vx
        self.commands[env_ids, 1] = vy
        self.commands[env_ids, 2] = vyaw
        
        # Reset command timer
        self.command_timer[env_ids] = 0
    
    def _check_termination(self) -> torch.Tensor:
        """Check for early termination conditions."""
        
        # Robot fallen (base too low or tilted)
        base_pos = self.robot.data.root_pos_w
        base_quat = self.robot.data.root_quat_w
        gravity_vec = quat_rotate_inverse(base_quat, torch.tensor([0, 0, -1], device=self.device))
        
        fallen = (base_pos[:, 2] < 0.08) | (gravity_vec[:, 2] < 0.5)
        
        # Joint limits exceeded
        joint_pos = self.robot.data.joint_pos
        joint_limit_exceeded = (torch.abs(joint_pos) > 2.0).any(dim=1)
        
        terminated = fallen | joint_limit_exceeded
        
        return terminated