# spiderbot_env_vdp_cpg.py
"""VDP-CPG Spider Bot Environment with IMU-Aware Actor

Key Features:
1. Van der Pol oscillators (superior to Hopf)
2. Actor includes IMU data (deployable with real MPU9250)
3. Richer phase information for better coordination
4. Multi-gait capability through learned phases
5. Asymmetric actor-critic for sensor-less joint control

Actor Observations (52D):
  - Commands + history: 18D (same as pure RL)
  - IMU linear velocity: 3D (deployable!)
  - IMU angular velocity: 3D (deployable!)
  - Projected gravity: 3D (deployable!)
  - CPG phase features: 8D (richer than before)
  - Previous actions: 17D
  
This makes the policy aware of body motion while still being
deployable on hardware (MPU9250 IMU provides all IMU data).
"""

from __future__ import annotations

import math
import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from .spiderbot_env_cfg import SpiderbotEnvCfg
from .cpg import AdaptiveVDPCPG


class SpiderbotEnv(DirectRLEnv):
    """VDP-CPG environment with IMU-aware actor."""
    
    cfg: SpiderbotEnvCfg

    def __init__(self, cfg: SpiderbotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ========== ACTION BUFFERS ==========
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        
        # CPG parameters (unscaled)
        self._cpg_frequency = torch.zeros(self.num_envs, 1, device=self.device)
        self._cpg_amplitudes = torch.zeros(self.num_envs, 12, device=self.device)
        self._cpg_phases = torch.zeros(self.num_envs, 4, device=self.device)
        
        # Smoothing buffers
        self._previous_frequency = torch.zeros_like(self._cpg_frequency)
        self._previous_amplitudes = torch.zeros_like(self._cpg_amplitudes)
        self._previous_phases = torch.zeros_like(self._cpg_phases)
        
        # ========== COMMAND BUFFERS ==========
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._cmd_history_len = self.cfg.cmd_history_len
        self._cmd_hist = torch.zeros(self.num_envs, self._cmd_history_len, 3, device=self.device)

        # ========== VDP CPG ==========
        self._cpg = AdaptiveVDPCPG(
            num_envs=self.num_envs,
            dt=self.step_dt,
            device=self.device,
            mu=self.cfg.cpg_mu,
            k_phase=self.cfg.cpg_k_phase,
            k_amp=self.cfg.cpg_k_amp,
        )

        # ========== TERMINATION DETECTION ==========
        self._base_id, _ = self._contact_sensor.find_bodies("base_link")
        self._min_base_z = 0.06
        self._max_tilt_deg = 65.0
        self._max_tilt_cos = math.cos(math.radians(self._max_tilt_deg))
        self._min_contact_force = 30.0
        self._contact_frames_needed = 3
        self._contact_hits = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # ========== REWARD LOGGING ==========
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "flat_orientation_l2",
                "joint_pos_limits",
                "cpg_phase_coherence",
            ]
        }

    def _setup_scene(self):
        """Setup scene with robot, sensors, and terrain."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        
        # Setup terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
        # Filter collisions for CPU
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions and update CPG.
        
        Actions: [frequency(1), amplitudes(12), phases(4)] = 17D
        """
        # Store previous
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone()
        
        # Parse and scale actions
        freq_raw = actions[:, 0:1]
        amp_raw = actions[:, 1:13]
        phase_raw = actions[:, 13:17]
        
        # Scale to parameter ranges
        freq_new = (
            self.cfg.cpg_frequency_min + 
            (freq_raw + 1.0) * 0.5 * (self.cfg.cpg_frequency_max - self.cfg.cpg_frequency_min)
        )
        amp_new = (
            self.cfg.cpg_amplitude_min +
            (amp_raw + 1.0) * 0.5 * (self.cfg.cpg_amplitude_max - self.cfg.cpg_amplitude_min)
        )
        phase_new = (
            self.cfg.cpg_phase_min +
            (phase_raw + 1.0) * 0.5 * (self.cfg.cpg_phase_max - self.cfg.cpg_phase_min)
        )
        
        # Apply smoothing (low-pass filter)
        beta = self.cfg.action_smoothing_beta
        self._cpg_frequency = self._previous_frequency + beta * (freq_new - self._previous_frequency)
        self._cpg_amplitudes = self._previous_amplitudes + beta * (amp_new - self._previous_amplitudes)
        self._cpg_phases = self._previous_phases + beta * (phase_new - self._previous_phases)
        
        # Update smoothing buffers
        self._previous_frequency = self._cpg_frequency.clone()
        self._previous_amplitudes = self._cpg_amplitudes.clone()
        self._previous_phases = self._cpg_phases.clone()
        
        # Compute CPG joint targets
        joint_deltas = self._cpg.compute_joint_targets(
            frequency=self._cpg_frequency,
            amplitudes=self._cpg_amplitudes,
            leg_phase_offsets=self._cpg_phases
        )
        
        # Add to default positions
        self._processed_actions = joint_deltas + self._robot.data.default_joint_pos
        
        # Update command history
        self._cmd_hist = torch.roll(self._cmd_hist, shifts=1, dims=1)
        self._cmd_hist[:, 0, :] = self._commands

    def _apply_action(self):
        """Apply joint targets to robot."""
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        """Compute asymmetric actor-critic observations.
        
        Actor (IMU-aware, deployable):
            - commands [3]
            - command_history [15]
            - IMU linear velocity [3] ← NEW!
            - IMU angular velocity [3] ← NEW!
            - projected gravity [3] ← NEW!
            - CPG phase features [8] (sin/cos for 4 legs)
            - previous actions [17]
            Total: 52D
            
        Critic (privileged):
            - actor obs [52]
            - joint positions [12]
            - joint velocities [12]
            Total: 76D
        """
        # Get CPG phase features
        cpg_phases = self._cpg.get_phase_features()  # (N,8)
        
        # Clamp IMU data to reasonable ranges (prevent NaN/Inf)
        lin_vel = torch.clamp(self._robot.data.root_lin_vel_b, -10.0, 10.0)
        ang_vel = torch.clamp(self._robot.data.root_ang_vel_b, -10.0, 10.0)
        proj_grav = self._robot.data.projected_gravity_b
        
        # Ensure finite values
        lin_vel = torch.nan_to_num(lin_vel, 0.0)
        ang_vel = torch.nan_to_num(ang_vel, 0.0)
        proj_grav = torch.nan_to_num(proj_grav, 0.0)
        
        # ========== ACTOR OBSERVATIONS (IMU-AWARE) ==========
        actor_obs = torch.cat([
            self._commands,                                # [3]
            self._cmd_hist.reshape(self.num_envs, -1),    # [15]
            lin_vel,                                       # [3] ← IMU data!
            ang_vel,                                       # [3] ← IMU data!
            proj_grav,                                     # [3] ← IMU data!
            cpg_phases,                                     # [8]
            self._previous_actions,                        # [17]
        ], dim=-1)  # Total: 52D
        
        # Clamp joint data
        joint_pos_rel = torch.clamp(
            self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            -3.14, 3.14
        )
        joint_vel = torch.clamp(self._robot.data.joint_vel, -10.0, 10.0)
        
        joint_pos_rel = torch.nan_to_num(joint_pos_rel, 0.0)
        joint_vel = torch.nan_to_num(joint_vel, 0.0)
        
        # ========== CRITIC OBSERVATIONS (PRIVILEGED) ==========
        critic_obs = torch.cat([
            actor_obs,        # [52]
            joint_pos_rel,    # [12]
            joint_vel,        # [12]
        ], dim=-1)  # Total: 76D
        
        # Final safety check
        actor_obs = torch.nan_to_num(actor_obs, 0.0)
        critic_obs = torch.nan_to_num(critic_obs, 0.0)
        
        return {
            "policy": actor_obs,
            "critic": critic_obs,
        }

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards with CPG-specific terms."""
        
        # ========== PRIMARY TRACKING REWARDS ==========
        lin_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), 
            dim=1
        )
        lin_vel_reward = torch.exp(-lin_vel_error / 0.25)
        
        yaw_rate_error = torch.square(
            self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2]
        )
        yaw_rate_reward = torch.exp(-yaw_rate_error / 0.25)
        
        # ========== STABILITY PENALTIES ==========
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        ang_vel_xy_error = torch.sum(
            torch.square(self._robot.data.root_ang_vel_b[:, :2]), 
            dim=1
        )
        flat_orientation_error = torch.sum(
            torch.square(self._robot.data.projected_gravity_b[:, :2]), 
            dim=1
        )
        
        # ========== EFFICIENCY PENALTIES ==========
        joint_torques = torch.sum(
            torch.square(self._robot.data.applied_torque), 
            dim=1
        )
        joint_accel = torch.sum(
            torch.square(self._robot.data.joint_acc), 
            dim=1
        )
        action_rate = torch.sum(
            torch.square(self._actions - self._previous_actions), 
            dim=1
        )
        
        # ========== JOINT LIMITS ==========
        joint_pos = self._robot.data.joint_pos
        joint_limits_low = self._robot.data.soft_joint_pos_limits[:, :, 0]
        joint_limits_high = self._robot.data.soft_joint_pos_limits[:, :, 1]
        
        margin = self.cfg.joint_pos_limit_margin
        dist_to_low = joint_pos - (joint_limits_low + margin)
        dist_to_high = (joint_limits_high - margin) - joint_pos
        
        joint_limit_violation = torch.sum(
            torch.clamp(-dist_to_low, min=0.0) + torch.clamp(-dist_to_high, min=0.0),
            dim=1
        )
        
        # ========== CPG PHASE COHERENCE ==========
        # Reward coherent phase relationships between diagonals
        # Get phase features (N,8) = [sin_FL, cos_FL, ..., sin_RR, cos_RR]
        cpg_phases = self._cpg.get_phase_features()
        
        # Extract sin/cos for each leg
        sin_FL, cos_FL = cpg_phases[:, 0:1], cpg_phases[:, 1:2]
        sin_FR, cos_FR = cpg_phases[:, 2:3], cpg_phases[:, 3:4]
        sin_RL, cos_RL = cpg_phases[:, 4:5], cpg_phases[:, 5:6]
        sin_RR, cos_RR = cpg_phases[:, 6:7], cpg_phases[:, 7:8]
        
        # Diagonal similarity (FL ↔ RR, FR ↔ RL)
        diag0_coherence = (sin_FL - sin_RR)**2 + (cos_FL - cos_RR)**2
        diag1_coherence = (sin_FR - sin_RL)**2 + (cos_FR - cos_RL)**2
        
        # Reward similarity (lower difference = higher reward)
        phase_coherence_reward = -0.5 * (diag0_coherence + diag1_coherence).squeeze(-1)
        
        # ========== AGGREGATE REWARDS ==========
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_reward * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_reward * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_xy_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation_error * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "joint_pos_limits": joint_limit_violation * self.cfg.joint_pos_limit_reward_scale * self.step_dt,
            "cpg_phase_coherence": phase_coherence_reward * self.cfg.cpg_phase_coherence_reward_scale * self.step_dt,
        }
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        for key, value in rewards.items():
            self._episode_sums[key] += value
            
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine episode termination."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Base height
        try:
            base_z = self._robot.data.root_pos_w[:, 2]
        except AttributeError:
            base_z = self._robot.data.root_state_w[:, 2]
        too_low = base_z < self._min_base_z
        
        # Tilt
        g_b = self._robot.data.projected_gravity_b
        g_norm = torch.linalg.norm(g_b, dim=1).clamp(min=1e-6)
        cos_tilt = torch.abs(g_b[:, 2]) / g_norm
        too_tilted = cos_tilt < self._max_tilt_cos
        
        # Sustained base contact
        net_forces = self._contact_sensor.data.net_forces_w_history
        base_id = self._base_id if isinstance(self._base_id, int) else int(self._base_id[0])
        base_force_hist = torch.linalg.norm(net_forces[:, :, base_id], dim=-1)
        base_force_max = torch.max(base_force_hist, dim=1)[0]
        touching = base_force_max > self._min_contact_force
        
        self._contact_hits = torch.where(
            touching, 
            self._contact_hits + 1, 
            torch.zeros_like(self._contact_hits)
        )
        sustained_touch = self._contact_hits >= self._contact_frames_needed
        
        # Bad state
        bad_state = ~torch.isfinite(self._robot.data.joint_pos).all(dim=1)
        
        died = too_low | too_tilted | sustained_touch | bad_state
        
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, 
                high=int(self.max_episode_length)
            )

        # Reset CPG
        self._cpg.reset(env_ids)
        self._contact_hits[env_ids] = 0

        # Reset CPG parameters to center of range
        self._cpg_frequency[env_ids] = (self.cfg.cpg_frequency_min + self.cfg.cpg_frequency_max) / 2.0
        self._cpg_amplitudes[env_ids] = (self.cfg.cpg_amplitude_min + self.cfg.cpg_amplitude_max) / 2.0
        self._cpg_phases[env_ids] = 0.0
        
        self._previous_frequency[env_ids] = self._cpg_frequency[env_ids]
        self._previous_amplitudes[env_ids] = self._cpg_amplitudes[env_ids]
        self._previous_phases[env_ids] = self._cpg_phases[env_ids]
        
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        
        # Reset command history
        self._cmd_hist[env_ids] = 0.0

        # Sample commands (multi-task)
        cmds = torch.zeros_like(self._commands[env_ids])
        cmds[:, 0].uniform_(*self.cfg.command_ranges["lin_vel_x"])
        cmds[:, 1].uniform_(*self.cfg.command_ranges["lin_vel_y"])
        cmds[:, 2].uniform_(*self.cfg.command_ranges["ang_vel_yaw"])
        
        self._commands[env_ids] = cmds
        self._cmd_hist[env_ids, :, :] = cmds.unsqueeze(1).expand(-1, self._cmd_history_len, -1)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Align PD targets
        self._robot.set_joint_position_target(self._robot.data.joint_pos)

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(
            self._contact_hits[env_ids] >= self._contact_frames_needed
        ).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        self.extras["log"].update(extras)