# spiderbot_env_.py
#  version with yaw bias fixes and better training curriculum

from __future__ import annotations

import math
import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from .spiderbot_env_cfg import SpiderbotEnvCfg
from .cpg import SpiderCPG


class SpiderbotEnv(DirectRLEnv):
    cfg: SpiderbotEnvCfg

    def __init__(self, cfg: SpiderbotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        
        # CPG parameters
        self._cpg_frequency = torch.zeros(self.num_envs, 1, device=self.device)
        self._cpg_amplitudes = torch.zeros(self.num_envs, 12, device=self.device)
        self._cpg_phases = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_frequency = torch.zeros_like(self._cpg_frequency)
        self._prev_amp = torch.zeros(self.num_envs, 12, device=self.device)
        self._prev_phase = torch.zeros(self.num_envs, 4, device=self.device)
        
        # Commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        
        # ✅ NEW: Track yaw direction statistics for bias detection
        self._yaw_direction_sum = torch.zeros(self.num_envs, device=self.device)
        self._yaw_command_sum = torch.zeros(self.num_envs, device=self.device)
        
        # ✅ NEW: Curriculum tracking
        self._curriculum_levels = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._success_buffer = torch.zeros(self.num_envs, device=self.device)
        self._curriculum_thresholds = [0.5, 0.6, 0.7, 0.8]  # Success rate thresholds
        
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base_link")
        
        # Termination thresholds & counters
        self._min_base_z = 0.06
        self._max_tilt_deg = 65.0
        self._max_tilt_cos = math.cos(math.radians(self._max_tilt_deg))
        self._min_contact_force = 30.0
        self._contact_frames_needed = 3
        self._contact_hits = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        
        # Initialize CPG with balanced parameters
        self._cpg = SpiderCPG(
            num_envs=self.num_envs,
            dt=self.step_dt,
            device=self.device,
            k_phase=0.7,  # Moderate phase coupling
            k_amp=0.8     # Strong amplitude coupling for stability
        )

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_x_exp",
                "track_lin_vel_y_exp",
                "lateral_drift_l2",
                "track_ang_vel_z_exp",
                "yaw_drift_l2",
                "yaw_symmetry",  # ✅ NEW
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "flat_orientation_l2",
                "leg_symmetry",  # ✅ NEW
            ]
        }

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone()
        
        # ✅ : Add noise to break symmetry during training
        if self.cfg.add_action_noise:
            noise = torch.randn_like(actions) * 0.01
            actions = actions + noise
        
        # Parse actions
        freq_raw = actions[:, 0:1]
        self._cpg_frequency = (
            self.cfg.cpg_frequency_min + 
            (freq_raw + 1.0) * 0.5 * (self.cfg.cpg_frequency_max - self.cfg.cpg_frequency_min)
        )
        
        amp_raw = actions[:, 1:13]
        self._cpg_amplitudes = (
            self.cfg.cpg_amplitude_min +
            (amp_raw + 1.0) * 0.5 * (self.cfg.cpg_amplitude_max - self.cfg.cpg_amplitude_min)
        )
        
        phase_raw = actions[:, 13:17]
        self._cpg_phases = (
            self.cfg.cpg_phase_min +
            (phase_raw + 1.0) * 0.5 * (self.cfg.cpg_phase_max - self.cfg.cpg_phase_min)
        )
        
        # ✅ : Adaptive smoothing based on curriculum level
        base_smooth = torch.tensor([0.2, 0.2, 0.2], device=self.device)
        curriculum_factor = self._curriculum_levels.float() / 3.0  # 0 to 1
        adaptive_smooth = base_smooth * (1.0 - 0.5 * curriculum_factor.unsqueeze(1))
        
        beta_f = adaptive_smooth[:, 0:1]
        beta_a = adaptive_smooth[:, 1:2]
        beta_p = adaptive_smooth[:, 2:3]
        
        new_freq = self._cpg_frequency
        new_amp = self._cpg_amplitudes
        new_phase = self._cpg_phases
        
        self._cpg_frequency = self._previous_frequency + beta_f * (new_freq - self._previous_frequency)
        self._cpg_amplitudes = self._prev_amp + beta_a * (new_amp - self._prev_amp)
        self._cpg_phases = self._prev_phase + beta_p * (new_phase - self._prev_phase)
        
        self._previous_frequency = self._cpg_frequency.clone()
        self._prev_amp = self._cpg_amplitudes.clone()
        self._prev_phase = self._cpg_phases.clone()
        
        # Compute joint targets
        joint_deltas = self._cpg.compute_joint_targets(
            frequency=self._cpg_frequency,
            amplitudes=self._cpg_amplitudes,
            leg_phase_offsets=self._cpg_phases
        )
        
        self._processed_actions = joint_deltas + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        # ✅ : Add curriculum level to observations
        curriculum_obs = self._curriculum_levels.float().unsqueeze(1) / 3.0
        
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    self._actions,
                    curriculum_obs,  # ✅ NEW
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Velocity tracking rewards
        lin_vel_x_error = torch.square(self._commands[:, 0] - self._robot.data.root_lin_vel_b[:, 0])
        lin_vel_x_mapped = torch.exp(-lin_vel_x_error / 0.25)
        
        lin_vel_y_error = torch.square(self._commands[:, 1] - self._robot.data.root_lin_vel_b[:, 1])
        lin_vel_y_mapped = torch.exp(-lin_vel_y_error / 0.25)
        
        # Lateral drift penalty
        lateral_drift = torch.square(self._robot.data.root_lin_vel_b[:, 1])
        lateral_drift_penalty = lateral_drift * (torch.abs(self._commands[:, 1]) < 0.05).float()
        
        # Yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        
        # ✅ NEW: Yaw symmetry reward - penalize consistent bias
        self._yaw_direction_sum += self._robot.data.root_ang_vel_b[:, 2] * self.step_dt
        yaw_bias = torch.abs(self._yaw_direction_sum) / (self.episode_length_buf.float() + 1.0)
        yaw_symmetry_reward = torch.exp(-yaw_bias * 2.0)  # Encourage balanced yaw
        
        # Yaw drift penalty
        yaw_drift = torch.square(self._robot.data.root_ang_vel_b[:, 2])
        yaw_drift_penalty = yaw_drift * (torch.abs(self._commands[:, 2]) < 0.05).float()
        
        # ✅ NEW: Leg symmetry reward - penalize asymmetric joint positions
        joint_pos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_pos_reshaped = joint_pos.view(-1, 4, 3)  # [N, 4 legs, 3 joints]
        
        # Compare diagonal pairs
        fl_rr_diff = torch.sum(torch.abs(joint_pos_reshaped[:, 0] - joint_pos_reshaped[:, 3]), dim=1)
        fr_rl_diff = torch.sum(torch.abs(joint_pos_reshaped[:, 1] - joint_pos_reshaped[:, 2]), dim=1)
        leg_symmetry = fl_rr_diff + fr_rl_diff
        
        # Other penalties
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        
        # ✅ : Curriculum-based reward scaling
        curriculum_scale = 1.0 + 0.5 * (self._curriculum_levels.float() / 3.0)
        
        rewards = {
            "track_lin_vel_x_exp": lin_vel_x_mapped * self.cfg.lin_vel_x_reward_scale * curriculum_scale * self.step_dt,
            "track_lin_vel_y_exp": lin_vel_y_mapped * self.cfg.lin_vel_y_reward_scale * curriculum_scale * self.step_dt,
            "lateral_drift_l2": lateral_drift_penalty * self.cfg.lateral_drift_penalty_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * curriculum_scale * self.step_dt,
            "yaw_drift_l2": yaw_drift_penalty * self.cfg.yaw_drift_penalty_scale * self.step_dt,
            "yaw_symmetry": yaw_symmetry_reward * 2.0 * self.step_dt,  # ✅ NEW
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "leg_symmetry": -leg_symmetry * 0.05 * self.step_dt,  # ✅ NEW
        }
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        
        return reward

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Base height
        try:
            base_z = self._robot.data.root_pos_w[:, 2]
        except AttributeError:
            base_z = self._robot.data.root_state_w[:, 2]
        too_low = base_z < self._min_base_z
        
        # Tilt from projected gravity
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
        self._contact_hits = torch.where(touching, self._contact_hits + 1, torch.zeros_like(self._contact_hits))
        sustained_touch = self._contact_hits >= self._contact_frames_needed
        
        bad_state = ~torch.isfinite(self._robot.data.joint_pos).all(dim=1)
        died = too_low | too_tilted | sustained_touch | bad_state
        
        return died, time_out

    def _update_curriculum(self, env_ids: torch.Tensor):
        """Update curriculum level based on success rate."""
        if len(env_ids) == 0:
            return
        
        # Calculate success metric (e.g., velocity tracking accuracy)
        vel_error = torch.abs(self._commands[env_ids, 0] - self._robot.data.root_lin_vel_b[env_ids, 0])
        success = (vel_error < 0.1).float()  # Within 0.1 m/s is success
        
        # Update success buffer
        self._success_buffer[env_ids] = 0.9 * self._success_buffer[env_ids] + 0.1 * success
        
        # Check for level advancement
        for idx in env_ids:
            current_level = self._curriculum_levels[idx].item()
            if current_level < 3:  # Max level is 3
                if self._success_buffer[idx] > self._curriculum_thresholds[current_level]:
                    self._curriculum_levels[idx] = min(3, current_level + 1)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # Normalize env_ids
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        
        # Update curriculum before reset
        self._update_curriculum(env_ids)
        
        # Reset robot & parent class
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        # Spread out resets
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        
        # Reset CPG state
        self._cpg.reset(env_ids)
        self._contact_hits[env_ids] = 0
        
        # Reset yaw tracking
        self._yaw_direction_sum[env_ids] = 0.0
        self._yaw_command_sum[env_ids] = 0.0
        
        # Initialize CPG parameters to CENTER of action range
        self._cpg_frequency[env_ids] = (self.cfg.cpg_frequency_min + self.cfg.cpg_frequency_max) / 2.0
        self._cpg_amplitudes[env_ids] = (self.cfg.cpg_amplitude_min + self.cfg.cpg_amplitude_max) / 2.0
        self._cpg_phases[env_ids] = 0.0
        
        self._prev_amp[env_ids] = self._cpg_amplitudes[env_ids]
        self._prev_phase[env_ids] = self._cpg_phases[env_ids]
        self._previous_frequency[env_ids] = self._cpg_frequency[env_ids]
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        
        # ✅ : Progressive curriculum with balanced commands
        cmds = torch.zeros_like(self._commands[env_ids])
        
        for i, env_id in enumerate(env_ids):
            level = self._curriculum_levels[env_id].item()
            
            if level == 0:  # Basic forward motion
                cmds[i, 0] = 0.2
                cmds[i, 1] = 0.0
                cmds[i, 2] = 0.0
                
            elif level == 1:  # Forward with small turns
                cmds[i, 0] = torch.empty(1).uniform_(0.1, 0.3).item()
                cmds[i, 1] = 0.0
                # ✅ CRITICAL: Alternating yaw commands to prevent bias
                if torch.rand(1) > 0.5:
                    cmds[i, 2] = torch.empty(1).uniform_(0.0, 0.15).item()
                else:
                    cmds[i, 2] = torch.empty(1).uniform_(-0.15, 0.0).item()
                    
            elif level == 2:  # Mixed forward/lateral with turns
                cmds[i, 0] = torch.empty(1).uniform_(0.0, 0.35).item()
                cmds[i, 1] = torch.empty(1).uniform_(-0.05, 0.05).item()
                # ✅ Balanced yaw distribution
                cmds[i, 2] = torch.empty(1).uniform_(-0.2, 0.2).item()
                
            else:  # level == 3: Full capability
                cmds[i, 0] = torch.empty(1).uniform_(-0.2, 0.5).item()
                cmds[i, 1] = torch.empty(1).uniform_(-0.1, 0.1).item()
                # ✅ Ensure symmetric yaw distribution
                cmds[i, 2] = torch.empty(1).uniform_(-0.3, 0.3).item()
        
        # Track commanded yaw for bias detection
        self._yaw_command_sum[env_ids] = cmds[:, 2]
        
        # ✅ FIXED: Correct coordinate transformation
        # Isaac Sim convention: X=forward, Y=left, Z=up
        # No need for complex transformation - keep it simple
        self._commands[env_ids, 0] = -cmds[:, 1]  # Forward velocity
        self._commands[env_ids, 1] = -cmds[:, 0]  # Lateral velocity  
        self._commands[env_ids, 2] = cmds[:, 2]  # Yaw rate
        
        # Reset robot state at env origins
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
        # ✅ NEW: Randomly initialize yaw to break symmetry
        random_yaw = torch.empty(len(env_ids), device=self.device).uniform_(-0.1, 0.1)
        # Convert to quaternion (small angle approximation)
        default_root_state[:, 3] = torch.cos(random_yaw / 2)  # qw
        default_root_state[:, 6] = torch.sin(random_yaw / 2)  # qz
        
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
        
        # ✅ NEW: Log yaw bias statistics
        if len(env_ids) > 0:
            avg_yaw_bias = torch.mean(torch.abs(self._yaw_direction_sum[env_ids])).item()
            extras["Episode_Stats/yaw_bias"] = avg_yaw_bias
            extras["Episode_Stats/curriculum_level"] = torch.mean(self._curriculum_levels[env_ids].float()).item()
        
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)