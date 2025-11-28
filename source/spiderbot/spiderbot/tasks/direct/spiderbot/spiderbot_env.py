# spiderbot_env.py
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

        # Actions and CPG state
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._cpg_frequency = torch.zeros(self.num_envs, 1, device=self.device)
        self._cpg_amplitudes = torch.zeros(self.num_envs, 12, device=self.device)
        self._cpg_phases = torch.zeros(self.num_envs, 4, device=self.device)
        
        # Smoothing memory
        self._previous_frequency = torch.zeros_like(self._cpg_frequency)
        self._prev_amp = torch.zeros_like(self._cpg_amplitudes)
        self._prev_phase = torch.zeros_like(self._cpg_phases)
        
        # Commands (vx, vy, yaw_rate)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Command history
        self._cmd_history_len = int(getattr(self.cfg, "cmd_history_len", 5))
        self._cmd_hist = torch.zeros(self.num_envs, self._cmd_history_len, 3, device=self.device)

        # Body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base_link")
        
        # Termination thresholds (relaxed for learning)
        self._min_base_z = 0.08   # Allow lower crouch
        self._max_tilt_deg = 75.0 # More lenient
        self._max_tilt_cos = math.cos(math.radians(self._max_tilt_deg))
        self._min_contact_force = 15.0  # Lower threshold for 3kg robot
        self._contact_frames_needed = 5
        self._contact_hits = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        
        # CPG
        self._cpg = SpiderCPG(
            num_envs=self.num_envs,
            dt=self.step_dt,
            device=self.device,
            k_phase=0.85,  # Strong diagonal coupling
            k_amp=0.90,
        )

        # ===== CURRICULUM PHASE TRACKING =====
        # Phase 0: Learn stable standing (minimal CPG)
        # Phase 1: Learn forward walking (full CPG)
        self._curriculum_phase = 0
        self._phase_success_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._phase_switch_threshold = 200  # Need 200 stable steps to advance

        # Logging
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
                "alive_bonus",
                "standing_bonus",
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
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone()
        
        # Decode actions to CPG parameters
        freq_raw = actions[:, 0:1]
        amp_raw = actions[:, 1:13]
        phase_raw = actions[:, 13:17]
        
        # Map from [-1, 1] to parameter ranges
        self._cpg_frequency = (
            self.cfg.cpg_frequency_min + 
            (freq_raw + 1.0) * 0.5 * (self.cfg.cpg_frequency_max - self.cfg.cpg_frequency_min)
        )
        
        self._cpg_amplitudes = (
            self.cfg.cpg_amplitude_min +
            (amp_raw + 1.0) * 0.5 * (self.cfg.cpg_amplitude_max - self.cfg.cpg_amplitude_min)
        )
        
        self._cpg_phases = (
            self.cfg.cpg_phase_min +
            (phase_raw + 1.0) * 0.5 * (self.cfg.cpg_phase_max - self.cfg.cpg_phase_min)
        )
        
        # ===== PHASE 0: SUPPRESS CPG FOR STANDING =====
        if self._curriculum_phase == 0:
            # Allow tiny oscillations to find stable pose
            self._cpg_frequency[:] = 0.05  # Very slow
            self._cpg_amplitudes[:] *= 0.1  # 10% of learned amplitude
        
        # Temporal smoothing
        beta_f, beta_a, beta_p = 0.3, 0.3, 0.3
        
        self._cpg_frequency = self._previous_frequency + beta_f * (self._cpg_frequency - self._previous_frequency)
        self._cpg_amplitudes = self._prev_amp + beta_a * (self._cpg_amplitudes - self._prev_amp)
        self._cpg_phases = self._prev_phase + beta_p * (self._cpg_phases - self._prev_phase)
        
        self._previous_frequency = self._cpg_frequency.clone()
        self._prev_amp = self._cpg_amplitudes.clone()
        self._prev_phase = self._cpg_phases.clone()
        
        # Generate joint targets
        joint_deltas = self._cpg.compute_joint_targets(
            frequency=self._cpg_frequency,
            amplitudes=self._cpg_amplitudes,
            leg_phase_offsets=self._cpg_phases
        )
        
        self._processed_actions = joint_deltas + self._robot.data.default_joint_pos
        
        # Update command history
        if self._cmd_history_len > 0:
            self._cmd_hist = torch.roll(self._cmd_hist, shifts=1, dims=1)
            self._cmd_hist[:, 0, :] = self._commands

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        cpg_phase = self._cpg.cpg.phase_angle()
        sin_phase = torch.sin(cpg_phase)
        cos_phase = torch.cos(cpg_phase)
        
        # ACTOR: Sensor-less observations (deployable)
        actor_obs = torch.cat([
            self._commands,
            self._cmd_hist.reshape(self.num_envs, -1),
            sin_phase,
            cos_phase,
            self._previous_actions,
        ], dim=-1)  # 37 dims
        
        # CRITIC: Privileged observations (training only)
        critic_obs = torch.cat([
            actor_obs,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            self._robot.data.joint_vel,
        ], dim=-1)  # 70 dims
        
        return {
            "policy": actor_obs,
            "critic": critic_obs,
        }

    def _get_rewards(self) -> torch.Tensor:
        """Phase-dependent reward shaping."""
        
        if self._curriculum_phase == 0:
            # ===== PHASE 0: STANDING REWARDS =====
            
            # Primary goal: stable upright orientation
            flat_orientation_penalty = torch.sum(
                torch.square(self._robot.data.projected_gravity_b[:, :2]), 
                dim=1
            )
            
            # Secondary goal: minimal motion
            lin_vel_penalty = torch.sum(
                torch.square(self._robot.data.root_lin_vel_b), 
                dim=1
            )
            ang_vel_penalty = torch.sum(
                torch.square(self._robot.data.root_ang_vel_b), 
                dim=1
            )
            
            # Tertiary: smooth joint motions
            joint_vel_penalty = torch.sum(
                torch.square(self._robot.data.joint_vel), 
                dim=1
            )
            joint_torques_penalty = torch.sum(
                torch.square(self._robot.data.applied_torque), 
                dim=1
            )
            
            # Height reward (want ~0.13m standing height for your robot)
            base_z = self._robot.data.root_pos_w[:, 2]
            target_height = 0.13  # 13cm ideal for your dimensions
            height_error = torch.square(base_z - target_height)
            
            # Standing success bonus
            is_standing_well = (
                (flat_orientation_penalty < 0.01) & 
                (lin_vel_penalty < 0.005) & 
                (height_error < 0.01)
            )
            
            # Track success for curriculum advancement
            self._phase_success_count = torch.where(
                is_standing_well,
                self._phase_success_count + 1,
                torch.zeros_like(self._phase_success_count)
            )
            
            rewards = {
                "flat_orientation_l2": flat_orientation_penalty * (-25.0) * self.step_dt,
                "lin_vel_z_l2": (lin_vel_penalty + height_error) * (-15.0) * self.step_dt,
                "ang_vel_xy_l2": ang_vel_penalty * (-8.0) * self.step_dt,
                "dof_acc_l2": joint_vel_penalty * (-3.0) * self.step_dt,
                "dof_torques_l2": joint_torques_penalty * (-2e-5) * self.step_dt,
                "action_rate_l2": torch.zeros_like(base_z) * self.step_dt,
                "standing_bonus": is_standing_well.float() * 8.0 * self.step_dt,
                "alive_bonus": torch.ones_like(base_z) * 1.0 * self.step_dt,
                "track_lin_vel_xy_exp": torch.zeros_like(base_z),
                "track_ang_vel_z_exp": torch.zeros_like(base_z),
            }
            
        else:
            # ===== PHASE 1: WALKING REWARDS =====
            
            lin_vel_error = torch.sum(
                torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), 
                dim=1
            )
            lin_vel_reward = torch.exp(-lin_vel_error / self.cfg.lin_vel_exp_scale)
            
            yaw_rate_error = torch.square(
                self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2]
            )
            yaw_rate_reward = torch.exp(-yaw_rate_error / self.cfg.yaw_rate_exp_scale)
            
            z_vel_penalty = torch.square(self._robot.data.root_lin_vel_b[:, 2])
            ang_vel_xy_penalty = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
            joint_torques_penalty = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
            joint_accel_penalty = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
            action_rate_penalty = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
            flat_orientation_penalty = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

            rewards = {
                "track_lin_vel_xy_exp": lin_vel_reward * self.cfg.lin_vel_reward_scale * self.step_dt,
                "track_ang_vel_z_exp": yaw_rate_reward * self.cfg.yaw_rate_reward_scale * self.step_dt,
                "lin_vel_z_l2": z_vel_penalty * self.cfg.z_vel_reward_scale * self.step_dt,
                "ang_vel_xy_l2": ang_vel_xy_penalty * self.cfg.ang_vel_reward_scale * self.step_dt,
                "dof_torques_l2": joint_torques_penalty * self.cfg.joint_torque_reward_scale * self.step_dt,
                "dof_acc_l2": joint_accel_penalty * self.cfg.joint_accel_reward_scale * self.step_dt,
                "action_rate_l2": action_rate_penalty * self.cfg.action_rate_reward_scale * self.step_dt,
                "flat_orientation_l2": flat_orientation_penalty * self.cfg.flat_orientation_reward_scale * self.step_dt,
                "alive_bonus": torch.ones(self.num_envs, device=self.device) * self.cfg.alive_reward_scale * self.step_dt,
                "standing_bonus": torch.zeros(self.num_envs, device=self.device),
            }
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        
        return reward

    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        base_z = self._robot.data.root_pos_w[:, 2]
        too_low = base_z < self._min_base_z
        
        g_b = self._robot.data.projected_gravity_b
        g_norm = torch.linalg.norm(g_b, dim=1).clamp(min=1e-6)
        cos_tilt = torch.abs(g_b[:, 2]) / g_norm
        too_tilted = cos_tilt < self._max_tilt_cos
        
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
        
        bad_state = ~torch.isfinite(self._robot.data.joint_pos).all(dim=1)
        
        died = too_low | too_tilted | sustained_touch | bad_state
        
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Stagger resets
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, 
                high=int(self.max_episode_length)
            )

        # Reset CPG and counters
        self._cpg.reset(env_ids)
        self._contact_hits[env_ids] = 0
        self._phase_success_count[env_ids] = 0

        # Initialize CPG to mid-range
        self._cpg_frequency[env_ids] = (self.cfg.cpg_frequency_min + self.cfg.cpg_frequency_max) / 2.0
        self._cpg_amplitudes[env_ids] = (self.cfg.cpg_amplitude_min + self.cfg.cpg_amplitude_max) / 2.0
        self._cpg_phases[env_ids] = 0.0
        
        self._prev_amp[env_ids] = self._cpg_amplitudes[env_ids]
        self._prev_phase[env_ids] = self._cpg_phases[env_ids]
        self._previous_frequency[env_ids] = self._cpg_frequency[env_ids]
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        
        if self._cmd_history_len > 0:
            self._cmd_hist[env_ids] = 0.0

        # ===== CURRICULUM: PHASE-DEPENDENT COMMANDS =====
        cmds = torch.zeros_like(self._commands[env_ids])
        
        if self._curriculum_phase == 0:
            # PHASE 0: No movement, just stand
            cmds[:, 0] = 0.0
            cmds[:, 1] = 0.0
            cmds[:, 2] = 0.0
        else:
            # PHASE 1: Forward walking
            cmds[:, 0] = 0.3  # 0.3 m/s forward
            cmds[:, 1] = 0.0
            cmds[:, 2] = 0.0
        
        self._commands[env_ids] = cmds
        
        if self._cmd_history_len > 0:
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

        # ===== CHECK FOR CURRICULUM PHASE ADVANCEMENT =====
        if self._curriculum_phase == 0:
            # Check if majority of envs have mastered standing
            num_ready = torch.sum(self._phase_success_count >= self._phase_switch_threshold).item()
            if num_ready > 0.7 * self.num_envs:  # 70% threshold
                print(f"\n{'='*60}")
                print(f"  CURRICULUM ADVANCEMENT: PHASE 0 → PHASE 1")
                print(f"  {num_ready}/{self.num_envs} envs mastered standing")
                print(f"  Now training forward walking...")
                print(f"{'='*60}\n")
                self._curriculum_phase = 1
                self._phase_success_count[:] = 0  # Reset counter

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        
        # Add curriculum phase info
        extras["Curriculum/phase"] = float(self._curriculum_phase)
        extras["Curriculum/success_count_mean"] = torch.mean(self._phase_success_count.float()).item()
        
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        
        extras = dict()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)