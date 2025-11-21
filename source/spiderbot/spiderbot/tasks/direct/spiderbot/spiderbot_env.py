# Omni-directional Spiderbot CPG-RL env (command-only obs + freq floor + phases + logging).
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

        # --- Actions (policy) -------------------------------------------------
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)

        # CPG buffers
        self._cpg_frequency = torch.zeros(self.num_envs, 1, device=self.device)
        self._cpg_amplitudes = torch.zeros(self.num_envs, 12, device=self.device)
        self._previous_frequency = torch.zeros_like(self._cpg_frequency)
        self._prev_amp = torch.zeros_like(self._cpg_amplitudes)

        # NEW: per-leg phase offsets (Δφ for [FL, FR, RL, RR])
        self._leg_phase_offsets = torch.zeros(self.num_envs, 4, device=self.device)
        self._prev_phase = torch.zeros_like(self._leg_phase_offsets)

        # --- Commands (desired body-frame velocities) ------------------------
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)          # [vx, vy, yaw]
        H = int(self.cfg.obs_cmd_hist_len)
        self._cmd_hist = torch.zeros(self.num_envs, H, 3, device=self.device)
        self._cmd_hist_len = H
        self._cmd_interval = max(1, int(self.cfg.command_change_interval_s / self.step_dt))

        # Terminations
        self._base_id = None
        self._min_base_z = 0.06
        self._max_tilt_deg = 65.0
        self._max_tilt_cos = math.cos(math.radians(self._max_tilt_deg))
        self._min_contact_force = 30.0
        self._contact_frames_needed = 3
        self._contact_hits = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # CPG with diagonal coupling (stabilizer)
        self._cpg = SpiderCPG(
            num_envs=self.num_envs,
            dt=self.step_dt,
            device=self.device,
            k_phase=float(self.cfg.cpg_k_phase),
            k_amp=float(self.cfg.cpg_k_amp),
        )

        # Logging
        self._global_step = 0

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

    # --- Command sampling / history -----------------------------------------
    def _resample_commands(self, env_ids: torch.Tensor):
        # Keep your remap exactly as you had it
        cmds = self._commands[env_ids]
        cmds[:, 0] = 0.3; cmds[:, 1] = 0.0; cmds[:, 2] = 0.0
        cmds_b = torch.zeros_like(cmds)
        cmds_b[:, 0] = -cmds[:, 1]
        cmds_b[:, 1] = -cmds[:, 0]
        cmds_b[:, 2] =  cmds[:, 2]
        self._commands[env_ids] = cmds_b

    def _update_command_history(self, env_ids: torch.Tensor):
        self._cmd_hist[env_ids] = torch.roll(self._cmd_hist[env_ids], shifts=-1, dims=1)
        self._cmd_hist[env_ids, -1, :] = self._commands[env_ids]

    # --- RL loop -------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        # Update commands piecewise-constant
        step_mod = (self.episode_length_buf % self._cmd_interval) == 0
        if torch.any(step_mod):
            self._resample_commands(torch.nonzero(step_mod, as_tuple=False).squeeze(-1))
        self._update_command_history(self._robot._ALL_INDICES)

        # Cache actions
        self._previous_actions.copy_(self._actions)
        self._actions = actions.clone()

        # Map actions: [0]=freq, [1:13]=amps, [13:17]=leg phases
        freq_raw = actions[:, 0:1].clamp(-1.0, 1.0)
        amp_raw  = actions[:, 1:13].clamp(-1.0, 1.0)
        phase_raw = actions[:, 13:17].clamp(-1.0, 1.0)  # (N,4)

        # Frequency [fmin,fmax]
        new_freq = self.cfg.cpg_frequency_min + 0.5 * (freq_raw + 1.0) * (self.cfg.cpg_frequency_max - self.cfg.cpg_frequency_min)

        # Frequency floor from command magnitude
        if getattr(self.cfg, "freq_floor_enable", True):
            v_des = self._commands[:, :2]
            v_max = torch.tensor(
                [max(abs(self.cfg.cmd_vx_min), self.cfg.cmd_vx_max),
                 max(abs(self.cfg.cmd_vy_min), self.cfg.cmd_vy_max)],
                device=self.device
            )
            speed = torch.linalg.norm(v_des / v_max, dim=1, keepdim=True).clamp(0.0, 1.0)
            floor = torch.where(
                speed > 0.0,
                self.cfg.freq_floor_idle + speed * (self.cfg.freq_floor_run - self.cfg.freq_floor_idle),
                torch.zeros_like(speed)
            )
        else:
            floor = torch.zeros_like(new_freq)

        # Amplitudes: abs so random actions move; zero action => standing
        new_amp = torch.abs(amp_raw) * self.cfg.cpg_amplitude_max

        # Per-leg phase offsets: Δφ ∈ [−phase_range, +phase_range] with smoothing
        phase_target = phase_raw * float(self.cfg.phase_range_rad)
        beta_p = float(self.cfg.phase_beta)
        new_phase = self._prev_phase + beta_p * (phase_target - self._prev_phase)

        # Smooth freq/amps, apply floor
        beta_f, beta_a = 0.25, 0.25
        freq_smoothed = self._previous_frequency + beta_f * (new_freq - self._previous_frequency)
        amp_smoothed  = self._prev_amp + beta_a * (new_amp - self._prev_amp)

        self._cpg_frequency = torch.maximum(freq_smoothed, floor)
        self._cpg_amplitudes = amp_smoothed
        self._leg_phase_offsets = new_phase

        self._previous_frequency.copy_(self._cpg_frequency)
        self._prev_amp.copy_(self._cpg_amplitudes)
        self._prev_phase.copy_(self._leg_phase_offsets)

        # Run CPG (adds intra-leg phases + couples diagonals internally)
        joint_deltas = self._cpg.compute_joint_targets(
            frequency=self._cpg_frequency,
            amplitudes=self._cpg_amplitudes,
            leg_phase_offsets=self._leg_phase_offsets,
        )

        # Default pose + oscillator deltas. Zero action => standing.
        self._processed_actions = joint_deltas + self._robot.data.default_joint_pos

        # Logging helpers
        self._last_freq_floor = floor
        self._global_step += 1

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        obs = self._cmd_hist.reshape(self.num_envs, self._cmd_hist_len * 3)
        return {"policy": obs}

    # Rewards / penalties (unchanged projection tracker + stabilizers) --------
    def _get_rewards(self) -> torch.Tensor:
        v_des = self._commands[:, :2]
        v_act = self._robot.data.root_lin_vel_b[:, :2]
        w_des = self._commands[:, 2]
        w_act = self._robot.data.root_ang_vel_b[:, 2]

        eps = 1e-6
        v_des_norm = torch.linalg.norm(v_des, dim=1).clamp(min=eps)
        u_des = v_des / v_des_norm.unsqueeze(1)

        proj = torch.sum(v_act * u_des, dim=1)
        speed_err = (proj - v_des_norm) ** 2
        lateral = v_act - u_des * proj.unsqueeze(1)
        lat_pen = torch.sum(lateral ** 2, dim=1)
        yaw_err = (w_act - w_des) ** 2

        speed_flag = (v_des_norm > 1e-4).float()
        freq_bonus = getattr(self.cfg, "w_freq_bonus", 0.0) * (self._cpg_frequency.squeeze(1) * speed_flag)

        z_vel = (self._robot.data.root_lin_vel_b[:, 2]) ** 2
        ang_vel_xy = torch.sum(self._robot.data.root_ang_vel_b[:, :2] ** 2, dim=1)
        torques_l2 = torch.sum(self._robot.data.applied_torque ** 2, dim=1)
        accel_l2 = torch.sum(self._robot.data.joint_acc ** 2, dim=1)
        action_rate = torch.sum((self._actions - self._previous_actions) ** 2, dim=1)
        flat_orientation = torch.sum(self._robot.data.projected_gravity_b[:, :2] ** 2, dim=1)

        r = (
            self.cfg.w_align * proj
            - self.cfg.w_speed_err * speed_err
            - self.cfg.w_lat * lat_pen
            - self.cfg.w_yaw_err * yaw_err
            + freq_bonus
            + self.cfg.z_vel_reward_scale * z_vel
            + self.cfg.ang_vel_reward_scale * ang_vel_xy
            + self.cfg.joint_torque_reward_scale * torques_l2
            + self.cfg.joint_accel_reward_scale * accel_l2
            + self.cfg.action_rate_reward_scale * action_rate
            + self.cfg.flat_orientation_reward_scale * flat_orientation
        ) * self.step_dt

        # Extras (kept)
        if getattr(self.cfg, "log_to_extras", True):
            freq = self._cpg_frequency.squeeze(1)
            floor = self._last_freq_floor.squeeze(1)
            below = (freq + 1e-6 < floor).float()
            self.extras["metrics/proj_along_cmd"] = proj.mean().item()
            self.extras["metrics/speed_err"] = speed_err.mean().item()
            self.extras["metrics/lat_vel"] = lat_pen.mean().item()
            self.extras["metrics/yaw_err"] = yaw_err.mean().item()
            self.extras["metrics/freq_mean"] = freq.mean().item()
            self.extras["metrics/freq_floor_mean"] = floor.mean().item()
            self.extras["metrics/freq_pct_below_floor"] = below.mean().item()
            self.extras["metrics/amp_mean"] = self._cpg_amplitudes.abs().mean().item()

        if (self._global_step % int(getattr(self.cfg, "log_every_steps", 100))) == 0:
            print(f"[Spiderbot] step={self._global_step} proj={proj.mean().item():.3f} "
                  f"speed_err={speed_err.mean().item():.3f} lat={lat_pen.mean().item():.3f} "
                  f"amp_mean={self._cpg_amplitudes.abs().mean().item():.3f} "
                  f"freq={self._cpg_frequency.mean().item():.3f} floor={self._last_freq_floor.mean().item():.3f}")

        return r

    # Terminations / Reset unchanged -----------------------------------------
    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        try:
            base_z = self._robot.data.root_pos_w[:, 2]
        except AttributeError:
            base_z = self._robot.data.root_state_w[:, 2]
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
        self._contact_hits = torch.where(touching, self._contact_hits + 1, torch.zeros_like(self._contact_hits))
        sustained_touch = self._contact_hits >= self._contact_frames_needed
        bad_state = ~torch.isfinite(self._robot.data.joint_pos).all(dim=1)
        died = too_low | too_tilted | sustained_touch | bad_state
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        if self._base_id is None:
            self._base_id, _ = self._contact_sensor.find_bodies("base_link")
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._cpg.reset(env_ids)
        self._contact_hits[env_ids] = 0

        self._cpg_frequency[env_ids] = (self.cfg.cpg_frequency_min + self.cfg.cpg_frequency_max) / 2.0
        self._cpg_amplitudes[env_ids] = 0.0
        self._prev_amp[env_ids] = 0.0
        self._previous_frequency[env_ids] = self._cpg_frequency[env_ids]
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        self._leg_phase_offsets[env_ids] = 0.0
        self._prev_phase[env_ids] = 0.0

        self._resample_commands(env_ids)
        self._cmd_hist[env_ids] = self._commands[env_ids].unsqueeze(1).repeat(1, self._cmd_hist_len, 1)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._robot.set_joint_position_target(self._robot.data.joint_pos)
