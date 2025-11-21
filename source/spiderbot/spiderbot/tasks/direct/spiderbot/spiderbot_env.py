# Minimal omni-directional Spiderbot CPG-RL env.
# Observations: command history only (vx, vy, yaw).
# Actions: [frequency, 12 amplitudes], zero => standing.
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
        # 1 freq + 12 amplitudes (robust CPG action space)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)

        # CPG buffers
        self._cpg_frequency = torch.zeros(self.num_envs, 1, device=self.device)
        self._cpg_amplitudes = torch.zeros(self.num_envs, 12, device=self.device)
        self._previous_frequency = torch.zeros_like(self._cpg_frequency)
        self._prev_amp = torch.zeros_like(self._cpg_amplitudes)
        self._zero_leg_offsets = torch.zeros(self.num_envs, 4, device=self.device)  # no per-leg phase actions

        # --- Commands (desired body-frame velocities) ------------------------
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)          # [vx, vy, yaw]
        H = int(self.cfg.obs_cmd_hist_len)
        self._cmd_hist = torch.zeros(self.num_envs, H, 3, device=self.device)       # (N, H, 3)
        self._cmd_hist_len = H
        self._cmd_interval = max(1, int(self.cfg.command_change_interval_s / self.step_dt))

        # Scene bits for terminations
        self._base_id = None
        self._min_base_z = 0.06
        self._max_tilt_deg = 65.0
        self._max_tilt_cos = math.cos(math.radians(self._max_tilt_deg))
        self._min_contact_force = 30.0
        self._contact_frames_needed = 3
        self._contact_hits = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # CPG with configurable coupling (can be turned off via cfg)
        self._cpg = SpiderCPG(
            num_envs=self.num_envs,
            dt=self.step_dt,
            device=self.device,
            k_phase=float(self.cfg.cpg_k_phase),
            k_amp=float(self.cfg.cpg_k_amp),
        )

    # --- Scene setup ---------------------------------------------------------
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

        # Lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Don't call find_bodies here - sensor not initialized yet

    # --- Command sampling / history -----------------------------------------
    def _resample_commands(self, env_ids: torch.Tensor):
        """Piecewise-constant sampling of desired body-frame velocities."""
        cmds = self._commands[env_ids]
        # cmds[:, 0].uniform_(self.cfg.cmd_vx_min, self.cfg.cmd_vx_max)   # vx
        # cmds[:, 1].uniform_(self.cfg.cmd_vy_min, self.cfg.cmd_vy_max)   # vy
        # cmds[:, 2].uniform_(self.cfg.cmd_yaw_min, self.cfg.cmd_yaw_max) # yaw


        self.curriculum_level = 0
        if self.curriculum_level == 0:
            cmds[:, 0] = 0.3; 
            cmds[:, 1] = 0.0; 
            cmds[:, 2] = 0.0
        elif self.curriculum_level == 1:
            cmds[:, 0].uniform_(0.0, 0.5); 
            cmds[:, 1] = 0.0; 
            cmds[:, 2] = 0.0
        elif self.curriculum_level == 2:
            cmds[:, 0].uniform_(0.1, 0.22); 
            cmds[:, 1] = 0.0; 
            cmds[:, 2].uniform_(-0.1, 0.1)
        else:
            cmds[:, 0].uniform_(0.0, 0.5); 
            cmds[:, 1].uniform_(-0.1, 0.1); 
            cmds[:, 2].uniform_(-0.3, 0.3)
            
    # user intent -> body frame (Isaac: +Y is left)
        cmds_b = torch.zeros_like(cmds)
        cmds_b[:, 0] =  -cmds[:, 1]   # body X  <= forward
        cmds_b[:, 1] = -cmds[:, 0]   # body Y  <= - right  (because +Y left in Isaac)
        cmds_b[:, 2] =  cmds[:, 2]   # yaw (flip sign later if needed)
        self._commands[env_ids] = cmds_b


    def _update_command_history(self, env_ids: torch.Tensor):
        self._cmd_hist[env_ids] = torch.roll(self._cmd_hist[env_ids], shifts=-1, dims=1)
        self._cmd_hist[env_ids, -1, :] = self._commands[env_ids]

    # --- RL loop -------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        # Refresh commands piecewise-const
        step_mod = (self.episode_length_buf % self._cmd_interval) == 0  # fixed
        if torch.any(step_mod):
            self._resample_commands(torch.nonzero(step_mod, as_tuple=False).squeeze(-1))
        self._update_command_history(self._robot._ALL_INDICES)

        # Actions -> CPG parameters
        self._previous_actions.copy_(self._actions)
        self._actions = actions.clone()

        # Frequency in [f_min, f_max]
        freq_raw = actions[:, 0:1].clamp(-1.0, 1.0)
        new_freq = self.cfg.cpg_frequency_min + 0.5 * (freq_raw + 1.0) * (self.cfg.cpg_frequency_max - self.cfg.cpg_frequency_min)

        # Amplitudes: robust mapping so zero -> standing (0 amplitude)
        # Use ReLU on [-1,1] then scale to [0, amp_max].
        amp_raw = actions[:, 1:13].clamp(-1.0, 1.0)
        new_amp = torch.relu(amp_raw) * self.cfg.cpg_amplitude_max  # (N,12)

        # Smooth parameters for stability
        beta_f, beta_a = 0.2, 0.2
        self._cpg_frequency = self._previous_frequency + beta_f * (new_freq - self._previous_frequency)
        self._cpg_amplitudes = self._prev_amp + beta_a * (new_amp - self._prev_amp)
        self._previous_frequency.copy_(self._cpg_frequency)
        self._prev_amp.copy_(self._cpg_amplitudes)

        # Run CPG (diagonal coupling handled inside SpiderCPG)
        joint_deltas = self._cpg.compute_joint_targets(
            frequency=self._cpg_frequency,
            amplitudes=self._cpg_amplitudes,
            leg_phase_offsets=self._zero_leg_offsets,  # no leg-phase actions
        )

        # Default pose + oscillator deltas. Zero action => default standing pose.
        self._processed_actions = joint_deltas + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    # --- Observations (policy) -----------------------------------------------
    def _get_observations(self) -> dict:
        # Only command history (vx, vy, yaw), flattened.
        obs = self._cmd_hist.reshape(self.num_envs, self._cmd_hist_len * 3)
        return {"policy": obs}

    # --- Rewards / penalties (simulation states only) ------------------------
    def _get_rewards(self) -> torch.Tensor:
        # Track linear velocity in body frame (xy) to desired commands
        lin_vel_error = torch.sum((self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]) ** 2, dim=1)
        lin_vel_exp = torch.exp(-lin_vel_error / 0.25)

        # Track yaw-rate
        yaw_err = (self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2]) ** 2
        yaw_exp = torch.exp(-yaw_err / 0.25)

        # Soft penalties (stabilizers)
        z_vel = (self._robot.data.root_lin_vel_b[:, 2]) ** 2
        ang_vel_xy = torch.sum(self._robot.data.root_ang_vel_b[:, :2] ** 2, dim=1)
        torques_l2 = torch.sum(self._robot.data.applied_torque ** 2, dim=1)
        accel_l2 = torch.sum(self._robot.data.joint_acc ** 2, dim=1)
        action_rate = torch.sum((self._actions - self._previous_actions) ** 2, dim=1)
        flat_orientation = torch.sum(self._robot.data.projected_gravity_b[:, :2] ** 2, dim=1)

        r = (
            self.cfg.lin_vel_reward_scale * lin_vel_exp
            + self.cfg.yaw_rate_reward_scale * yaw_exp
            + self.cfg.z_vel_reward_scale * z_vel
            + self.cfg.ang_vel_reward_scale * ang_vel_xy
            + self.cfg.joint_torque_reward_scale * torques_l2
            + self.cfg.joint_accel_reward_scale * accel_l2
            + self.cfg.action_rate_reward_scale * action_rate
            + self.cfg.flat_orientation_reward_scale * flat_orientation
        ) * self.step_dt
        return r

    # --- Terminations --------------------------------------------------------
    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Base height check
        try:
            base_z = self._robot.data.root_pos_w[:, 2]
        except AttributeError:
            base_z = self._robot.data.root_state_w[:, 2]
        too_low = base_z < self._min_base_z

        # Tilt (use projected gravity)
        g_b = self._robot.data.projected_gravity_b
        g_norm = torch.linalg.norm(g_b, dim=1).clamp(min=1e-6)
        cos_tilt = torch.abs(g_b[:, 2]) / g_norm
        too_tilted = cos_tilt < self._max_tilt_cos

        # Sustained base contact (from contact sensor history)
        net_forces = self._contact_sensor.data.net_forces_w_history  # [E,H,B,3]
        base_id = self._base_id if isinstance(self._base_id, int) else int(self._base_id[0])
        base_force_hist = torch.linalg.norm(net_forces[:, :, base_id], dim=-1)  # [E,H]
        base_force_max = torch.max(base_force_hist, dim=1)[0]
        touching = base_force_max > self._min_contact_force
        self._contact_hits = torch.where(touching, self._contact_hits + 1, torch.zeros_like(self._contact_hits))
        sustained_touch = self._contact_hits >= self._contact_frames_needed

        bad_state = ~torch.isfinite(self._robot.data.joint_pos).all(dim=1)
        died = too_low | too_tilted | sustained_touch | bad_state
        return died, time_out

    # --- Reset ---------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Initialize base_id on first reset (sensor is now ready)
        if self._base_id is None:
            self._base_id, _ = self._contact_sensor.find_bodies("base_link")

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # Reset CPG state
        self._cpg.reset(env_ids)
        self._contact_hits[env_ids] = 0

        # Initialize CPG parameters so zero action => standing
        self._cpg_frequency[env_ids] = (self.cfg.cpg_frequency_min + self.cfg.cpg_frequency_max) / 2.0
        self._cpg_amplitudes[env_ids] = 0.0
        self._prev_amp[env_ids] = 0.0
        self._previous_frequency[env_ids] = self._cpg_frequency[env_ids]
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # Sample new commands and seed history with same value
        self._resample_commands(env_ids)
        self._cmd_hist[env_ids] = self._commands[env_ids].unsqueeze(1).repeat(1, self._cmd_hist_len, 1)

        # Reset robot pose/vel at env origins
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Align PD targets to actual pose at reset
        self._robot.set_joint_position_target(self._robot.data.joint_pos)
