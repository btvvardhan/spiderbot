
# Spiderbot_env.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from .spiderbot_env_cfg import SpiderbotEnvCfg


class SpiderbotEnv(DirectRLEnv):
    """Isaac Lab DirectRLEnv for Spiderbot with commands-only observation and robust terminations."""
    cfg: SpiderbotEnvCfg

    def __init__(self, cfg: SpiderbotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Actions: joint position targets (delta from defaults)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)

        # Commands: [Vx, Vy, YawRate]
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Commands-only observation helpers
        self._H = int(getattr(self.cfg, "cmd_history_len", 9))
        self._use_phase = bool(getattr(self.cfg, "use_phase", True))
        self._use_cmds_only_obs = bool(getattr(self.cfg, "use_commands_only_obs", True))
        self._cmd_hist = torch.zeros(self.num_envs, self._H, 3, device=self.device)
        self._phase = torch.zeros(self.num_envs, 1, device=self.device)
        self._cmd_timer = torch.zeros(self.num_envs, 1, device=self.device)

        # Indices for terminations/rewards (resolved after sensor creation)
        self._base_id = None
        self._feet_ids = None

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
                "pen_base_contact",
                "pen_low_height",
                "stance_contacts",
            ]
        }

    # ---------------------------------------------------------------------
    # Scene setup
    # ---------------------------------------------------------------------
    def _setup_scene(self):
        # Articulation & contact sensor
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # Filter collisions for CPU
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # (Defer body-id resolution to first reset when sensor buffers are ready)

    # ---------------------------------------------------------------------
    # Control and physics
    # ---------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        # Store latest actions and map to absolute joint position targets
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

        # Advance open-loop phase (optional)
        if self._use_phase:
            speed = torch.norm(self._commands[:, :2], dim=1, keepdim=True)  # m/s
            freq = float(self.cfg.phase_base_hz) + float(self.cfg.phase_k_v) * speed
            self._phase = (self._phase + 2.0 * math.pi * freq * self.step_dt) % (2.0 * math.pi)

        # Roll command history (latest at index 0)
        if self._H > 0:
            self._cmd_hist = torch.roll(self._cmd_hist, shifts=1, dims=1)
            self._cmd_hist[:, 0, :] = self._commands

        # Mid-episode command resampling for omni-direction
        self._cmd_timer += self.step_dt
        hold = float(self.cfg.cmd_hold_time_s)
        mask = (self._cmd_timer[:, 0] >= hold)
        if mask.any():
            self._sample_commands(mask)   # <-- missing before (now implemented below)
            self._cmd_timer[mask, 0] = 0.0

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    # ---------------------------------------------------------------------
    # Observations (commands-only)
    # ---------------------------------------------------------------------
    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        if self._use_cmds_only_obs:
            parts = [self._commands]
            if self._H > 0:
                parts.append(self._cmd_hist.reshape(self.num_envs, -1))
            if self._use_phase:
                parts += [torch.sin(self._phase), torch.cos(self._phase)]
            obs = torch.cat(parts, dim=-1)
            return {"policy": obs}

        # Fallback: rich observation (not used for deploy)
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
                )
                if tensor is not None
            ],
            dim=-1,
        )
        return {"policy": obs}

    # ---------------------------------------------------------------------
    # Rewards / Dones
    # ---------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        # Privileged signals (not observed by policy)
        root_lin_b = self._robot.data.root_lin_vel_b
        root_ang_b = self._robot.data.root_ang_vel_b
        g_b = self._robot.data.projected_gravity_b
        base_height = self._robot.data.root_pos_w[:, 2]

        # Contact forces (current step = last in history)
        netF_hist = self._contact_sensor.data.net_forces_w_history  # [E, H, B, 3]
        F_now = netF_hist[:, -1, :, :]                               # [E, B, 3]

        # Base contact magnitude (safe for 1 or many ids)
        if self._base_id is not None and self._base_id.numel() > 0:
            if self._base_id.numel() > 1:
                baseF = torch.norm(F_now[:, self._base_id, :], dim=-1).max(dim=1).values
            else:
                baseF = torch.norm(F_now[:, self._base_id[0], :], dim=-1)
        else:
            baseF = torch.zeros(self.num_envs, device=self.device)

        # Feet contact magnitudes
        if self._feet_ids is not None and self._feet_ids.numel() > 0:
            feetF = torch.norm(F_now[:, self._feet_ids, :], dim=-1)
        else:
            feetF = torch.zeros(self.num_envs, 0, device=self.device)

        # ====== Original tracking and regularization terms ======
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - root_lin_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        yaw_rate_error = torch.square(self._commands[:, 2] - root_ang_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        z_vel_error = torch.square(root_lin_b[:, 2])
        ang_vel_error = torch.sum(torch.square(root_ang_b[:, :2]), dim=1)
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        flat_orientation = torch.sum(torch.square(g_b[:, :2]), dim=1)

        # ====== Anti-belly-sledding terms ======
        base_contact_pen = baseF * float(self.cfg.base_contact_penalty_scale)
        height_deficit = torch.clamp(self.cfg.base_height_target - base_height, min=0.0)
        low_height_pen = height_deficit * float(self.cfg.base_height_low_penalty_scale)
        if feetF.numel() > 0:
            feet_contact = (feetF > float(self.cfg.foot_contact_force_thresh)).float()
            stance_frac = torch.mean(feet_contact, dim=1)
            stance_reward = stance_frac * float(self.cfg.stance_contact_reward_scale)
        else:
            stance_reward = torch.zeros(self.num_envs, device=self.device)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp":  yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2":         z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2":        ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2":       joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2":           joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2":       action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "flat_orientation_l2":  flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "pen_base_contact":     base_contact_pen * self.step_dt,
            "pen_low_height":       low_height_pen * self.step_dt,
            "stance_contacts":      stance_reward * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Robust died conditions
        g_b = self._robot.data.projected_gravity_b
        cos_tilt = (-g_b[:, 2]).clamp(-1.0, 1.0)      # -gz ~ 1 when upright
        cos_max = torch.cos(torch.deg2rad(torch.tensor(self.cfg.max_tilt_angle_deg, device=self.device)))
        too_tilted = cos_tilt < cos_max

        base_height = self._robot.data.root_pos_w[:, 2]
        too_low = base_height < float(self.cfg.min_base_height)

        # Base link contact force over threshold
        netF_hist = self._contact_sensor.data.net_forces_w_history
        F_now = netF_hist[:, -1, :, :]
        if self._base_id is not None and self._base_id.numel() > 0:
            if self._base_id.numel() > 1:
                baseF = torch.norm(F_now[:, self._base_id, :], dim=-1).max(dim=1).values
            else:
                baseF = torch.norm(F_now[:, self._base_id[0], :], dim=-1)
            base_contact = baseF > float(self.cfg.base_contact_force_thresh)
        else:
            base_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Joint angle sanity (optional guard)
        joint_pos = self._robot.data.joint_pos
        joint_violation = torch.any(torch.abs(joint_pos) > float(self.cfg.joint_pos_limit_rad), dim=1)

        # NaN/Inf guard
        def _bad(t):
            t = t.view(self.num_envs, -1)
            return torch.isnan(t).any(dim=1) | torch.isinf(t).any(dim=1)
        bad_state = _bad(self._robot.data.joint_pos) | _bad(self._robot.data.root_lin_vel_b) | _bad(self._actions)

        died = too_tilted | too_low | base_contact | joint_violation | bad_state
        return died, time_out

    # ---------------------------------------------------------------------
    # Reset
    # ---------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Initialize body IDs on first reset (sensor is now ready)
        if self._base_id is None:
            base_ids_list, _ = self._contact_sensor.find_bodies("base_link")
            feet_ids_list, _ = self._contact_sensor.find_bodies(
                ["fl_tibia_link", "fr_tibia_link", "rl_tibia_link", "rr_tibia_link"]
            )
            self._base_id = torch.tensor(base_ids_list, dtype=torch.long, device=self.device)
            self._feet_ids = torch.tensor(feet_ids_list, dtype=torch.long, device=self.device)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out resets to avoid spikes in training
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Clear actions/history/phase
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._cmd_timer[env_ids] = 0.0
        if self._H > 0:
            self._cmd_hist[env_ids] = 0.0
        self._phase[env_ids] = 0.0

        # Sample fresh omni-direction commands
        self._sample_commands(env_ids)
        if self._H > 0:
            self._cmd_hist[env_ids, 0, :] = self._commands[env_ids]

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        extras = dict()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    # ---------------------------------------------------------------------
    # External API
    # ---------------------------------------------------------------------
    def set_commands(self, vx_vy_yaw: torch.Tensor):
        """Set per-env velocity commands (shape: [num_envs, 3])."""
        assert vx_vy_yaw.shape == self._commands.shape
        self._commands[:] = vx_vy_yaw.to(self._commands.device)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _sample_commands(self, env_selector):
        """Sample omni-directional commands into self._commands for selected envs.
        env_selector: boolean mask (N_env,) or env_ids tensor.
        """
        if isinstance(env_selector, torch.Tensor) and env_selector.dtype == torch.bool:
            idx = torch.nonzero(env_selector, as_tuple=False).squeeze(-1)
        else:
            idx = env_selector if isinstance(env_selector, torch.Tensor) else torch.as_tensor(env_selector, device=self.device)

        if idx.numel() == 0:
            return

        vx_min, vx_max = self.cfg.cmd_vx_range
        vy_min, vy_max = self.cfg.cmd_vy_range
        wz_min, wz_max = self.cfg.cmd_yaw_range

        cmds = torch.zeros(idx.numel(), 3, device=self.device)
        # cmds[:, 0].uniform_(float(vx_min), float(vx_max))
        # cmds[:, 1].uniform_(float(vy_min), float(vy_max))
        # cmds[:, 2].uniform_(float(wz_min), float(wz_max))
        cmds[:, 0].uniform_(0.0,0.0)
        cmds[:, 1].uniform_(0.0,0.0)
        cmds[:, 2].uniform_(0.0,0.0)
        self._commands[idx] = cmds