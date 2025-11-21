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
    """Isaac Lab DirectRLEnv for Spiderbot.

    Key differences from the previous version:
    - Observation is **commands-only** (+ optional history & phase), so deployment requires no sensors.
    - Actuation targets the 12 leg joints defined in the new URDF (configured in SPIDERBOT_CFG).
    - Termination is triggered by **base_link** contact force (no false-positives on feet).
    """
    cfg: SpiderbotEnvCfg

    def __init__(self, cfg: SpiderbotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Action buffers (joint position targets as offsets from default positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)

        # Command buffer: [Vx, Vy, YawRate]
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # === Commands-only observation helpers ===
        self._H = int(getattr(self.cfg, "cmd_history_len", 9))
        self._use_phase = bool(getattr(self.cfg, "use_phase", True))
        self._phase = torch.zeros(self.num_envs, 1, device=self.device)
        self._cmd_hist = torch.zeros(self.num_envs, self._H, 3, device=self.device)

        # Body ids resolved in _setup_scene (after sensors are created)
        self._base_body_ids = None

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

        # Don't resolve body indices here - wait until post_initialization

    # ---------------------------------------------------------------------
    # Control and physics
    # ---------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        # Store latest actions and map to absolute joint position targets
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

        # === Advance open-loop phase (no sensors required) ===
        if self._use_phase:
            base_hz = float(getattr(self.cfg, "phase_base_hz", 1.5))
            kv = float(getattr(self.cfg, "phase_k_v", 1.0))
            speed = torch.norm(self._commands[:, :2], dim=1, keepdim=True)  # m/s
            freq = base_hz + kv * speed  # Hz
            self._phase = (self._phase + 2.0 * math.pi * freq * self.step_dt) % (2.0 * math.pi)

        # === Roll command history (latest at index 0) ===
        if self._H > 0:
            self._cmd_hist = torch.roll(self._cmd_hist, shifts=1, dims=1)
            self._cmd_hist[:, 0, :] = self._commands

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    # ---------------------------------------------------------------------
    # Observations (commands-only)
    # ---------------------------------------------------------------------
    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        parts = [self._commands]
        if self._H > 0:
            parts.append(self._cmd_hist.reshape(self.num_envs, -1))
        if self._use_phase:
            parts += [torch.sin(self._phase), torch.cos(self._phase)]

        obs = torch.cat(parts, dim=-1)
        return {"policy": obs}

    # ---------------------------------------------------------------------
    # Rewards / Dones
    # ---------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        # NOTE: Rewards can use privileged sim state; these are NOT fed to the policy.
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)

        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)

        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)

        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Fallen if base link contacts the world with significant force
        died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._base_body_ids is not None:
            net_contact_forces = self._contact_sensor.data.net_forces_w_history  # [E, H, B, 3]
            base_forces = net_contact_forces[:, :, self._base_body_ids, :]       # [E, H, K, 3]
            force_norm = torch.norm(base_forces, dim=-1)                         # [E, H, K]
            max_over_hist, _ = torch.max(force_norm, dim=1)                      # [E, K]
            max_over_bodies, _ = torch.max(max_over_hist, dim=1)                 # [E]
            died = max_over_bodies > getattr(self.cfg, "base_contact_terminate_force", 1.0)

        return died, time_out

    # ---------------------------------------------------------------------
    # Reset
    # ---------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Resolve base body IDs on first reset (sensor is now initialized)
        if self._base_body_ids is None:
            self._base_body_ids, _ = self._contact_sensor.find_bodies("base_link")

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out resets to avoid spikes in training
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Clear actions/history/phase
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        if self._H > 0:
            self._cmd_hist[env_ids] = 0.0
        self._phase[env_ids] = 0.0

        # Sample new commands for walking (vx forward, small yaw)
        cmds = torch.zeros_like(self._commands[env_ids])
        cmds[:, 0].uniform_(0.0, 0.0)   # vx in m/s
        cmds[:, 1] = 0.0                  # vy
        cmds[:, 2].uniform_(-0.0, 0.0)    # yaw rate
        self._commands[env_ids] = cmds
        if self._H > 0:
            self._cmd_hist[env_ids, 0, :] = cmds

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