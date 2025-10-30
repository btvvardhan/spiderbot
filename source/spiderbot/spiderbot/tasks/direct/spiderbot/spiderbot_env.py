# # spdrbot3_env.py
# # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# from __future__ import annotations

# import gymnasium as gym
# import torch
# import math
# import isaaclab.sim as sim_utils
# from isaaclab.assets import Articulation
# from isaaclab.envs import DirectRLEnv
# from isaaclab.sensors import ContactSensor

# from .spiderbot_env_cfg import SpiderbotEnvCfg
# from .cpg import SpiderCPG


# class SpiderbotEnv(DirectRLEnv):
#     cfg: SpiderbotEnvCfg

#     # ----------------------------
#     # Helpers
#     # ----------------------------
#     def _extract_joint_limits(self):
#         """Return (low, high) joint limits as [num_envs, dof] tensors, robust to different shapes."""
#         limits = self._robot.data.joint_pos_limits
#         if isinstance(limits, (tuple, list)) and len(limits) == 2:
#             low, high = limits
#         elif torch.is_tensor(limits):
#             if limits.ndim >= 2 and limits.shape[0] == 2:
#                 # [2, dof] or [2, ...]
#                 low, high = limits[0], limits[1]
#             elif limits.ndim >= 1 and limits.shape[-1] == 2:
#                 # [..., 2]
#                 low, high = limits[..., 0], limits[..., 1]
#             else:
#                 raise RuntimeError(f"Unexpected joint_pos_limits shape {tuple(limits.shape)}")
#         else:
#             raise RuntimeError(f"Unexpected joint_pos_limits type: {type(limits)}")

#         # Expand to [num_envs, dof] if needed
#         if low.ndim == 1:
#             low = low.unsqueeze(0).expand(self.num_envs, -1)
#             high = high.unsqueeze(0).expand(self.num_envs, -1)
#         return low, high

#     def __init__(self, cfg: SpiderbotEnvCfg, render_mode: str | None = None, **kwargs):
#         super().__init__(cfg, render_mode, **kwargs)

#         # Now robot & scene exist; compute a per-joint amplitude cap from available room.
#         low, high = self._extract_joint_limits()
#         mid = self._robot.data.default_joint_pos
#         room = torch.minimum(high - mid, mid - low)  # [num_envs, dof]
#         self._amp_clip = 0.8 * room                  # leave 20% headroom

#         # Joint position command (deviation from default joint positions)
#         act_dim = gym.spaces.flatdim(self.single_action_space)
#         self._actions = torch.zeros(self.num_envs, act_dim, device=self.device)
#         self._previous_actions = torch.zeros(self.num_envs, act_dim, device=self.device)

#         # CPG parameter buffers
#         self._cpg_frequency = torch.zeros(self.num_envs, 1, device=self.device)   # Hz
#         self._cpg_amplitudes = torch.zeros(self.num_envs, 12, device=self.device) # rad
#         self._cpg_phases = torch.zeros(self.num_envs, 4, device=self.device)      # rad
#         self._previous_frequency = torch.zeros_like(self._cpg_frequency)

#         # Command buffer: [vx, vy, yaw_rate]
#         self._commands = torch.zeros(self.num_envs, 3, device=self.device)

#         # Body indices for contact/termination
#         self._base_id, _ = self._contact_sensor.find_bodies("base_link")
#         self._die_body_ids, _ = self._contact_sensor.find_bodies(
#             ["arm_a_1_1", "arm_a_2_1", "arm_a_3_1", "arm_a_4_1"]
#         )

#         # CPG integrates at the RL step dt
#         self._cpg = SpiderCPG(num_envs=self.num_envs, dt=self.step_dt, device=self.device)

#         # Logging
#         self._episode_sums = {
#             key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
#             for key in [
#                 "track_lin_vel_xy_exp",
#                 "track_ang_vel_z_exp",
#                 "lin_vel_z_l2",
#                 "ang_vel_xy_l2",
#                 "dof_torques_l2",
#                 "dof_acc_l2",
#                 "action_rate_l2",
#                 "flat_orientation_l2",
#                 "freq_smooth",
#             ]
#         }

#     def _setup_scene(self):
#         self._robot = Articulation(self.cfg.robot_cfg)
#         self.scene.articulations["robot"] = self._robot

#         self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
#         self.scene.sensors["contact_sensor"] = self._contact_sensor

#         self.cfg.terrain.num_envs = self.scene.cfg.num_envs
#         self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
#         self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

#         # clone and replicate
#         self.scene.clone_environments(copy_from_source=False)

#         # filter collisions for CPU simulation
#         if self.device == "cpu":
#             self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

#         # add lights
#         light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
#         light_cfg.func("/World/Light", light_cfg)

#     def _pre_physics_step(self, actions: torch.Tensor):
#         # Keep previous actions for rate penalty
#         self._previous_actions = self._actions.clone()
#         self._actions = actions.clone()

#         # Map actions -> CPG parameters
#         freq_raw = actions[:, 0:1]
#         self._cpg_frequency = (
#             self.cfg.cpg_frequency_min
#             + (freq_raw + 1.0) * 0.5 * (self.cfg.cpg_frequency_max - self.cfg.cpg_frequency_min)
#         )

#         amp_raw = actions[:, 1:13]
#         self._cpg_amplitudes = (
#             self.cfg.cpg_amplitude_min
#             + (amp_raw + 1.0) * 0.5 * (self.cfg.cpg_amplitude_max - self.cfg.cpg_amplitude_min)
#         )
#         # Cap amplitudes by per-joint available room to avoid flat-topping when clamped later
#         self._cpg_amplitudes = torch.minimum(self._cpg_amplitudes, self._amp_clip)

#         phase_raw = actions[:, 13:17]
#         self._cpg_phases = (
#             self.cfg.cpg_phase_min
#             + (phase_raw + 1.0) * 0.5 * (self.cfg.cpg_phase_max - self.cfg.cpg_phase_min)
#         )

#         # Compute joint deltas from the CPG
#         joint_deltas = self._cpg.compute_joint_targets(
#             frequency=self._cpg_frequency,        # Hz — ensure CPG converts to rad/s internally
#             amplitudes=self._cpg_amplitudes,      # rad
#             leg_phase_offsets=self._cpg_phases    # rad
#         )

#         # Build final joint targets and clamp to limits
#         self._processed_actions = joint_deltas + self._robot.data.default_joint_pos
#         low, high = self._extract_joint_limits()
#         self._processed_actions = torch.clamp(self._processed_actions, low, high)

#     def _apply_action(self):
#         self._robot.set_joint_position_target(self._processed_actions)

#     def _get_observations(self) -> dict:
#         obs = torch.cat(
#             [
#                 tensor
#                 for tensor in (
#                     self._robot.data.root_lin_vel_b,                               # 3
#                     self._robot.data.root_ang_vel_b,                               # 3
#                     self._robot.data.projected_gravity_b,                          # 3
#                     self._commands,                                                # 3
#                     self._robot.data.joint_pos - self._robot.data.default_joint_pos,  # 12
#                     self._robot.data.joint_vel,                                    # 12
#                     self._actions,                                                 # 17
#                 )
#                 if tensor is not None
#             ],
#             dim=-1,
#         )
#         # in SpiderbotEnv._get_observations, right before return
#         #obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
#         return {"policy": obs}

#     def _get_rewards(self) -> torch.Tensor:
#         # linear velocity tracking (xy)
#         lin_vel_error = torch.sum(
#             torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]),
#             dim=1,
#         )
#         lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)

#         # yaw rate tracking
#         yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
#         yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

#         # z velocity (want near zero)
#         z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])

#         # angular velocity x/y (want small)
#         ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)

#         # joint torques / accelerations
#         joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
#         joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)

#         # action rate (smooth actions)
#         action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)

#         # flat orientation (small xy components of gravity in body frame)
#         flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

#         # frequency smoothness (avoid big freq jumps)
#         freq_change = torch.abs(self._cpg_frequency.squeeze(-1) - self._previous_frequency.squeeze(-1))

#         rewards = {
#             "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
#             "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
#             "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
#             "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
#             "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
#             "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
#             "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
#             "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
#             "freq_smooth": -0.01 * freq_change * self.step_dt,
#         }

#         reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

#         # Logging
#         for key, value in rewards.items():
#             self._episode_sums[key] += value

#         # Update prev frequency AFTER using it for the penalty
#         self._previous_frequency = self._cpg_frequency.clone()

#         return reward

#     def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
#         time_out = self.episode_length_buf >= self.max_episode_length - 1

#         # Tilt-based fail-safe
#         tilt = torch.norm(self._robot.data.projected_gravity_b[:, :2], dim=1)
#         max_tilt = math.sin(math.radians(self.cfg.max_tilt_angle_deg))
#         died = tilt > max_tilt

#         # Optional contact-based fail if configured bodies exist
#         if hasattr(self, "_die_body_ids") and len(self._die_body_ids) > 0:
#             net_contact_forces = self._contact_sensor.data.net_forces_w_history
#             contact_fail = torch.any(
#                 torch.max(torch.norm(net_contact_forces[:, :, self._die_body_ids], dim=-1), dim=1)[0] > 1.0, dim=1
#             )
#             died = torch.logical_or(died, contact_fail)

#         return died, time_out

#     def _reset_idx(self, env_ids: torch.Tensor | None):
#         if env_ids is None or len(env_ids) == self.num_envs:
#             env_ids = self._robot._ALL_INDICES

#         self._robot.reset(env_ids)
#         super()._reset_idx(env_ids)

#         if len(env_ids) == self.num_envs:
#             # Spread out resets to avoid spikes when many envs reset at once
#             self.episode_length_buf[:] = torch.randint_like(
#                 self.episode_length_buf, high=int(self.max_episode_length)
#             )

#         self._cpg.reset(env_ids)

#         # Initialize CPG parameters to CENTER of action range
#         self._cpg_frequency[env_ids] = (self.cfg.cpg_frequency_min + self.cfg.cpg_frequency_max) / 2.0
#         self._cpg_amplitudes[env_ids] = (self.cfg.cpg_amplitude_min + self.cfg.cpg_amplitude_max) / 2.0
#         self._cpg_phases[env_ids] = 0.0

#         self._previous_frequency[env_ids] = self._cpg_frequency[env_ids]
#         self._actions[env_ids] = 0.0
#         self._previous_actions[env_ids] = 0.0

#         # Sample new commands (curriculum)
#         cmds = torch.zeros_like(self._commands[env_ids])
#         self.curriculum_level = 0
#         if self.curriculum_level == 0:
#             # LEVEL 0: Constant slow forward
#             cmds[:, 0] = 0.3
#             cmds[:, 1] = 0.0
#             cmds[:, 2] = 0.0
#         elif self.curriculum_level == 1:
#             # LEVEL 1: Variable forward speed
#             cmds[:, 0].uniform_(0.1, 0.2)
#             cmds[:, 1] = 0.0
#             cmds[:, 2] = 0.0
#         elif self.curriculum_level == 2:
#             # LEVEL 2: Add turning
#             cmds[:, 0].uniform_(0.1, 0.22)
#             cmds[:, 1] = 0.0
#             cmds[:, 2].uniform_(-0.1, 0.1)
#         else:
#             # LEVEL 3+: Full complexity
#             cmds[:, 0].uniform_(0.1, 0.25)
#             cmds[:, 1].uniform_(-0.03, 0.03)
#             cmds[:, 2].uniform_(-0.15, 0.15)
#         self._commands[env_ids] = cmds

#         # Reset robot state
#         joint_pos = self._robot.data.default_joint_pos[env_ids]
#         joint_vel = self._robot.data.default_joint_vel[env_ids]
#         default_root_state = self._robot.data.default_root_state[env_ids]
#         default_root_state[:, :3] += self._terrain.env_origins[env_ids]
#         self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
#         self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
#         self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

#         # Logging
#         extras = dict()
#         for key in self._episode_sums.keys():
#             episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
#             extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
#             self._episode_sums[key][env_ids] = 0.0
#         self.extras["log"] = dict()
#         self.extras["log"].update(extras)
#         extras = dict()
#         extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
#         self.extras["log"].update(extras)


# spdrbot3_env.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

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
        self._cpg_frequency = torch.zeros(self.num_envs, 1, device=self.device)
        self._cpg_amplitudes = torch.zeros(self.num_envs, 12, device=self.device)
        self._cpg_phases = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_frequency = torch.zeros_like(self._cpg_frequency)

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base_link")
        self._die_body_ids, _ = self._contact_sensor.find_bodies(["arm_a_1_1", "arm_a_2_1", "arm_a_3_1", "arm_a_4_1"])


        self._cpg = SpiderCPG(
                    num_envs=self.num_envs,
                    dt=self.step_dt,
                    device=self.device
                )

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

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
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
        
        joint_deltas = self._cpg.compute_joint_targets(
            frequency=self._cpg_frequency,
            amplitudes=self._cpg_amplitudes,
            leg_phase_offsets=self._cpg_phases
        )
        
        self._processed_actions = joint_deltas + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
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
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        # stationary penalty
        lin_vel_norm = torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=1)

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
        died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._die_body_ids], dim=-1), dim=1)[0] > 1.0, dim=1
        )
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        


        self._cpg.reset(env_ids)

        # Initialize CPG parameters to CENTER of action range
        self._cpg_frequency[env_ids] = (self.cfg.cpg_frequency_min + self.cfg.cpg_frequency_max) / 2.0
        self._cpg_amplitudes[env_ids] = (self.cfg.cpg_amplitude_min + self.cfg.cpg_amplitude_max) / 2.0
        self._cpg_phases[env_ids] = 0.0

        self._previous_frequency[env_ids] = self._cpg_frequency[env_ids]
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands
        cmds = torch.zeros_like(self._commands[env_ids])
        self.curriculum_level = 0
        if self.curriculum_level == 0:
            # LEVEL 0: Constant slow forward
            cmds[:, 0] = 0.8
            cmds[:, 1] = 0.0
            cmds[:, 2] = 0.0
        elif self.curriculum_level == 1:
            # LEVEL 1: Variable forward speed
            cmds[:, 0].uniform_(0.1, 0.2)
            cmds[:, 1] = 0.0
            cmds[:, 2] = 0.0
        elif self.curriculum_level == 2:
            # LEVEL 2: Add turning
            cmds[:, 0].uniform_(0.1, 0.22)
            cmds[:, 1] = 0.0
            cmds[:, 2].uniform_(-0.1, 0.1)
        else:
            # LEVEL 3+: Full complexity
            cmds[:, 0].uniform_(0.1, 0.25)
            cmds[:, 1].uniform_(-0.03, 0.03)
            cmds[:, 2].uniform_(-0.15, 0.15)
        self._commands[env_ids] = cmds

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