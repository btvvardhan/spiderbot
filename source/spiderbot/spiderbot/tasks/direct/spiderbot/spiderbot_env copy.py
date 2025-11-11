# spiderbot_env.py
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
        self._prev_amp   = torch.zeros(self.num_envs, 12, device=self.device)
        self._prev_phase = torch.zeros(self.num_envs, 4,  device=self.device)
        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base_link")
        # Termination thresholds & counters
        self._min_base_z = 0.06
        self._max_tilt_deg = 65.0
        self._max_tilt_cos = math.cos(math.radians(self._max_tilt_deg))
        self._min_contact_force = 30.0
        self._contact_frames_needed = 3
        self._contact_hits = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
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
        beta_f, beta_a, beta_p = 0.2, 0.2, 0.2

        new_freq  = self._cpg_frequency
        new_amp   = self._cpg_amplitudes
        new_phase = self._cpg_phases

        self._cpg_frequency = self._previous_frequency + beta_f * (new_freq  - self._previous_frequency)
        self._cpg_amplitudes = self._prev_amp          + beta_a * (new_amp   - self._prev_amp)
        self._cpg_phases     = self._prev_phase        + beta_p * (new_phase - self._prev_phase)

        self._previous_frequency = self._cpg_frequency.clone()
        self._prev_amp   = self._cpg_amplitudes.clone()
        self._prev_phase = self._cpg_phases.clone()
        # smooth factors (0..1), smaller = smoother
        
        joint_deltas = self._cpg.compute_joint_targets(
            frequency=self._cpg_frequency,
            amplitudes=self._cpg_amplitudes,
            leg_phase_offsets=self._cpg_phases
        )
        
        
        self._processed_actions = joint_deltas + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
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

    def _get_dones(self):

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # Base height
        try:
            base_z = self._robot.data.root_pos_w[:, 2]
        except AttributeError:
            base_z = self._robot.data.root_state_w[:, 2]
        too_low = base_z < self._min_base_z  
        # Tilt from projected gravity (body up vs world up)
        g_b = self._robot.data.projected_gravity_b  # [E,3], points down in body frame
        g_norm = torch.linalg.norm(g_b, dim=1).clamp(min=1e-6)
        cos_tilt = torch.abs(g_b[:, 2]) / g_norm
        too_tilted = cos_tilt < self._max_tilt_cos
        # Sustained base contact using contact force history (no ground filter needed)
        net_forces = self._contact_sensor.data.net_forces_w_history  # [E,H,B,3]
        base_id = self._base_id if isinstance(self._base_id, int) else int(self._base_id[0])
        base_force_hist = torch.linalg.norm(net_forces[:, :, base_id], dim=-1)  # [E,H]
        base_force_max = torch.max(base_force_hist, dim=1)[0]                   # [E]
        touching = base_force_max > self._min_contact_force
        self._contact_hits = torch.where(touching, self._contact_hits + 1, torch.zeros_like(self._contact_hits))
        sustained_touch = self._contact_hits >= self._contact_frames_needed
        bad_state = ~torch.isfinite(self._robot.data.joint_pos).all(dim=1)
        died = too_low | too_tilted | sustained_touch | bad_state
        return died, time_out
    def _reset_idx(self, env_ids: torch.Tensor | None):
        # Normalize env_ids
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
    
        # Reset robot & parent class
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
    
        # Spread out resets to avoid spikes
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
    
        # Reset CPG state
        self._cpg.reset(env_ids)
        self._contact_hits[env_ids] = 0
    
        # Initialize CPG parameters to CENTER of action range
        self._cpg_frequency[env_ids] = (self.cfg.cpg_frequency_min + self.cfg.cpg_frequency_max) / 2.0
        self._cpg_amplitudes[env_ids] = (self.cfg.cpg_amplitude_min + self.cfg.cpg_amplitude_max) / 2.0
        self._cpg_phases[env_ids] = 0.0
                # after setting _cpg_frequency/_cpg_amplitudes/_cpg_phases
        self._prev_amp[env_ids]   = self._cpg_amplitudes[env_ids]
        self._prev_phase[env_ids] = self._cpg_phases[env_ids]
        # you already do:
        # self._previous_frequency[env_ids] = self._cpg_frequency[env_ids]

        self._previous_frequency[env_ids] = self._cpg_frequency[env_ids]
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
    
        # Sample new commands per curriculum
        cmds = torch.zeros_like(self._commands[env_ids])
        self.curriculum_level = 0
        if self.curriculum_level == 0:
            cmds[:, 0] = 0.3; 
            cmds[:, 1] = 0.0; 
            cmds[:, 2] = 0.0
        elif self.curriculum_level == 1:
            cmds[:, 0].uniform_(0.1, 0.3); 
            cmds[:, 1] = 0.0; 
            cmds[:, 2] = 0.0
        elif self.curriculum_level == 2:
            cmds[:, 0].uniform_(0.1, 0.22); 
            cmds[:, 1] = 0.0; 
            cmds[:, 2].uniform_(-0.1, 0.1)
        else:
            cmds[:, 0].uniform_(-0.5, 0.5); 
            cmds[:, 1].uniform_(-0.1, 0.1); 
            cmds[:, 2].uniform_(-0.3, 0.3)
            
    # user intent -> body frame (Isaac: +Y is left)
        cmds_b = torch.zeros_like(cmds)
        cmds_b[:, 0] =  -cmds[:, 1]   # body X  <= forward
        cmds_b[:, 1] = -cmds[:, 0]   # body Y  <= - right  (because +Y left in Isaac)
        cmds_b[:, 2] =  cmds[:, 2]   # yaw (flip sign later if needed)
        self._commands[env_ids] = cmds_b


        # Reset robot state at env origins
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    
        # This removes the initial jerk by aligning PD targets to the actual pose at reset.
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
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
    
