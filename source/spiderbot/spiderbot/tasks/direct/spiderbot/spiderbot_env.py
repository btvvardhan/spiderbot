# spiderbot_env.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Spider Robot CPG-RL Environment
================================

BIO-INSPIRED APPROACH:
----------------------
Instead of directly controlling joint positions, the RL policy learns to modulate
Central Pattern Generators (CPGs) that produce rhythmic leg movements.

This mirrors how real spiders work:
- Spiders have neural oscillators (CPGs) that generate walking rhythms
- Higher brain centers modulate these oscillators (frequency, amplitude)
- Coordination emerges from phase coupling between oscillators

KEY DIFFERENCES FROM PURE RL:
------------------------------
1. Actions are CPG PARAMETERS, not joint positions
2. CPG layer generates smooth, rhythmic joint trajectories
3. Natural gaits emerge from oscillator coupling
4. More robust and bio-realistic locomotion
"""

from __future__ import annotations

import gymnasium as gym
import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from .spiderbot_env_cfg import SpiderbotEnvCfg
from .cpg import SpiderCPG  # ← NEW: Import our CPG module


class SpiderbotEnv(DirectRLEnv):
    """Spider robot environment with CPG-RL."""
    
    cfg: SpiderbotEnvCfg

    def __init__(self, cfg: SpiderbotEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ============================================
        # CPG PARAMETERS (17 total)
        # ============================================
        # Actions now represent CPG parameters, not joint positions
        # [0]: frequency
        # [1-12]: amplitudes
        # [13-16]: leg phase offsets
        self._actions = torch.zeros(
            self.num_envs, 
            gym.spaces.flatdim(self.single_action_space), 
            device=self.device
        )
        self._previous_actions = torch.zeros_like(self._actions)
        
        # Track processed CPG parameters separately for rewards
        self._cpg_frequency = torch.zeros(self.num_envs, 1, device=self.device)
        self._cpg_amplitudes = torch.zeros(self.num_envs, 12, device=self.device)
        self._cpg_phases = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_frequency = torch.zeros_like(self._cpg_frequency)

        # ============================================
        # VELOCITY COMMANDS
        # ============================================
        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # ============================================
        # BODY INDICES FOR CONTACT SENSING
        # ============================================
        self._base_id, _ = self._contact_sensor.find_bodies("base_link")
        # Bodies that shouldn't touch ground (fail condition)
        self._die_body_ids, _ = self._contact_sensor.find_bodies([
            "arm_a_1_1", "arm_a_2_1", "arm_a_3_1", "arm_a_4_1"
        ])
        
        # ============================================
        # NEW: INITIALIZE CPG
        # ============================================
        # This is the core of bio-inspired locomotion
        # CPG runs at physics frequency (200 Hz) to generate smooth trajectories
        self._cpg = SpiderCPG(
            num_envs=self.num_envs,
            dt=self.physics_dt,  # Use physics dt (1/200 = 0.005s), not decimated dt
            device=self.device
        )
        

                # ============================================
        # FOOT (END-EFFECTOR) TRACKING
        # ============================================
        # Feet are the last link (arm_c) of each leg chain
        # Leg structure: arm_a (hip) → arm_b (knee) → arm_c (foot)
        self._foot_ids, _ = self._contact_sensor.find_bodies([
            "arm_c_1_1",  # Leg 1 foot (last link)
            "arm_c_2_1",  # Leg 2 foot (last link)
            "arm_c_3_1",  # Leg 3 foot (last link)
            "arm_c_4_1",  # Leg 4 foot (last link)
        ])

        # ============================================
        # LOGGING
        # ============================================
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                # Existing rewards
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "flat_orientation_l2",
                # NEW: CPG-specific rewards
                "frequency_change",
                "gait_symmetry",
                "contact_timing",
                "foot_slip",
            ]
        }

    def _setup_scene(self):
        """Setup the simulation scene."""
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot
        
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
        # Filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """
        Process actions and convert to joint targets using CPG.
        
        This is where the magic happens:
        1. RL outputs CPG parameters (frequency, amplitude, phase)
        2. CPG generates smooth oscillatory joint patterns
        3. Joint patterns sent to robot
        
        Args:
            actions: [num_envs, 17] - CPG parameters from policy
        """
        self._actions = actions.clone()
        
        # ============================================
        # DECODE CPG PARAMETERS FROM ACTIONS
        # ============================================
        # Actions come from NN in range [-1, 1]
        # We need to scale them to meaningful CPG parameter ranges
        
        # [0]: Frequency (rad/s)
        # Map [-1, 1] → [freq_min, freq_max]
        freq_raw = actions[:, 0:1]  # [num_envs, 1]
        self._cpg_frequency = (
            self.cfg.cpg_frequency_min + 
            (freq_raw + 1.0) * 0.5 * (self.cfg.cpg_frequency_max - self.cfg.cpg_frequency_min)
        )
        
        # [1-12]: Amplitudes (radians)
        # Map [-1, 1] → [amp_min, amp_max]
        amp_raw = actions[:, 1:13]  # [num_envs, 12]
        self._cpg_amplitudes = (
            self.cfg.cpg_amplitude_min +
            (amp_raw + 1.0) * 0.5 * (self.cfg.cpg_amplitude_max - self.cfg.cpg_amplitude_min)
        )
        
        # [13-16]: Leg phase offsets (radians)
        # Map [-1, 1] → [phase_min, phase_max]
        phase_raw = actions[:, 13:17]  # [num_envs, 4]
        self._cpg_phases = (
            self.cfg.cpg_phase_min +
            (phase_raw + 1.0) * 0.5 * (self.cfg.cpg_phase_max - self.cfg.cpg_phase_min)
        )
        
        # ============================================
        # GENERATE JOINT TARGETS FROM CPG
        # ============================================
        # CPG produces smooth oscillations based on current parameters
        joint_deltas = self._cpg.compute_joint_targets(
            frequency=self._cpg_frequency,
            amplitudes=self._cpg_amplitudes,
            leg_phase_offsets=self._cpg_phases
        )
        
        # Add oscillations to default joint positions
        # This means robot oscillates around its neutral pose
        self._processed_actions = joint_deltas + self._robot.data.default_joint_pos

    def _apply_action(self):
        """Send computed joint targets to robot."""
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        """
        Construct observation vector for policy.
        
        Observations include:
        - Robot state (velocities, orientation)
        - Velocity commands (what we want robot to do)
        - Joint positions and velocities
        - Previous actions (for temporal awareness)
        
        Returns:
            Dictionary with "policy" key containing observation tensor
        """
        self._previous_actions = self._actions.clone()
        
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,       # [3] - linear velocity in body frame
                    self._robot.data.root_ang_vel_b,       # [3] - angular velocity in body frame
                    self._robot.data.projected_gravity_b,  # [3] - gravity direction (for orientation)
                    self._commands,                        # [3] - desired velocities
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,  # [12] - joint offsets
                    self._robot.data.joint_vel,            # [12] - joint velocities
                    self._actions,                         # [17] - previous CPG parameters
                )
                if tensor is not None
            ],
            dim=-1,
        )
        # Total: 3+3+3+3+12+12+17 = 53... wait, cfg says 48?
        # You may need to adjust observation_space in cfg or remove some observations
        
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """
        Compute reward signal for RL.
        
        Rewards encourage:
        1. Following velocity commands (task objective)
        2. Stable, efficient locomotion (energy, smoothness)
        3. Bio-inspired gait patterns (symmetry, coordination)
        
        Returns:
            Total reward per environment [num_envs]
        """
        
        # ============================================
        # VELOCITY TRACKING REWARDS
        # ============================================
        # Reward tracking desired linear velocity (x, y)
        lin_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), 
            dim=1
        )
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        
        # Reward tracking desired yaw rate
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        
        # ============================================
        # STABILITY PENALTIES
        # ============================================
        # Penalize vertical velocity (should stay at constant height)
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        
        # Penalize angular velocity in x/y (should not roll/pitch)
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        
        # Penalize large torques (energy efficiency)
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        
        # Penalize joint acceleration (smoothness)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        
        # Penalize action rate (smooth CPG parameter changes)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        
        # Penalize tilting (keep body upright)
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        
        # ============================================
        # NEW: CPG-SPECIFIC REWARDS
        # ============================================
        
        # 1. Smooth frequency changes
        # Biological gaits don't change frequency abruptly
        frequency_change = torch.sum(
            torch.square(self._cpg_frequency - self._previous_frequency), 
            dim=1
        )
        self._previous_frequency = self._cpg_frequency.clone()
        
        # 2. Gait symmetry
        # Left and right legs should have similar amplitudes (balanced gait)
        # Assuming leg order: FL, FR, BL, BR (each with 3 joints)
        left_amps = self._cpg_amplitudes[:, [0,1,2, 6,7,8]]   # FL + BL
        right_amps = self._cpg_amplitudes[:, [3,4,5, 9,10,11]]  # FR + BR
        #gait_symmetry = torch.sum(torch.square(left_amps.mean(dim=1) - right_amps.mean(dim=1)))
        gait_symmetry = torch.square(left_amps.mean(dim=1) - right_amps.mean(dim=1))

        # 3. Contact timing reward
        # Feet should touch ground when CPG is in stance phase
        # This requires computing expected contact from CPG phase
        # Simplified: reward having 2-3 feet in contact (stable tetrapod)
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        foot_contacts = torch.norm(net_contact_forces[:, :, self._foot_ids], dim=-1)  # [num_envs, history, 4_feet]
        feet_in_contact = (foot_contacts[:, -1, :] > 1.0).sum(dim=1)  # Count feet touching ground
        # Ideal: 2-3 feet in contact (tetrapod gait)
        contact_timing = -torch.square(feet_in_contact.float() - 2.5)  # Peak reward at 2.5 feet
        
        # 4. Foot slip penalty
        # During stance phase, feet should not slide
        # Simplified: penalize foot velocity when in contact
        # This requires foot body velocities - may need to add to observations
        # For now, use a placeholder (you can implement later)
        foot_slip = torch.zeros(self.num_envs, device=self.device)
        
        # ============================================
        # COMBINE ALL REWARDS
        # ============================================
        rewards = {
            # Existing rewards
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            # NEW: CPG rewards
            "frequency_change": frequency_change * self.cfg.frequency_change_reward_scale * self.step_dt,
            "gait_symmetry": gait_symmetry * self.cfg.gait_symmetry_reward_scale * self.step_dt,
            "contact_timing": contact_timing * self.cfg.contact_timing_reward_scale * self.step_dt,
            "foot_slip": foot_slip * self.cfg.foot_slip_reward_scale * self.step_dt,
        }
        
        # Sum all rewards
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
            
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Determine which episodes should terminate.
        
        Termination conditions:
        1. Timeout: Episode reached max length
        2. Failure: Body parts that shouldn't touch ground made contact
        
        Returns:
            died: [num_envs] - True if robot failed
            time_out: [num_envs] - True if episode timed out
        """
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Check if die bodies (upper leg links) touched ground
        died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._die_body_ids], dim=-1), dim=1)[0] > 1.0, 
            dim=1
        )
        
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """
        Reset specific environments.
        
        Called when:
        - Episode times out
        - Robot fails (falls over)
        - At start of training
        
        Args:
            env_ids: Indices of environments to reset
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
            
        # Reset robot articulation
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        if len(env_ids) == self.num_envs:
            # Stagger resets to avoid training spikes
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, 
                high=int(self.max_episode_length)
            )
        
        # Reset actions
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        
        # ============================================
        # RESET CPG STATE
        # ============================================
        # Important: CPG oscillators need to be reset to avoid
        # carrying over state from previous episode
        self._cpg.reset(env_ids)
        
        # Reset CPG parameter tracking
        self._cpg_frequency[env_ids] = 2.0  # Start at moderate frequency
        self._cpg_amplitudes[env_ids] = 0.2  # Start at moderate amplitude
        self._cpg_phases[env_ids] = 0.0
        self._previous_frequency[env_ids] = 2.0
        
        # # ============================================
        # # SAMPLE NEW VELOCITY COMMANDS
        # # ============================================
        # # Give robot a new task each episode
        # cmds = torch.zeros_like(self._commands[env_ids])
        # cmds[:, 0].uniform_(0.15, 0.35)  # Forward velocity: 0.15-0.35 m/s
        # cmds[:, 1] = 0.0                 # No lateral movement
        # cmds[:, 2].uniform_(-0.2, 0.2)   # Small turning: ±0.2 rad/s
        # self._commands[env_ids] = cmds



    current_iteration = self.episode_length_buf.max().item() // 1000  # Rough estimate
    
    cmds = torch.zeros_like(self._commands[env_ids])
    
    if current_iteration < 200:
        # Phase 1: Very slow walking (natural CPG speeds)
        cmds[:, 0].uniform_(0.05, 0.15)  # Slow forward: 0.05-0.15 m/s
        cmds[:, 1] = 0.0                  # No lateral
        cmds[:, 2].uniform_(-0.1, 0.1)    # Tiny turning
    elif current_iteration < 500:
        # Phase 2: Normal walking
        cmds[:, 0].uniform_(0.15, 0.35)  # Normal forward: 0.15-0.35 m/s
        cmds[:, 1] = 0.0                  # No lateral
        cmds[:, 2].uniform_(-0.2, 0.2)    # Small turning
    else:
        # Phase 3: Fast walking + omnidirectional
        cmds[:, 0].uniform_(0.15, 0.50)   # Fast forward
        cmds[:, 1].uniform_(-0.15, 0.15)  # Add lateral movement
        cmds[:, 2].uniform_(-0.3, 0.3)    # More turning
    
    self._commands[env_ids] = cmds
        
        # ============================================
        # RESET ROBOT STATE IN SIMULATION
        # ============================================
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # ============================================
        # LOGGING
        # ============================================
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