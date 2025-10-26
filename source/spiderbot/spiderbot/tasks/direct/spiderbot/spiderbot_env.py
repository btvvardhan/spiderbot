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

        self._actions = torch.zeros(
            self.num_envs, 
            gym.spaces.flatdim(self.single_action_space), 
            device=self.device
        )
        self._previous_actions = torch.zeros_like(self._actions)
        
        self._cpg_frequency = torch.zeros(self.num_envs, 1, device=self.device)
        self._cpg_amplitudes = torch.zeros(self.num_envs, 12, device=self.device)
        self._cpg_phases = torch.zeros(self.num_envs, 4, device=self.device)
        self._previous_frequency = torch.zeros_like(self._cpg_frequency)

        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        self._initial_root_pos = torch.zeros(self.num_envs, 3, device=self.device)
        
        self._previous_root_pos = torch.zeros(self.num_envs, 2, device=self.device)
        
        self._base_id, _ = self._contact_sensor.find_bodies("base_link")
        
        self._die_body_ids, _ = self._contact_sensor.find_bodies([
            "arm_a_1_1", "arm_a_2_1", "arm_a_3_1", "arm_a_4_1"
        ])
        
        self._foot_ids, _ = self._contact_sensor.find_bodies([
            "arm_c_1_1",
            "arm_c_2_1",
            "arm_c_3_1",
            "arm_c_4_1",
        ])
        
        self._cpg = SpiderCPG(
            num_envs=self.num_envs,
            dt=self.physics_dt,
            device=self.device
        )

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
        
        self.scene.clone_environments(copy_from_source=False)
        
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
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
        
        base_height = self._robot.data.root_pos_w[:, 2:3]
        
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._commands,
                base_height,
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                self._robot.data.joint_vel,
                self._actions,
            ],
            dim=-1,
        )
        
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        current_pos = self._robot.data.root_pos_w[:, :2]
        displacement = current_pos - self._previous_root_pos
        
        displacement_magnitude = torch.norm(displacement, dim=1)
        lin_vel_reward = displacement_magnitude * self.cfg.lin_vel_reward_scale
        
        self._previous_root_pos = current_pos.clone()
        
        yaw_rate_error = torch.square(
            self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2]
        )
        yaw_rate_reward = torch.exp(-yaw_rate_error / 0.25) * self.cfg.yaw_rate_reward_scale
        
        z_vel_penalty = torch.square(
            self._robot.data.root_lin_vel_b[:, 2]
        ) * self.cfg.z_vel_reward_scale
        
        ang_vel_penalty = torch.sum(
            torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1
        ) * self.cfg.ang_vel_reward_scale
        
        flat_orientation_penalty = torch.sum(
            torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1
        ) * self.cfg.flat_orientation_reward_scale
        
        joint_torque_penalty = torch.sum(
            torch.square(self._robot.data.applied_torque), dim=1
        ) * self.cfg.joint_torque_reward_scale
        
        joint_accel_penalty = torch.sum(
            torch.square(self._robot.data.joint_acc), dim=1
        ) * self.cfg.joint_accel_reward_scale
        
        action_rate_penalty = torch.mean(
            torch.square(self._actions - self._previous_actions), dim=1
        ) * self.cfg.action_rate_reward_scale
        
        rewards = {
            "track_lin_vel_xy_exp": lin_vel_reward ,
            "track_ang_vel_z_exp": yaw_rate_reward * self.step_dt,
            "lin_vel_z_l2": z_vel_penalty * self.step_dt,
            "ang_vel_xy_l2": ang_vel_penalty * self.step_dt,
            "dof_torques_l2": joint_torque_penalty * self.step_dt,
            "dof_acc_l2": joint_accel_penalty * self.step_dt,
            "action_rate_l2": action_rate_penalty * self.step_dt,
            "flat_orientation_l2": flat_orientation_penalty * self.step_dt,
        }
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        for key, value in rewards.items():
            self._episode_sums[key] += value
        
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        
        contact_norms = torch.norm(net_contact_forces[:, :, self._die_body_ids], dim=-1)
        max_contact = torch.max(contact_norms, dim=1)[0]
        
        died = torch.any(max_contact > 1000.0, dim=1)
        
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        
        super()._reset_idx(env_ids)
        
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = 0
        
        self._cpg.reset(env_ids)
        
        self._cpg_frequency[env_ids] = 1.0
        
        self._cpg_amplitudes[env_ids] = 0.5
        
        self._cpg_phases[env_ids, 0] = 0.0
        self._cpg_phases[env_ids, 1] = 3.14159
        self._cpg_phases[env_ids, 2] = 3.14159
        self._cpg_phases[env_ids, 3] = 0.0
        
        self._previous_frequency[env_ids] = self._cpg_frequency[env_ids]
        
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        
        cmds = torch.zeros_like(self._commands[env_ids])
        cmds[:, 0].uniform_(0.1, 0.25)
        cmds[:, 1] = 0.0
        cmds[:, 2].uniform_(-0.15, 0.15)
        self._commands[env_ids] = cmds
        
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        self._initial_root_pos[env_ids] = default_root_state[:, :3].clone()
        
        self._previous_root_pos[env_ids] = default_root_state[:, :2].clone()
        
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        
        extras = dict()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        self.extras["log"].update(extras)