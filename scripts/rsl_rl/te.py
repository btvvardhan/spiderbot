# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL with keyboard control.

Launch Isaac Sim Simulator first.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import omni  # Import omni modules

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL using keyboard control.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--max_lin_vel", type=float, default=1.0, help="Maximum linear velocity for keyboard control.")
parser.add_argument("--max_ang_vel", type=float, default=1.0, help="Maximum angular velocity for keyboard control.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import carb
import csv
import socket
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import spiderbot.tasks  # noqa: F401


class CarbKeyboardController:
    """
    Keyboard controller using Carb (Isaac Sim native input).

    Controls:
        W/S: Forward/Backward
        A/D: Left/Right lateral movement
        Q/E: Rotate Left/Right (yaw)
        SPACE: Emergency stop (zero all velocities)
        ESC: Exit simulation
        R: Reset velocities to zero
        UP/DOWN Arrow: Increase/Decrease velocity scale
        F5: Reset simulation (restart episode)
    """

    def __init__(self, max_lin_vel=1.0, max_ang_vel=1.0, vel_increment=0.15):
        """
        Initialize keyboard controller using Carb input.
        """
        self.cmd_vel = [0.0, 0.0, 0.0]  # [forward, lateral, yaw]
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        self.vel_increment = vel_increment
        self.vel_scale = 1.0

        # Get Carb input interface
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )

        self.should_exit = False
        self.request_env_reset = False
        self._last_scale_change_time = 0.0
        self._scale_change_cooldown = 0.2  # seconds

        self._print_controls()

    def _print_controls(self):
        print("\n" + "=" * 60)
        print("  SPIDERBOT KEYBOARD CONTROLS (Carb Native)")
        print("=" * 60)
        print("  Movement:")
        print("    W / S           : Forward / Backward")
        print("    A / D           : Strafe Left / Right")
        print("    Q / E           : Rotate Left / Right")
        print("")
        print("  Control:")
        print("    SPACE           : Emergency Stop (zero all velocities)")
        print("    R               : Reset to zero velocity")
        print("    F5              : Reset simulation (restart episode)")
        print("    UP / DOWN Arrow : Increase / Decrease velocity scale")
        print("")
        print("  System:")
        print("    ESC             : Exit simulation")
        print("=" * 60)
        print(f"  Max Linear Vel  : {self.max_lin_vel:.2f} m/s")
        print(f"  Max Angular Vel : {self.max_ang_vel:.2f} rad/s")
        print(f"  Velocity Scale  : {self.vel_scale:.2f}x")
        print("=" * 60 + "\n")

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key = event.input

            # ESC to exit
            if key == carb.input.KeyboardInput.ESCAPE:
                self.should_exit = True
                print("\n[INFO] ESC pressed - Exiting simulation...")
                return True

            # Reset velocities to zero
            if key == carb.input.KeyboardInput.O:
                self.cmd_vel = [0.0, 0.0, 0.0]
                print("[INFO] Velocities reset to zero")
                return True

            # F5 to request env reset (mapped to R here in this script)
            if key == carb.input.KeyboardInput.R:
                self.request_env_reset = True
                print("[INFO] Requested environment reset")
                return True

            # Velocity scale changes with cooldown
            current_time = time.time()
            if current_time - self._last_scale_change_time > self._scale_change_cooldown:
                if key == carb.input.KeyboardInput.UP:
                    self.vel_scale = min(3.0, self.vel_scale + 0.1)
                    print(f"[INFO] Velocity scale: {self.vel_scale:.2f}x")
                    self._last_scale_change_time = current_time
                    return True
                if key == carb.input.KeyboardInput.DOWN:
                    self.vel_scale = max(0.1, self.vel_scale - 0.1)
                    print(f"[INFO] Velocity scale: {self.vel_scale:.2f}x")
                    self._last_scale_change_time = current_time
                    return True
        return True

    def update(self):
        """Update velocity commands based on current keyboard state. Call this every frame."""
        # Emergency stop
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.SPACE):
            self.cmd_vel = [0.0, 0.0, 0.0]
            return

        forward = 0.0
        lateral = 0.0
        yaw = 0.0

        # Forward/back
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.W):
            forward += self.vel_increment
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.S):
            forward -= self.vel_increment

        # Strafe
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.A):
            lateral += self.vel_increment
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.D):
            lateral -= self.vel_increment

        # Yaw
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.Q):
            yaw += self.vel_increment
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.E):
            yaw -= self.vel_increment

        # Apply scale
        forward *= self.vel_scale
        lateral *= self.vel_scale
        yaw *= self.vel_scale

        # Clamp
        self.cmd_vel[0] = max(min(forward, self.max_lin_vel), -self.max_lin_vel)
        self.cmd_vel[1] = max(min(lateral, self.max_lin_vel), -self.max_lin_vel)
        self.cmd_vel[2] = max(min(yaw, self.max_ang_vel), -self.max_ang_vel)

    def get_command_tensor(self, device, num_envs):
        cmd = torch.tensor(self.cmd_vel, device=device, dtype=torch.float32)
        return cmd.unsqueeze(0).repeat(num_envs, 1)

    def should_exit_simulation(self):
        return self.should_exit

    def cleanup(self):
        if self._sub_keyboard:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
            self._sub_keyboard = None
        print("[INFO] Keyboard controller stopped.")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent using keyboard control."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] No pre-trained checkpoint available for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Export (optional)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    kb_controller = CarbKeyboardController(
        max_lin_vel=args_cli.max_lin_vel,
        max_ang_vel=args_cli.max_ang_vel,
        vel_increment=0.15,
    )

    dt = env.unwrapped.step_dt

    obs = env.get_observations()
    timestep = 0

    # CSV logging
    csv_log_dir = os.path.join(log_dir, "teleop_logs")
    os.makedirs(csv_log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(csv_log_dir, f"teleop_log_{timestamp}.csv")

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    headers_written = False
    start_sim_time = time.time()

    # UDP socket to send joint angles to ROS bridge
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_target = ("127.0.0.1", 9000)  # localhost:9000

    print("[INFO] Starting simulation with keyboard control...")
    print("[INFO] Use WASD keys to move, QE to rotate, SPACE to stop, ESC to exit")
    print("[INFO] Press F5 (R key here) to reset the episode.\n")
    print("[INFO] UDP joint angles -> 127.0.0.1:9000")

    try:
        while simulation_app.is_running():
            if kb_controller.should_exit_simulation():
                print("[INFO] User requested exit via ESC key")
                break

            start_time = time.time()

            kb_controller.update()
            cmd = kb_controller.get_command_tensor(
                device=env.unwrapped.device,
                num_envs=env.unwrapped.num_envs,
            )
            env.unwrapped._commands[:] = cmd

            if kb_controller.request_env_reset:
                try:
                    obs = env.reset()
                except Exception:
                    if hasattr(env, "reset"):
                        obs = env.reset()
                    elif hasattr(env, "get_observations"):
                        obs = env.get_observations()
                    else:
                        obs = None
                kb_controller.request_env_reset = False
                timestep = 0
                env.unwrapped._commands[:] = cmd
                print("[INFO] Environment reset complete.")
                continue

            obs = env.get_observations()

            with torch.inference_mode():
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            # processed joint targets from env (after CPG)
            if hasattr(env.unwrapped, "_processed_actions"):
                joint_targets = env.unwrapped._processed_actions
            else:
                joint_targets = actions  # fallback

            joint_targets_list = joint_targets[0].cpu().tolist()

            # === Send via UDP as comma-separated string ===
            try:
                line = ",".join(f"{a:.6f}" for a in joint_targets_list)
                udp_sock.sendto(line.encode("utf-8"), udp_target)
            except Exception as e:
                print(f"[WARNING] Failed to send UDP joint angles: {e}")

            # === CSV log ===
            try:
                time_elapsed = time.time() - start_sim_time
                cmd_list = cmd[0].cpu().tolist()
                obs_list = obs[0].cpu().tolist()

                if not headers_written:
                    headers = ["timestep", "time_elapsed"]
                    headers.extend(["cmd_forward", "cmd_lateral", "cmd_yaw"])
                    headers.extend([f"obs_{i}" for i in range(len(obs_list))])
                    headers.extend([f"joint_{i}" for i in range(len(joint_targets_list))])
                    csv_writer.writerow(headers)
                    headers_written = True
                    print(f"[INFO] Logging observations and joints to: {csv_path}")

                row = [timestep, time_elapsed]
                row.extend(cmd_list)
                row.extend(obs_list)
                row.extend(joint_targets_list)
                csv_writer.writerow(row)
            except Exception as e:
                print(f"[WARNING] Failed to log to CSV: {e}")

            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    print(f"[INFO] Video recording complete ({args_cli.video_length} steps)")
                    break

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Simulation interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n[ERROR] Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[INFO] Cleaning up...")
        kb_controller.cleanup()

        try:
            csv_file.close()
            print(f"[INFO] Log saved to: {csv_path}")
        except Exception:
            pass

        try:
            udp_sock.close()
        except Exception:
            pass

        env.close()
        print("[INFO] Simulation ended successfully")


if __name__ == "__main__":
    main()
    simulation_app.close()
