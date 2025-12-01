# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL with keyboard control."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import omni  # Import omni modules

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
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

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

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
        F5: Reset simulation (restart episode)  # <— added
    """

    def __init__(self, max_lin_vel=1.0, max_ang_vel=1.0, vel_increment=0.15):
        """
        Initialize keyboard controller using Carb input.

        Args:
            max_lin_vel: Maximum linear velocity (m/s)
            max_ang_vel: Maximum angular velocity (rad/s)
            vel_increment: Velocity increment per frame when key is held
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
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

        self.should_exit = False
        self.request_env_reset = False  # <— added
        self._last_scale_change_time = 0.0
        self._scale_change_cooldown = 0.2  # seconds

        self._print_controls()

    def _print_controls(self):
        """Print keyboard control instructions."""
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
        print("    F5              : Reset simulation (restart episode)")  # <— added
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
        """Handle keyboard events."""
        # Handle ESC key
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.ESCAPE:
                self.should_exit = True
                print("\n[INFO] ESC pressed - Exiting simulation...")
                return True

            # Handle reset-to-zero key
            if event.input == carb.input.KeyboardInput.O:
                self.cmd_vel = [0.0, 0.0, 0.0]
                print("[INFO] Velocities reset to zero")
                return True

            # Handle env reset key (F5)  # <— added
            if event.input == carb.input.KeyboardInput.R:
                self.request_env_reset = True
                print("[INFO] Requested environment reset")
                return True

            # Handle velocity scale changes with cooldown
            current_time = time.time()
            if current_time - self._last_scale_change_time > self._scale_change_cooldown:
                if event.input == carb.input.KeyboardInput.UP:
                    self.vel_scale = min(2.0, self.vel_scale + 0.1)
                    print(f"[INFO] Velocity scale: {self.vel_scale:.2f}x")
                    self._last_scale_change_time = current_time
                    return True
                elif event.input == carb.input.KeyboardInput.DOWN:
                    self.vel_scale = max(0.1, self.vel_scale - 0.1)
                    print(f"[INFO] Velocity scale: {self.vel_scale:.2f}x")
                    self._last_scale_change_time = current_time
                    return True

        return True

    def update(self):
        """Update velocity commands based on current keyboard state. Call this every frame."""
        # Check for emergency stop (SPACE)
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.SPACE):
            self.cmd_vel = [0.0, 0.0, 0.0]
            return

        # Reset velocities
        forward = 0.0
        lateral = 0.0
        yaw = 0.0

        # Forward/Backward (X-axis)
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.W):
            forward += self.vel_increment
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.S):
            forward -= self.vel_increment

        # Left/Right lateral (Y-axis)
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.A):
            lateral += self.vel_increment
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.D):
            lateral -= self.vel_increment

        # Rotation (Yaw)
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.Q):
            yaw += self.vel_increment
        if self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.E):
            yaw -= self.vel_increment

        # Apply velocity scale
        forward *= self.vel_scale
        lateral *= self.vel_scale
        yaw *= self.vel_scale

        # Clamp values to maximum velocities
        self.cmd_vel[0] = max(min(forward, self.max_lin_vel), -self.max_lin_vel)
        self.cmd_vel[1] = max(min(lateral, self.max_lin_vel), -self.max_lin_vel)
        self.cmd_vel[2] = max(min(yaw, self.max_ang_vel), -self.max_ang_vel)

    def get_command(self):
        """
        Get current velocity command.

        Returns:
            List of [forward_vel, lateral_vel, yaw_vel]
        """
        return self.cmd_vel.copy()

    def get_command_tensor(self, device, num_envs):
        """
        Get current velocity command as PyTorch tensor.

        Args:
            device: Torch device
            num_envs: Number of environments to replicate command for

        Returns:
            Torch tensor of shape (num_envs, 3)
        """
        cmd = torch.tensor(self.cmd_vel, device=device, dtype=torch.float32)
        return cmd.unsqueeze(0).repeat(num_envs, 1)

    def should_exit_simulation(self):
        """Check if user requested to exit."""
        return self.should_exit

    def cleanup(self):
        """Cleanup keyboard subscriptions."""
        if self._sub_keyboard:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
            self._sub_keyboard = None
        print("[INFO] Keyboard controller stopped.")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent using keyboard control."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    # Initialize keyboard controller using Carb
    kb_controller = CarbKeyboardController(
        max_lin_vel=args_cli.max_lin_vel,
        max_ang_vel=args_cli.max_ang_vel,
        vel_increment=0.15,
    )

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0

    # Setup CSV logging
    csv_log_dir = os.path.join(log_dir, "teleop_logs")
    os.makedirs(csv_log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(csv_log_dir, f"teleop_log_{timestamp}.csv")
    
    # Initialize CSV file (headers will be written after first step when we know dimensions)
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    headers_written = False
    start_sim_time = time.time()

    print("[INFO] Starting simulation with keyboard control...")
    print("[INFO] Use WASD keys to move, QE to rotate, SPACE to stop, ESC to exit")
    print("[INFO] Press F5 to reset the episode. Focus the viewport window to receive keyboard input!\n")

    # simulate environment
    try:
        while simulation_app.is_running():
            # Check if user wants to exit
            if kb_controller.should_exit_simulation():
                print("[INFO] User requested exit via ESC key")
                break

            start_time = time.time()

            # Update keyboard controller (reads current key states)
            kb_controller.update()

            # Get keyboard command and apply to all environments
            cmd = kb_controller.get_command_tensor(
                device=env.unwrapped.device,
                num_envs=env.unwrapped.num_envs,
            )
            env.unwrapped._commands[:] = cmd

            # --- Minimal addition: F5 triggers an episode reset ---
            if kb_controller.request_env_reset:
                try:
                    obs = env.reset()
                except Exception:
                    # Fallback for older wrappers
                    if hasattr(env, "reset"):
                        obs = env.reset()
                    elif hasattr(env, "get_observations"):
                        # If no reset, just refresh obs to avoid crashing
                        obs = env.get_observations()
                    else:
                        obs = None
                kb_controller.request_env_reset = False
                timestep = 0
                # re-apply the current command after reset
                env.unwrapped._commands[:] = cmd
                print("[INFO] Environment reset complete.")
                # skip stepping this frame for a clean restart
                continue
            # ------------------------------------------------------

            # Get observations
            obs = env.get_observations()

            # Run inference mode
            with torch.inference_mode():
                # Agent stepping
                actions = policy(obs)
                # Environment stepping
            obs, _, _, _ = env.step(actions)
            
            # Get the processed joint targets (12 DOF) after CPG processing
            if hasattr(env.unwrapped, '_processed_actions'):
                joint_targets = env.unwrapped._processed_actions
            else:
                joint_targets = actions  # Fallback to raw actions if not available
            
            # Log to CSV (for first environment only to keep file manageable)
            try:
                time_elapsed = time.time() - start_sim_time
                cmd_list = cmd[0].cpu().tolist()  # Get first env command
                obs_list = obs[0].cpu().tolist()  # Get first env observation
                joint_targets_list = joint_targets[0].cpu().tolist()  # Get first env joint targets (12 DOF)
                
                # Write headers on first log entry
                if not headers_written:
                    headers = ['timestep', 'time_elapsed']
                    headers.extend(['cmd_forward', 'cmd_lateral', 'cmd_yaw'])
                    headers.extend([f'obs_{i}' for i in range(len(obs_list))])
                    headers.extend([f'joint_{i}' for i in range(len(joint_targets_list))])
                    csv_writer.writerow(headers)
                    headers_written = True
                    print(f"[INFO] Logging observations and actions to: {csv_path}")
                    print(f"[INFO] Logging {len(obs_list)} observations and {len(joint_targets_list)} joint targets")
                
                # Prepare row data
                row = [timestep, time_elapsed]
                row.extend(cmd_list)
                row.extend(obs_list)
                row.extend(joint_targets_list)
                
                csv_writer.writerow(row)
                
            except Exception as e:
                print(f"[WARNING] Failed to log to CSV: {e}")

            # Handle video recording
            if args_cli.video:
                timestep += 1
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    print(f"[INFO] Video recording complete ({args_cli.video_length} steps)")
                    break

            # Time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Simulation interrupted by user (Ctrl+C)")

    except Exception as e:
        print(f"\n[ERROR] An error occurred during simulation: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("\n[INFO] Cleaning up...")
        kb_controller.cleanup()
        
        # Close CSV file
        try:
            csv_file.close()
            print(f"[INFO] Log saved to: {csv_path}")
        except:
            pass
        
        env.close()
        print("[INFO] Simulation ended successfully")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
