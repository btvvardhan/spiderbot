
# ~/spiderbot/scripts/debug_joints.py
from isaaclab.app import AppLauncher

# 1) Boot Kit (no .start() on this version)
app = AppLauncher(headless=True)
simulation_app = app.app

# 2) Now it's safe to import env + cfg
import gymnasium as gym
import spiderbot  # registers your Gym IDs

# Import your config class and instantiate it
from spiderbot.tasks.direct.spiderbot.spiderbot_env_cfg import SpiderbotEnvCfg

env = gym.make("Template-Spiderbot-Direct-v0", cfg=SpiderbotEnvCfg(), render_mode=None)
env.reset()

# 3) Grab the robot and list joints
robot = env.unwrapped.scene.articulations["robot"]

# Try several attributes depending on version
names = getattr(robot, "joint_names", None)
if names is None:
    names = getattr(robot.data, "joint_names", None)
if names is None:
    # Fallback: get indices with a regex, then pull names by index if available
    idxs, _ = robot.find_joints(".*")
    names = [f"joint_{i}" for i in idxs]

print("\nDOF count:", robot.data.joint_pos.shape[1])
print("Joint names:")
for i, n in enumerate(names):
    print(f"{i:2d}  {n}")

# 4) Clean shutdown
env.close()
simulation_app.close()
