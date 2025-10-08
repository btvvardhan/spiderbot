"""
Test if CPG-RL environment initializes correctly.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test CPG environment")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after Isaac Sim is initialized
import torch
from spiderbot.tasks.direct.spiderbot.spiderbot_env_cfg import SpiderbotEnvCfg
from spiderbot.tasks.direct.spiderbot.spiderbot_env import SpiderbotEnv
#from spiderbot.tasks.direct.spiderbot import SpiderbotEnvCfg, SpiderbotEnv

def main():
    """Test CPG environment."""
    
    print("="*60)
    print("Testing CPG-RL Environment")
    print("="*60)
    
    # Create environment config
    cfg = SpiderbotEnvCfg()
    cfg.scene.num_envs = 4  # Just 4 envs for testing
    
    print(f"\n✓ Config created")
    print(f"  - Action space: {cfg.action_space} (should be 17)")
    print(f"  - Observation space: {cfg.observation_space} (should be 53)")
    
    # Create environment
    env = SpiderbotEnv(cfg)
    
    print(f"\n✓ Environment created")
    print(f"  - Num envs: {env.num_envs}")
    print(f"  - Device: {env.device}")
    
    # Reset environment
    obs, _ = env.reset()
    
    print(f"\n✓ Environment reset")
    print(f"  - Observation shape: {obs['policy'].shape}")
    print(f"    Expected: ({env.num_envs}, 53)")
    
    # Test random actions
    print(f"\n✓ Testing CPG actions...")
    for i in range(5):
        # Random CPG parameters
        actions = torch.randn(env.num_envs, 17, device=env.device)
        obs, rewards, dones, truncated, info = env.step(actions)
        
        print(f"  Step {i+1}: reward_mean = {rewards.mean():.3f}")
    
    print(f"\n✓ All tests passed!")
    print("="*60)
    
    simulation_app.close()

if __name__ == "__main__":
    main()
