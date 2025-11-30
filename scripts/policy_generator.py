"""
Policy Generator for VDP-CPG Spider Robot

Extracts trained actor network from checkpoint and packages it with:
- Actor state dict (weights & biases)
- Observation normalization statistics (mean/var)
- Network architecture metadata
- CPG parameters
- Action/observation space info

Usage:
    python policy_generator.py \
        --ckpt logs/rsl_rl/spiderbot/2025-11-30/model_5000.pt \
        --out exported_policies/spiderbot_policy.pt
"""

import argparse
import torch
import os
from pathlib import Path


def to_cpu_tensor(x):
    """Safely convert to CPU tensor."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    try:
        return torch.as_tensor(x).detach().cpu()
    except Exception:
        return None


def pull_state(ckpt):
    """Extract model state dict from checkpoint."""
    # Common keys used by various trainers
    state_keys = [
        "model_state_dict",  # Standard PyTorch
        "state_dict",        # Common alternative
        "model",             # Direct model save
        "ac_state_dict",     # Actor-critic
        "actor_critic",      # Alternative AC
    ]
    
    for key in state_keys:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    
    # Sometimes the checkpoint itself is a state_dict
    if isinstance(ckpt, dict):
        # Check if it looks like a state dict (all values are tensors or dicts)
        if all(isinstance(v, (torch.Tensor, dict)) for v in ckpt.values()):
            return ckpt
    
    # Debug: print available keys
    print("\n❌ Could not find model state dict!")
    print("Available checkpoint keys:")
    for k in ckpt.keys():
        print(f"  - {k}: {type(ckpt[k])}")
    
    raise RuntimeError("Could not find a model state dict in checkpoint")


def slice_actor(sd, verbose=True):
    """Extract actor parameters from full state dict."""
    # Common actor prefixes in different frameworks
    prefixes = [
        "actor.",                    # Standard
        "module.actor.",             # DataParallel
        "policy.",                   # Alternative naming
        "pi.",                       # Short for policy
        "module.policy.",            # DataParallel policy
        "ac.actor.",                 # Actor-critic
        "module.ac.actor.",          # DataParallel AC
        "actor_critic.actor.",       # Full AC naming
        "_actor.",                   # Private naming
        "network.actor.",            # Nested network
    ]
    
    for prefix in prefixes:
        actor_sub = {
            k[len(prefix):]: v 
            for k, v in sd.items() 
            if k.startswith(prefix)
        }
        if actor_sub:
            if verbose:
                print(f"✅ Found actor params with prefix: '{prefix}'")
                print(f"   Actor has {len(actor_sub)} parameters")
            return actor_sub
    
    # Fallback: check if keys already look like actor params (no prefix)
    # Look for common layer names
    actor_indicators = ["mlp.", "l1.", "fc.", "layer.", "linear."]
    has_actor_layers = any(
        any(k.startswith(ind) for ind in actor_indicators)
        for k in sd.keys()
    )
    
    if has_actor_layers:
        if verbose:
            print("✅ Assuming entire state dict is actor (no prefix found)")
        return sd
    
    # Debug: print available keys
    print("\n❌ Could not identify actor parameters!")
    print("Available state dict keys (first 20):")
    for i, k in enumerate(list(sd.keys())[:20]):
        print(f"  - {k}")
    if len(sd) > 20:
        print(f"  ... and {len(sd) - 20} more")
    
    raise RuntimeError(
        "Actor params not found. Check if checkpoint contains actor weights. "
        "You may need to add a custom prefix to the slice_actor() function."
    )


def pull_obs_rms(ckpt):
    """Extract observation normalization statistics."""
    # Method 1: Nested dict with 'mean' and 'var'
    rms = ckpt.get("obs_rms", None)
    if isinstance(rms, dict):
        mean = to_cpu_tensor(rms.get("mean"))
        var = to_cpu_tensor(rms.get("var"))
        if mean is not None or var is not None:
            return {"mean": mean, "var": var}
    
    # Method 2: Direct tensors
    mean = ckpt.get("obs_mean", None)
    var = ckpt.get("obs_var", None)
    if mean is not None or var is not None:
        return {
            "mean": to_cpu_tensor(mean),
            "var": to_cpu_tensor(var)
        }
    
    # Method 3: Nested in 'normalizer' or 'obs_normalizer'
    normalizer = ckpt.get("obs_normalizer", ckpt.get("normalizer", None))
    if isinstance(normalizer, dict):
        mean = to_cpu_tensor(normalizer.get("mean"))
        var = to_cpu_tensor(normalizer.get("var"))
        if mean is not None or var is not None:
            return {"mean": mean, "var": var}
    
    return None


def infer_network_architecture(actor_sd):
    """Infer network architecture from state dict."""
    architecture = {
        "hidden_layers": [],
        "activation": "elu",  # Default for Isaac Lab
        "input_dim": None,
        "output_dim": None,
    }
    
    # Find layer dimensions
    layer_keys = sorted([k for k in actor_sd.keys() if 'weight' in k])
    
    for key in layer_keys:
        weight = actor_sd[key]
        if len(weight.shape) == 2:  # Fully connected layer
            out_dim, in_dim = weight.shape
            
            if architecture["input_dim"] is None:
                architecture["input_dim"] = in_dim
            
            # Check if this is a hidden layer or output layer
            # Output layer is typically the last one
            if key == layer_keys[-1]:
                architecture["output_dim"] = out_dim
            else:
                architecture["hidden_layers"].append(out_dim)
    
    return architecture


def main():
    parser = argparse.ArgumentParser(
        description="Extract and export trained policy for deployment"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g., logs/.../model_5000.pt)"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path for exported policy (e.g., policy.pt)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    
    # Create output directory if needed
    out_dir = os.path.dirname(args.out)
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🤖 VDP-CPG Spider Robot Policy Generator")
    print(f"{'='*60}\n")
    
    # Load checkpoint
    print(f"📂 Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    print(f"   ✅ Checkpoint loaded ({len(ckpt)} top-level keys)")
    
    # Extract state dict
    print(f"\n🔍 Extracting model state...")
    state_dict = pull_state(ckpt)
    print(f"   ✅ Found state dict with {len(state_dict)} parameters")
    
    # Extract actor
    print(f"\n🎭 Extracting actor network...")
    actor_sd = slice_actor(state_dict, verbose=args.verbose)
    
    # Infer architecture
    print(f"\n🏗️  Inferring network architecture...")
    architecture = infer_network_architecture(actor_sd)
    print(f"   Input dim: {architecture['input_dim']}")
    print(f"   Hidden layers: {architecture['hidden_layers']}")
    print(f"   Output dim: {architecture['output_dim']}")
    print(f"   Activation: {architecture['activation']}")
    
    # Extract normalization stats
    print(f"\n📊 Extracting observation normalization...")
    obs_rms = pull_obs_rms(ckpt)
    if obs_rms and obs_rms.get("mean") is not None:
        print(f"   ✅ Found obs normalization (mean shape: {obs_rms['mean'].shape})")
    else:
        print(f"   ⚠️  No obs normalization found (will use raw observations)")
    
    # VDP-CPG specific parameters
    cpg_params = {
        "mu": 2.5,           # VDP nonlinearity
        "k_phase": 0.5,      # Phase coupling strength
        "k_amp": 0.3,        # Amplitude coupling
        "dt": 1.0 / 200.0,   # Physics timestep
    }
    
    # Action space info for VDP-CPG
    action_info = {
        "dim": 17,
        "components": {
            "frequency": {"index": 0, "range": [0.3, 2.5], "unit": "Hz"},
            "amplitudes": {"index": "1:13", "range": [0.0, 0.7], "unit": "rad"},
            "phase_offsets": {"index": "13:17", "range": [-1.0, 1.0], "unit": "rad"},
        },
        "description": "CPG parameters: [freq(1), amps(12), phases(4)]"
    }
    
    # Observation space info
    obs_info = {
        "dim": 52,
        "components": {
            "commands": 3,
            "command_history": 15,
            "imu_lin_vel": 3,
            "imu_ang_vel": 3,
            "projected_gravity": 3,
            "cpg_phases": 8,
            "prev_actions": 17,
        },
        "description": "IMU-aware actor observations"
    }
    
    # Create payload
    payload = {
        # Model weights
        "actor_state_dict": {k: v.cpu() for k, v in actor_sd.items()},
        
        # Normalization
        "obs_mean": (obs_rms.get("mean") if obs_rms else None),
        "obs_var": (obs_rms.get("var") if obs_rms else None),
        
        # Architecture
        "architecture": architecture,
        
        # VDP-CPG parameters
        "cpg_params": cpg_params,
        
        # Space definitions
        "action_space": action_info,
        "observation_space": obs_info,
        
        # Metadata
        "meta": {
            "robot": "Spider Robot (12-DOF)",
            "control": "VDP-CPG with RL modulation",
            "gait": "Trot (diagonal coordination)",
            "action_type": "continuous_unbounded",  # NOT tanh!
            "obs_normalization": obs_rms is not None,
            "deployment": {
                "hardware": "Jetson Nano + Arduino Mega + PCA9685",
                "control_freq": 50,  # Hz
                "imu_required": True,
                "imu_type": "MPU9250 (9-axis)",
            },
            "note": (
                "Actions are CPG parameters, NOT joint positions. "
                "Reconstruct VDP-CPG on deployment to generate joint targets."
            ),
        },
    }
    
    # Save
    print(f"\n💾 Saving exported policy...")
    torch.save(payload, args.out)
    
    # Verify file was created
    file_size = os.path.getsize(args.out) / 1024  # KB
    print(f"   ✅ Saved to: {args.out}")
    print(f"   📦 File size: {file_size:.1f} KB")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Export Complete!")
    print(f"{'='*60}")
    print(f"\n📋 Policy Summary:")
    print(f"   • Actor parameters: {len(actor_sd)}")
    print(f"   • Input dim: {architecture['input_dim']} (observations)")
    print(f"   • Output dim: {architecture['output_dim']} (CPG params)")
    print(f"   • Obs normalization: {'Yes' if obs_rms else 'No'}")
    print(f"   • CPG type: Van der Pol (VDP)")
    print(f"   • Gait: Trot (diagonal coordination)")
    
    print(f"\n🚀 Deployment Instructions:")
    print(f"   1. Load policy: policy = torch.load('{args.out}')")
    print(f"   2. Reconstruct actor MLP with architecture info")
    print(f"   3. Load weights: actor.load_state_dict(policy['actor_state_dict'])")
    print(f"   4. Create VDP-CPG with policy['cpg_params']")
    print(f"   5. Run inference: actions = actor(observations)")
    print(f"   6. Generate joint targets: cpg.compute_targets(actions)")
    print(f"\n💡 See deployment script for complete implementation\n")


if __name__ == "__main__":
    main()