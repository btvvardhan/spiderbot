
##################################
import argparse, torch

def to_cpu_tensor(x):
    if x is None: return None
    if isinstance(x, torch.Tensor): return x.detach().cpu()
    try: return torch.as_tensor(x).detach().cpu()
    except Exception: return None

def pull_state(ckpt):
    # common keys used by various trainers
    for k in ("model_state_dict", "state_dict", "model", "ac_state_dict"):
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    # sometimes the checkpoint itself is a state_dict
    if isinstance(ckpt, dict) and all(isinstance(v, (torch.Tensor, dict)) for v in ckpt.values()):
        return ckpt
    raise RuntimeError("Could not find a model state dict in checkpoint")

def slice_actor(sd):
    prefixes = [
        "actor.", "module.actor.", "policy.", "pi.", "module.policy.",
        "ac.actor.", "module.ac.actor.", "actor_critic.actor."
    ]
    for p in prefixes:
        actor_sub = { k[len(p):]: v for k,v in sd.items() if k.startswith(p) }
        if actor_sub: return actor_sub
    # fallback: if keys look like "mlp.0.weight" already (no prefix)
    has_mlp = any(k.startswith("mlp.") or k.startswith("l1.") for k in sd.keys())
    if has_mlp:  # assume it's already the actor
        return sd
    raise RuntimeError("Actor params not found. Inspect checkpoint keys.")

def pull_obs_rms(ckpt):
    # rsl_rl usually stores a dict with 'mean' and 'var'
    rms = ckpt.get("obs_rms", None)
    if isinstance(rms, dict):
        return {"mean": to_cpu_tensor(rms.get("mean")), "var": to_cpu_tensor(rms.get("var"))}
    # sometimes stored as simple tensors
    m, v = ckpt.get("obs_mean", None), ckpt.get("obs_var", None)
    if m is not None or v is not None:
        return {"mean": to_cpu_tensor(m), "var": to_cpu_tensor(v)}
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=False, help="/home/teja/spiderbot/logs/rsl_rl/cartpole_direct/2025-11-22_22_11_25/model_3000.pt")
    ap.add_argument("--out",  required=False, help="/home/teja/spiderbot/logs/rsl_rl/cartpole_direct/2025-11-22_22_11_25/export/policy.pt")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd   = pull_state(ckpt)
    actor_sd = slice_actor(sd)
    obs_rms  = pull_obs_rms(ckpt)

    payload = {
        "actor_state_dict": {k: v.cpu() for k, v in actor_sd.items()},
        "obs_mean": (obs_rms.get("mean") if obs_rms else None),
        "obs_var":  (obs_rms.get("var")  if obs_rms else None),
        "meta": {
            "expects_tanh": True,
            "note": "Rebuild your actor MLP at inference to match training (hidden sizes & activation), then load this state_dict.",
        },
    }
    torch.save(payload, args.out)
    print(f"Saved actor-only policy: {args.out}")

if __name__ == "__main__":
    main()

