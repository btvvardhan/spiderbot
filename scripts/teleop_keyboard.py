# teleop_keyboard.py
import argparse, os, yaml, torch, gymnasium as gym
from isaaclab.app import AppLauncher

# --- CLI ---
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Template-Spiderbot-Direct-v0")
parser.add_argument("--load_run", type=str, required=True)         # e.g. 2025-09-18_22-25-46
parser.add_argument("--load_checkpoint", type=str, required=True)  # e.g. "model_199.pt" or "199"
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)  # adds --device, --headless, etc.
args, hydra_args = parser.parse_known_args()

# --- Launch app first (so omni.* exists) ---
app = AppLauncher(args).app

# --- Register tasks ---
import isaaclab_tasks      # registers stock tasks
import spiderbot.tasks     # registers your task id

# --- Build env cfg from registry ---
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, get_checkpoint_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import omni

env_cfg_or_cls = load_cfg_from_registry(args.task, "env_cfg_entry_point")
env_cfg = env_cfg_or_cls() if isinstance(env_cfg_or_cls, type) else env_cfg_or_cls
env_cfg.scene.num_envs = args.num_envs
if args.device:
    env_cfg.sim.device = args.device

# --- Make env + wrapper ---
env = gym.make(args.task, cfg=env_cfg, render_mode=None)
venv = RslRlVecEnvWrapper(env, clip_actions=1.0)   # numeric, not bool

# --- Load same agent config you trained with ---
run_dir = os.path.abspath(os.path.join("logs", "rsl_rl", "cartpole_direct", str(args.load_run)))
with open(os.path.join(run_dir, "params", "agent.yaml"), "r") as f:
    agent_cfg_dict = yaml.safe_load(f)
agent_cfg_dict["device"] = env_cfg.sim.device

# --- Build runner + load checkpoint ---
runner = OnPolicyRunner(venv, agent_cfg_dict, log_dir=".", device=env_cfg.sim.device)
resume_path = get_checkpoint_path(os.path.dirname(run_dir), str(args.load_run), str(args.load_checkpoint))
print(f"[INFO] Loading checkpoint: {resume_path}")
runner.load(resume_path)

# --- Keyboard handling (carb.input) ---
import carb
import carb.input as ci
from carb.input import Keyboard, KeyboardInput, KeyboardEventType  # <-- IMPORTANT: import Keyboard enum too

kit_app = omni.kit.app.get_app()
inp = ci.acquire_input_interface()

class KeyboardEventHandler:
    def __init__(self):
        # keep lightweight Python state (thread-safe)
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.v_gain = 0.3
        self.w_gain = 0.7
        self._num_envs = env.unwrapped.num_envs

    def on_keyboard_event(self, event: ci.KeyboardEvent):
        key, et = event.input, event.type
        if et == KeyboardEventType.KEY_PRESS:
            if key == KeyboardInput.W: self.vx = 1.0
            elif key == KeyboardInput.S: self.vx = -1.0
            elif key == KeyboardInput.A: self.vy = 1.0
            elif key == KeyboardInput.D: self.vy = -1.0
            elif key == KeyboardInput.Q: self.wz = 1.0
            elif key == KeyboardInput.E: self.wz = -1.0
            elif key == KeyboardInput.SPACE: self.vx = self.vy = self.wz = 0.0
        elif et == KeyboardEventType.KEY_RELEASE:
            if key in (KeyboardInput.W, KeyboardInput.S): self.vx = 0.0
            if key in (KeyboardInput.A, KeyboardInput.D): self.vy = 0.0
            if key in (KeyboardInput.Q, KeyboardInput.E): self.wz = 0.0
        return True

    def get(self):
        # return a CPU tensor (env will move it to the sim device in _pre_physics_step)
        vals = torch.tensor([self.vx * self.v_gain, self.vy * self.v_gain, self.wz * self.w_gain], dtype=torch.float32)
        return vals.view(1, 3).repeat(self._num_envs, 1)

handler = KeyboardEventHandler()

# Subscribe to the single logical keyboard device in Kit
# Some carb builds don't expose a constructible Keyboard device or Keyboard.KEYBOARD.
# Instead subscribe to all input events and forward KeyboardEvent instances to our handler.
sub_id = None

# add a generic input event forwarder on the handler
def _on_input_event_forwarder(*args):
    # subscribe_to_input_events may call with (event,) or (device, event)
    if len(args) == 1:
        evt = args[0]
    elif len(args) >= 2:
        evt = args[1]
    else:
        return True

    try:
        # debug: uncomment to verify events seen
        # print("[DEBUG] input forwarder got:", type(evt), repr(evt))
        if isinstance(evt, ci.KeyboardEvent):
            return handler.on_keyboard_event(evt)
    except Exception as e:
        print("[DEBUG] forwarder exception:", e)
    return True

try:
    # try to subscribe so we get events before UI (if API supports order)
    try:
        sub_id = inp.subscribe_to_input_events(_on_input_event_forwarder, ci.SUBSCRIPTION_ORDER_FIRST)
    except TypeError:
        # some carb versions only accept the callback
        sub_id = inp.subscribe_to_input_events(_on_input_event_forwarder)
    print("[INFO] Subscribed to input events (sub_id=%s), forwarding KeyboardEvent to handler." % sub_id)
except Exception as e:
    print("[ERROR] subscribe_to_input_events failed:", e)
    print("[DEBUG] dir(inp):", dir(inp))
    print("[DEBUG] dir(ci):", dir(ci))
    raise

print("W/S: fwd/back, A/D: left/right, Q/E: yaw, SPACE: stop   (Close the window or Ctrl+C to quit.)")

# --- Main loop ---
try:
    while kit_app.is_running():
        env.unwrapped.extras["teleop_cmd"] = handler.get()
        with torch.no_grad():
            # venv.get_observations() may return obs or (obs, extra); handle both
            res = venv.get_observations()
            obs = res[0] if isinstance(res, (tuple, list)) and len(res) >= 1 else res
            actions = runner.alg.act(obs)
        venv.step(actions)
finally:
    # unsubscribe and cleanup to avoid resource leaks
    try:
        if sub_id is not None:
            inp.unsubscribe_to_input_events(sub_id)
    except Exception:
        pass
    env.close()
    app.close()
