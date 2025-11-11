import torch
import math

class HopfCPG:
    def __init__(self, num_envs: int, dt: float, device: str, freq_floor_hz: float = 1e-3):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.freq_floor = freq_floor_hz  # avoid ω=0 collapse
        # single oscillator per env
        self.x = torch.zeros(num_envs, 1, device=device)
        self.y = torch.zeros(num_envs, 1, device=device)
        self.alpha = 8.0
        self.reset(torch.arange(num_envs, device=device))

    def reset(self, env_ids: torch.Tensor):
        self.x[env_ids] = 0.99
        self.y[env_ids] = 0.0

    @torch.no_grad()
    def step(self, frequency: torch.Tensor, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        # frequency: (N,1); amplitude, phase: (N,12)
        # clamp freq to a small positive floor (or allow 0 as "stand" if you prefer)
        omega = 2.0 * math.pi * torch.clamp(frequency, min=self.freq_floor)  # (N,1)

        r2 = self.x * self.x + self.y * self.y
        mu = 1.0
        dx_dt = self.alpha * (mu - r2) * self.x - omega * self.y
        dy_dt = self.alpha * (mu - r2) * self.y + omega * self.x

        self.x += dx_dt * self.dt
        self.y += dy_dt * self.dt

        # readout — broadcast single internal phase to all joints
        cos_phase = torch.cos(phase)  # (N,12)
        sin_phase = torch.sin(phase)  # (N,12)
        # self.x/self.y are (N,1) and broadcast along joint dim
        x_shifted = self.x * cos_phase - self.y * sin_phase
        return amplitude * x_shifted


class SpiderCPG:
    def __init__(self, num_envs: int, dt: float, device: str):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device

        # single-oscillator internal CPG
        self.cpg = HopfCPG(num_envs, dt=dt, device=device)

        # cache the 12-joint trot template on device
        # (coxa, femur, tibia) × 4 legs, so repeat each leg's phase 3 times
        base = torch.tensor([0.0, math.pi, 0.0, math.pi], device=device)
        self.default_joint_phases = base.repeat_interleave(3).unsqueeze(0)  # (1,12)

    def reset(self, env_ids: torch.Tensor):
        self.cpg.reset(env_ids)

    def compute_joint_targets(self, frequency: torch.Tensor, amplitudes: torch.Tensor, leg_phase_offsets: torch.Tensor):
        joint_phases = leg_phase_offsets.repeat_interleave(3, dim=1) + self.default_joint_phases  # (N,12)
        return self.cpg.step(frequency, amplitudes, joint_phases)
