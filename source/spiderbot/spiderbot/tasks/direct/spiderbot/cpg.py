# cpg.py  — drop-in patch

import torch
import math


class HopfCPG:
    def __init__(self, num_envs: int, num_oscillators: int, dt: float, device: str):
        self.num_envs = num_envs
        self.num_oscillators = num_oscillators
        self.dt = dt
        self.device = device

        # Internal oscillator states
        self.x = torch.zeros(num_envs, num_oscillators, device=device)
        self.y = torch.zeros(num_envs, num_oscillators, device=device)

        # Faster convergence to limit cycle
        self.alpha = 8.0

        # Start in a deterministic phase (no random offsets)
        self.reset(torch.arange(num_envs, device=device))

    def reset(self, env_ids: torch.Tensor):
        # Deterministic, phase-aligned reset (all oscillators same internal phase)
        self.x[env_ids] = 0.99
        self.y[env_ids] = 0.0

    def step(
        self,
        frequency: torch.Tensor,       # (N, 1)
        amplitude: torch.Tensor,       # (N, 12)
        phase: torch.Tensor            # (N, 12)  (includes your leg template offsets)
    ) -> torch.Tensor:
        # Natural frequency per env, broadcast to all oscillators
        omega = 2.0 * math.pi * frequency.expand(-1, self.num_oscillators)

        # Hopf dynamics
        r2 = self.x * self.x + self.y * self.y
        mu = 1.0
        dx_dt = self.alpha * (mu - r2) * self.x - omega * self.y
        dy_dt = self.alpha * (mu - r2) * self.y + omega * self.x

        # Integrate
        self.x = self.x + dx_dt * self.dt
        self.y = self.y + dy_dt * self.dt

        # *** PHASE LOCK: force a shared internal phase across all 12 joints ***
        x0 = self.x[:, :1]  # take oscillator 0 as the master
        y0 = self.y[:, :1]
        self.x = x0.expand(-1, self.num_oscillators)
        self.y = y0.expand(-1, self.num_oscillators)

        # Readout with your per-leg phase offsets (trot template + learned leg offsets)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        x_shifted = self.x * cos_phase - self.y * sin_phase

        # Per-joint amplitudes
        return amplitude * x_shifted


class SpiderCPG:
    def __init__(self, num_envs: int, dt: float, device: str):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device

        # Keep 12 outputs (3 joints × 4 legs); internal phase is shared by the patch above
        self.cpg = HopfCPG(num_envs, num_oscillators=12, dt=dt, device=device)

        # Diagonal trot template (used at readout only)
        self.default_leg_phases = torch.tensor([0.0, math.pi, 0.0, math.pi], device=device)

    def reset(self, env_ids: torch.Tensor):
        # Deterministic, aligned CPG start each episode
        self.cpg.reset(env_ids)

    def compute_joint_targets(
        self,
        frequency: torch.Tensor,          # (N,1)
        amplitudes: torch.Tensor,         # (N,12)
        leg_phase_offsets: torch.Tensor   # (N,4)
    ) -> torch.Tensor:
        # Expand leg offsets to joints and add fixed trot template
        joint_phases = leg_phase_offsets.repeat_interleave(3, dim=1)
        default_phases = self.default_leg_phases.repeat_interleave(3).unsqueeze(0)
        joint_phases = joint_phases + default_phases  # (N,12)

        return self.cpg.step(frequency, amplitudes, joint_phases)
