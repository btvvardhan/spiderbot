import torch
import math

class HopfCPG:
    def __init__(self, num_envs: int, dt: float, device: str, freq_floor_hz: float = 1e-3):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.freq_floor = freq_floor_hz
        self.x = torch.zeros(num_envs, 1, device=device)
        self.y = torch.zeros(num_envs, 1, device=device)
        self.alpha = 50.0
        self.reset(torch.arange(num_envs, device=device))

    def reset(self, env_ids: torch.Tensor):
        self.x[env_ids] = 0.99
        self.y[env_ids] = 0.0

    @torch.no_grad()
    def step(self, frequency: torch.Tensor, amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        omega = 2.0 * math.pi * torch.clamp(frequency, min=self.freq_floor)
        r2 = self.x * self.x + self.y * self.y
        mu = 1.0
        dx_dt = self.alpha * (mu - r2) * self.x - omega * self.y
        dy_dt = self.alpha * (mu - r2) * self.y + omega * self.x
        self.x += dx_dt * self.dt
        self.y += dy_dt * self.dt
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        x_shifted = self.x * cos_phase - self.y * sin_phase
        return amplitude * x_shifted

    @torch.no_grad()
    def phase_angle(self) -> torch.Tensor:
        return torch.atan2(self.y, self.x)


class SpiderCPG_Walk:
    """
    CPG for WALK gait (3 legs always on ground, sequential lifting).
    
    Walk sequence: FL → RR → RL → FR (diagonal pairs alternating)
    Phase offsets: FL=0°, RR=90°, RL=180°, FR=270°
    
    This is more stable than trot at low speeds.
    """
    def __init__(self, num_envs: int, dt: float, device: str, k_phase: float = 0.6, k_amp: float = 0.7):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.k_phase = float(k_phase)  # Lower coupling for walk (more independence)
        self.k_amp = float(k_amp)
        self.cpg = HopfCPG(num_envs, dt=dt, device=device)

        # WALK GAIT PHASES: Sequential leg lifting
        # FL → RR → RL → FR (90° apart)
        leg_phases = torch.tensor([
            0.0,           # FL starts
            0.5 * math.pi, # RR follows (90° later)
            math.pi,       # RL follows (180° later)
            1.5 * math.pi, # FR follows (270° later)
        ], device=device)

        # Intra-leg offsets: femur/tibia lift during swing
        intra = torch.tensor([
            0.0,  +0.3*math.pi,  +0.5*math.pi,  # FL
            0.0,  +0.3*math.pi,  +0.5*math.pi,  # RR
            0.0,  +0.3*math.pi,  +0.5*math.pi,  # RL
            0.0,  +0.3*math.pi,  +0.5*math.pi,  # FR
        ], device=device).unsqueeze(0)

        self.default_joint_phases = leg_phases.repeat_interleave(3).unsqueeze(0) + intra

        # Joint-type scaling for walk (more conservative)
        self.joint_type_scales = torch.tensor([
            1.0, 0.35, 0.25,  # FL
            1.0, 0.35, 0.25,  # RR
            1.0, 0.35, 0.25,  # RL
            1.0, 0.35, 0.25,  # FR
        ], device=device).unsqueeze(0)

    def reset(self, env_ids: torch.Tensor):
        self.cpg.reset(env_ids)

    @torch.no_grad()
    def _couple_diagonal_phases(self, leg_phase_offsets: torch.Tensor) -> torch.Tensor:
        """Light coupling for walk gait (legs more independent)."""
        phi = leg_phase_offsets  # (N,4) [FL, RR, RL, FR]
        # For walk, we still want some diagonal coordination but less strict
        d0 = 0.5 * (phi[:, 0:1] + phi[:, 1:2])  # FL & RR
        d1 = 0.5 * (phi[:, 2:3] + phi[:, 3:4])  # RL & FR
        coupled = torch.cat([d0, d0, d1, d1], dim=1)
        return (1.0 - self.k_phase) * phi + self.k_phase * coupled

    @torch.no_grad()
    def _couple_diagonal_amplitudes(self, amplitudes: torch.Tensor) -> torch.Tensor:
        """Light amplitude coupling for walk gait."""
        N = amplitudes.shape[0]
        amps = amplitudes.view(N, 4, 3)
        
        # Couple FL-RR and RL-FR pairs
        avg0 = 0.5 * (amps[:, 0, :] + amps[:, 1, :])
        avg1 = 0.5 * (amps[:, 2, :] + amps[:, 3, :])
        
        k = self.k_amp
        amps[:, 0, :] = (1 - k) * amps[:, 0, :] + k * avg0
        amps[:, 1, :] = (1 - k) * amps[:, 1, :] + k * avg0
        amps[:, 2, :] = (1 - k) * amps[:, 2, :] + k * avg1
        amps[:, 3, :] = (1 - k) * amps[:, 3, :] + k * avg1
        
        return amps.reshape(N, 12)

    def compute_joint_targets(
        self,
        frequency: torch.Tensor,
        amplitudes: torch.Tensor,
        leg_phase_offsets: torch.Tensor
    ) -> torch.Tensor:
        leg_phase_offsets = self._couple_diagonal_phases(leg_phase_offsets)
        amplitudes = self._couple_diagonal_amplitudes(amplitudes)
        
        joint_phases = leg_phase_offsets.repeat_interleave(3, dim=1) + self.default_joint_phases
        scaled_amplitudes = amplitudes * self.joint_type_scales
        
        return self.cpg.step(frequency, scaled_amplitudes, joint_phases)