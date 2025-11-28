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
        self.alpha = 50.0  # Increased from 8.0 for faster convergence
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
        """Return current oscillator angle φ in [-π,π] per env (N,1)."""
        return torch.atan2(self.y, self.x)


class SpiderCPG:
    """
    CPG with diagonal coupling for trot gait.
    k_phase: phase coupling strength (0=independent, 1=identical)
    k_amp: amplitude coupling strength (0=independent, 1=identical)
    """
    def __init__(self, num_envs: int, dt: float, device: str, k_phase: float = 0.8, k_amp: float = 0.9):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.k_phase = float(k_phase)  # Increased from 0.7
        self.k_amp = float(k_amp)      # Increased from 1.0
        self.cpg = HopfCPG(num_envs, dt=dt, device=device)

        # Trot phases: FL and RR in phase, FR and RL in phase (opposite to first pair)
        leg_phases = torch.tensor([0.0, math.pi, math.pi, 0.0], device=device)

        # Intra-leg offsets for femur/tibia lift timing
        intra = torch.tensor([
            0.0,  +0.4*math.pi,  +0.6*math.pi,  # FL: coxa, femur, tibia
            0.0,  +0.4*math.pi,  +0.6*math.pi,  # FR
            0.0,  +0.4*math.pi,  +0.6*math.pi,  # RL
            0.0,  +0.4*math.pi,  +0.6*math.pi,  # RR
        ], device=device).unsqueeze(0)

        self.default_joint_phases = leg_phases.repeat_interleave(3).unsqueeze(0) + intra

        # Joint-type scaling: coxa does most work, femur/tibia assist
        self.joint_type_scales = torch.tensor([
            1.0, 0.4, 0.3,  # FL: higher femur contribution
            1.0, 0.4, 0.3,  # FR
            1.0, 0.4, 0.3,  # RL
            1.0, 0.4, 0.3,  # RR
        ], device=device).unsqueeze(0)

    def reset(self, env_ids: torch.Tensor):
        self.cpg.reset(env_ids)

    @torch.no_grad()
    def _couple_diagonal_phases(self, leg_phase_offsets: torch.Tensor) -> torch.Tensor:
        """Soft couple diagonal leg phases (FL↔RR, FR↔RL)."""
        phi = leg_phase_offsets  # (N,4) [FL, FR, RL, RR]
        d0 = 0.5 * (phi[:, 0:1] + phi[:, 3:4])  # FL & RR average
        d1 = 0.5 * (phi[:, 1:2] + phi[:, 2:3])  # FR & RL average
        coupled = torch.cat([d0, d1, d1, d0], dim=1)
        return (1.0 - self.k_phase) * phi + self.k_phase * coupled

    @torch.no_grad()
    def _couple_diagonal_amplitudes(self, amplitudes: torch.Tensor) -> torch.Tensor:
        """Soft couple diagonal leg amplitudes."""
        N = amplitudes.shape[0]
        amps = amplitudes.view(N, 4, 3)  # (N, 4legs, 3joints)
        
        # Average diagonal pairs
        avg0 = 0.5 * (amps[:, 0, :] + amps[:, 3, :])  # FL & RR
        avg1 = 0.5 * (amps[:, 1, :] + amps[:, 2, :])  # FR & RL
        
        k = self.k_amp
        amps[:, 0, :] = (1 - k) * amps[:, 0, :] + k * avg0
        amps[:, 3, :] = (1 - k) * amps[:, 3, :] + k * avg0
        amps[:, 1, :] = (1 - k) * amps[:, 1, :] + k * avg1
        amps[:, 2, :] = (1 - k) * amps[:, 2, :] + k * avg1
        
        return amps.reshape(N, 12)

    def compute_joint_targets(
        self,
        frequency: torch.Tensor,
        amplitudes: torch.Tensor,
        leg_phase_offsets: torch.Tensor
    ) -> torch.Tensor:
        # Couple phases and amplitudes
        leg_phase_offsets = self._couple_diagonal_phases(leg_phase_offsets)
        amplitudes = self._couple_diagonal_amplitudes(amplitudes)
        
        # Expand to joint phases
        joint_phases = leg_phase_offsets.repeat_interleave(3, dim=1) + self.default_joint_phases
        
        # Apply joint-type scaling
        scaled_amplitudes = amplitudes * self.joint_type_scales
        
        # Generate oscillations
        return self.cpg.step(frequency, scaled_amplitudes, joint_phases)