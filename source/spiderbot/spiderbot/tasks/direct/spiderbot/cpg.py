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
        self.alpha = 8.0
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
    CPG that enforces diagonal similarity:
      - Phase coupling strength k_phase (0..1)
      - Amplitude coupling strength k_amp   (0..1)
    Set k_phase=k_amp=1.0 to make diagonals move exactly the same way.
    """
    def __init__(self, num_envs: int, dt: float, device: str, k_phase: float = 0.7, k_amp: float = 1.0):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.k_phase = float(k_phase)
        self.k_amp = float(k_amp)
        self.cpg = HopfCPG(num_envs, dt=dt, device=device)

        # Trot phases for legs: [FL, FR, RL, RR]
        leg_phases = torch.tensor([0.0, math.pi, math.pi, 0.0], device=device)

        # Intra-leg offsets so femur/tibia lift near mid-swing
        intra = torch.tensor([
            0.0,  +0.5*math.pi,  +0.5*math.pi,
            0.0,  +0.5*math.pi,  +0.5*math.pi,
            0.0,  +0.5*math.pi,  +0.5*math.pi,
            0.0,  +0.5*math.pi,  +0.5*math.pi,
        ], device=device).unsqueeze(0)

        self.default_joint_phases = leg_phases.repeat_interleave(3).unsqueeze(0) + intra  # (1,12)

        # Joint-type scaling [coxa, femur, tibia] × 4 legs
        self.joint_type_scales = torch.tensor([
            0.8, 0.3, 0.15,
            0.8, 0.3, 0.15,
            0.8, 0.3, 0.15,
            0.8, 0.3, 0.15,
        ], device=device).unsqueeze(0)  # (1,12)

    def reset(self, env_ids: torch.Tensor):
        self.cpg.reset(env_ids)

    @torch.no_grad()
    def _couple_diagonal_phases(self, leg_phase_offsets: torch.Tensor) -> torch.Tensor:
        """Soft/hard tie per-leg phase offsets across diagonals."""
        phi = leg_phase_offsets  # (N,4) [FL, FR, RL, RR]
        d0 = 0.5 * (phi[:, 0:1] + phi[:, 3:4])  # FL&RR
        d1 = 0.5 * (phi[:, 1:2] + phi[:, 2:3])  # FR&RL
        coupled = torch.cat([d0, d1, d1, d0], dim=1)
        return (1.0 - self.k_phase) * phi + self.k_phase * coupled

    @torch.no_grad()
    def _couple_diagonal_amplitudes(self, amplitudes: torch.Tensor) -> torch.Tensor:
        """
        Soft/hard tie per-leg amplitudes across diagonals.
        amplitudes: (N,12) ordered by [FL(c,f,t), FR(c,f,t), RL(c,f,t), RR(c,f,t)]
        """
        N = amplitudes.shape[0]
        amps = amplitudes.view(N, 4, 3)         # (N,4legs,3joints)
        # diagonal 0: legs 0 (FL) & 3 (RR), diagonal 1: legs 1 (FR) & 2 (RL)
        avg0 = 0.5 * (amps[:, 0, :] + amps[:, 3, :])  # (N,3)
        avg1 = 0.5 * (amps[:, 1, :] + amps[:, 2, :])  # (N,3)
        k = self.k_amp
        # Blend each leg toward its diagonal average
        amps[:, 0, :] = (1 - k) * amps[:, 0, :] + k * avg0
        amps[:, 3, :] = (1 - k) * amps[:, 3, :] + k * avg0
        amps[:, 1, :] = (1 - k) * amps[:, 1, :] + k * avg1
        amps[:, 2, :] = (1 - k) * amps[:, 2, :] + k * avg1
        return amps.reshape(N, 12)

    def compute_joint_targets(
        self,
        frequency: torch.Tensor,        # (N,1)
        amplitudes: torch.Tensor,       # (N,12)
        leg_phase_offsets: torch.Tensor # (N,4)
    ) -> torch.Tensor:

        # 1) Couple diagonal phases (FL↔RR, FR↔RL)
        leg_phase_offsets = self._couple_diagonal_phases(leg_phase_offsets)

        # 2) Expand to joint phases and add trot template
        joint_phases = leg_phase_offsets.repeat_interleave(3, dim=1) + self.default_joint_phases  # (N,12)

        # 3) Couple diagonal amplitudes, then apply joint-type scaling
        amplitudes = self._couple_diagonal_amplitudes(amplitudes)
        scaled_amplitudes = amplitudes * self.joint_type_scales

        # 4) Hopf oscillator step → joint deltas
        return self.cpg.step(frequency, scaled_amplitudes, joint_phases)


class SpiderCPG_Advanced:
    """
    Advanced CPG: interpret 12 amps as [4 coxa, 4 femur, 4 tibia] then interleave.
    (Now uses trot phases and safe reshape.)
    """
    def __init__(self, num_envs: int, dt: float, device: str, k_phase: float = 0.7, k_amp: float = 1.0):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.k_phase = float(k_phase)
        self.k_amp = float(k_amp)
        self.cpg = HopfCPG(num_envs, dt=dt, device=device)

        leg_phases = torch.tensor([0.0, math.pi, math.pi, 0.0], device=device)
        self.default_joint_phases = leg_phases.repeat_interleave(3).unsqueeze(0)

    def reset(self, env_ids: torch.Tensor):
        self.cpg.reset(env_ids)

    @torch.no_grad()
    def _couple_diagonal_phases(self, leg_phase_offsets: torch.Tensor) -> torch.Tensor:
        phi = leg_phase_offsets
        d0 = 0.5 * (phi[:, 0:1] + phi[:, 3:4])
        d1 = 0.5 * (phi[:, 1:2] + phi[:, 2:3])
        coupled = torch.cat([d0, d1, d1, d0], dim=1)
        return (1.0 - self.k_phase) * phi + self.k_phase * coupled

    @torch.no_grad()
    def _couple_diagonal_amplitudes_legwise(self, coxa_amps, femur_amps, tibia_amps):
        # Each is (N,4) in [FL, FR, RL, RR] order; couple FL↔RR and FR↔RL
        def couple_4(v):
            avg0 = 0.5 * (v[:, 0:1] + v[:, 3:4])
            avg1 = 0.5 * (v[:, 1:2] + v[:, 2:3])
            k = self.k_amp
            out = v.clone()
            out[:, 0:1] = (1 - k) * v[:, 0:1] + k * avg0
            out[:, 3:4] = (1 - k) * v[:, 3:4] + k * avg0
            out[:, 1:2] = (1 - k) * v[:, 1:2] + k * avg1
            out[:, 2:3] = (1 - k) * v[:, 2:3] + k * avg1
            return out
        return couple_4(coxa_amps), couple_4(femur_amps), couple_4(tibia_amps)

    def compute_joint_targets(
        self,
        frequency: torch.Tensor,           # (N,1)
        amplitude_params: torch.Tensor,    # (N,12) = [4 coxa, 4 femur, 4 tibia]
        leg_phase_offsets: torch.Tensor    # (N,4)
    ) -> torch.Tensor:

        # Couple phases
        leg_phase_offsets = self._couple_diagonal_phases(leg_phase_offsets)

        # Unpack amplitudes by joint type
        coxa_amps  = amplitude_params[:, 0:4]
        femur_amps = amplitude_params[:, 4:8]
        tibia_amps = amplitude_params[:, 8:12]

        # Couple diagonal amplitudes within each joint type
        coxa_amps, femur_amps, tibia_amps = self._couple_diagonal_amplitudes_legwise(coxa_amps, femur_amps, tibia_amps)

        # Interleave back to joint-major order
        amplitudes = torch.stack([coxa_amps, femur_amps, tibia_amps], dim=2)  # (N,4,3)
        amplitudes = amplitudes.reshape(amplitude_params.shape[0], 12)        # (N,12)

        # Expand to joint phases
        joint_phases = leg_phase_offsets.repeat_interleave(3, dim=1) + self.default_joint_phases

        return self.cpg.step(frequency, amplitudes, joint_phases)
