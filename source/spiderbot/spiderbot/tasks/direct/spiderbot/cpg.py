"""Advanced Van der Pol CPG for Spider Robot Locomotion

Van der Pol (VDP) oscillators are superior to Hopf for quadruped locomotion:
1. Non-sinusoidal waveforms with distinct swing/stance phases
2. Adjustable duty cycle (stance vs swing duration)
3. Relaxation oscillations (more biologically realistic)
4. Better synchronization properties for multi-leg coordination
5. Can generate sharper transitions between phases

Research basis:
- "Van Der Pol Central Pattern Generator (VDP-CPG) Model for Quadruped Robot" (2013)
- "Gaits generation of quadruped locomotion for the CPG controller by 
   the delay-coupled VDP oscillators" (Nonlinear Dynamics, 2023)
- Recent 2024-2025 studies on VDP-based CPG for adaptive locomotion
"""

import torch
import math


class VanDerPolOscillator:
    """Van der Pol oscillator for CPG.
    
    The VDP oscillator is a non-conservative oscillator with nonlinear damping:
    ẍ - μ(1 - x²)ẋ + ω²x = 0
    
    We use a modified form with explicit amplitude control:
    ẍ - μ(1 - (x/A)²)ẋ + ω²x = 0
    
    Key properties:
    - Limit cycle behavior (stable oscillations)
    - Adjustable duty cycle via μ parameter
    - Non-sinusoidal waveforms (relaxation oscillations)
    - Natural swing/stance phase distinction
    """
    
    def __init__(
        self, 
        num_envs: int, 
        dt: float, 
        device: str,
        mu: float = 2.0,        # Nonlinearity parameter (higher = sharper transitions)
        freq_floor_hz: float = 1e-3
    ):
        """Initialize VDP oscillator.
        
        Args:
            num_envs: Number of parallel environments
            dt: Integration timestep
            device: torch device
            mu: Nonlinearity parameter (1-5 typical, higher = more relaxation)
            freq_floor_hz: Minimum frequency to prevent instability
        """
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.freq_floor = freq_floor_hz
        self.mu = mu  # Nonlinearity strength
        
        # State: position and velocity
        self.x = torch.zeros(num_envs, 1, device=device)
        self.v = torch.zeros(num_envs, 1, device=device)
        
        self.reset(torch.arange(num_envs, device=device))
    
    def reset(self, env_ids: torch.Tensor):
        """Reset oscillator state."""
        # Start from limit cycle
        self.x[env_ids] = 0.5
        self.v[env_ids] = 0.0
    
    @torch.no_grad()
    def step(
        self, 
        frequency: torch.Tensor,   # (N,1)
        amplitude: torch.Tensor,   # (N,1)
        phase: torch.Tensor        # (N,1)
    ) -> torch.Tensor:
        """Step the VDP oscillator.
        
        Modified VDP equation with amplitude control:
        ẍ = μ(1 - (x/A)²)ẋ - ω²x
        
        Args:
            frequency: Oscillation frequency (Hz)
            amplitude: Target amplitude
            phase: Phase offset (rad)
            
        Returns:
            output: Oscillator output (N,1)
        """
        # Clamp frequency and amplitude to safe ranges
        omega = 2.0 * math.pi * torch.clamp(frequency, min=self.freq_floor, max=10.0)
        omega_sq = omega * omega
        
        # Clamp amplitude to prevent division by zero
        A = torch.clamp(amplitude, min=0.01, max=2.0)
        
        # VDP dynamics with amplitude scaling
        # ẍ = μ(1 - (x/A)²)ẋ - ω²x
        x_scaled = torch.clamp(self.x / A, -5.0, 5.0)  # Prevent overflow
        damping_term = self.mu * (1.0 - x_scaled * x_scaled) * self.v
        restoring_term = -omega_sq * self.x
        
        accel = damping_term + restoring_term
        
        # Clamp acceleration to prevent instability
        accel = torch.clamp(accel, -100.0, 100.0)
        
        # Euler integration
        self.v += accel * self.dt
        self.x += self.v * self.dt
        
        # Clamp state to prevent divergence
        self.x = torch.clamp(self.x, -10.0, 10.0)
        self.v = torch.clamp(self.v, -50.0, 50.0)
        
        # Safety: Replace any NaN/Inf with zeros
        self.x = torch.nan_to_num(self.x, 0.0)
        self.v = torch.nan_to_num(self.v, 0.0)
        
        # Apply phase shift
        phase = torch.clamp(phase, -2*math.pi, 2*math.pi)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        
        # Rotate in phase space
        x_shifted = self.x * cos_phase - self.v / (omega + 1e-6) * sin_phase
        
        # Final safety check
        x_shifted = torch.nan_to_num(x_shifted, 0.0)
        
        return x_shifted
    
    @torch.no_grad()
    def phase_angle(self) -> torch.Tensor:
        """Return current phase angle φ in [-π,π]."""
        # For VDP, phase is more complex than simple atan2
        # Use approximation based on position and velocity
        omega = 2.0 * math.pi * 1.0  # Nominal frequency for phase computation
        
        # Safety: clamp values before atan2
        x_safe = torch.clamp(self.x, -10.0, 10.0)
        v_safe = torch.clamp(self.v / (omega + 1e-6), -10.0, 10.0)
        
        phase = torch.atan2(v_safe, x_safe)
        
        # Ensure finite
        phase = torch.nan_to_num(phase, 0.0)
        
        return phase


class AdaptiveVDPCPG:
    """Advanced VDP-based CPG for quadruped locomotion.
    
    Features:
    1. Van der Pol oscillators (better than Hopf)
    2. Adaptive gait with learned phase relationships
    3. Diagonal coupling for quadruped coordination
    4. Multi-gait capability (walk, trot, pace, bound)
    5. Sensory feedback integration (via policy)
    
    Architecture:
    - 4 VDP oscillators (one per leg)
    - Phase coupling between diagonals
    - Independent amplitude per joint type
    - Learned gait parameters
    """
    
    def __init__(
        self, 
        num_envs: int, 
        dt: float, 
        device: str,
        mu: float = 2.5,           # VDP nonlinearity (2-3 for locomotion)
        k_phase: float = 0.5,      # Phase coupling strength (0-1)
        k_amp: float = 0.3         # Amplitude coupling strength (0-1)
    ):
        """Initialize adaptive VDP CPG.
        
        Args:
            num_envs: Number of parallel environments
            dt: Integration timestep
            device: torch device
            mu: VDP nonlinearity parameter
            k_phase: Phase coupling strength (diagonal coordination)
            k_amp: Amplitude coupling strength
        """
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.k_phase = float(k_phase)
        self.k_amp = float(k_amp)
        
        # 4 VDP oscillators (one per leg: FL, FR, RL, RR)
        self.oscillators = [
            VanDerPolOscillator(num_envs, dt, device, mu=mu)
            for _ in range(4)
        ]
        
        # Gait templates (phase relationships)
        # These are BASE phases - policy can modulate them
        self.gait_templates = {
            'walk': torch.tensor([0.0, 0.5*math.pi, math.pi, 1.5*math.pi], device=device),
            'trot': torch.tensor([0.0, math.pi, math.pi, 0.0], device=device),
            'pace': torch.tensor([0.0, math.pi, 0.0, math.pi], device=device),
            'bound': torch.tensor([0.0, 0.0, math.pi, math.pi], device=device),
        }
        
        # Default to trot for spiders (most stable at low speed)
        self.default_gait = self.gait_templates['trot'].unsqueeze(0)  # (1,4)
        
        # Intra-leg phase offsets (femur/tibia lag behind coxa)
        self.intra_leg_offsets = torch.tensor([
            0.0, 0.3*math.pi, 0.6*math.pi,  # FL: coxa, femur, tibia
            0.0, 0.3*math.pi, 0.6*math.pi,  # FR
            0.0, 0.3*math.pi, 0.6*math.pi,  # RL
            0.0, 0.3*math.pi, 0.6*math.pi,  # RR
        ], device=device).unsqueeze(0)  # (1,12)
    
    def reset(self, env_ids: torch.Tensor):
        """Reset all oscillators."""
        for osc in self.oscillators:
            osc.reset(env_ids)
    
    @torch.no_grad()
    def _couple_diagonal_phases(self, leg_phase_offsets: torch.Tensor) -> torch.Tensor:
        """Couple diagonal leg phases for coordination.
        
        Diagonals in quadrupeds naturally coordinate:
        - Diagonal 0: FL & RR
        - Diagonal 1: FR & RL
        
        Args:
            leg_phase_offsets: (N,4) phase offsets for [FL, FR, RL, RR]
            
        Returns:
            coupled_phases: (N,4) coupled phase offsets
        """
        phi = leg_phase_offsets
        
        # Average diagonal phases
        d0 = 0.5 * (phi[:, 0:1] + phi[:, 3:4])  # FL & RR
        d1 = 0.5 * (phi[:, 1:2] + phi[:, 2:3])  # FR & RL
        
        # Blend original and coupled
        coupled = torch.cat([d0, d1, d1, d0], dim=1)
        return (1.0 - self.k_phase) * phi + self.k_phase * coupled
    
    @torch.no_grad()
    def _couple_diagonal_amplitudes(self, amplitudes: torch.Tensor) -> torch.Tensor:
        """Couple diagonal leg amplitudes.
        
        Args:
            amplitudes: (N,12) amplitudes for all joints
            
        Returns:
            coupled_amps: (N,12) coupled amplitudes
        """
        N = amplitudes.shape[0]
        amps = amplitudes.view(N, 4, 3)  # (N, 4legs, 3joints)
        
        # Average diagonal amplitudes
        avg0 = 0.5 * (amps[:, 0, :] + amps[:, 3, :])  # FL & RR
        avg1 = 0.5 * (amps[:, 1, :] + amps[:, 2, :])  # FR & RL
        
        # Blend
        k = self.k_amp
        amps[:, 0, :] = (1 - k) * amps[:, 0, :] + k * avg0
        amps[:, 3, :] = (1 - k) * amps[:, 3, :] + k * avg0
        amps[:, 1, :] = (1 - k) * amps[:, 1, :] + k * avg1
        amps[:, 2, :] = (1 - k) * amps[:, 2, :] + k * avg1
        
        return amps.reshape(N, 12)
    
    def compute_joint_targets(
        self,
        frequency: torch.Tensor,        # (N,1) shared frequency
        amplitudes: torch.Tensor,       # (N,12) per-joint amplitudes
        leg_phase_offsets: torch.Tensor # (N,4) per-leg phase offsets
    ) -> torch.Tensor:
        """Compute joint targets from CPG.
        
        Args:
            frequency: Oscillation frequency (Hz)
            amplitudes: Joint amplitudes [FL(c,f,t), FR(c,f,t), RL(c,f,t), RR(c,f,t)]
            leg_phase_offsets: Phase offsets for each leg
            
        Returns:
            joint_deltas: (N,12) joint position deltas from default
        """
        # Couple phases for diagonal coordination
        leg_phases = self._couple_diagonal_phases(leg_phase_offsets)
        
        # Add default gait template
        leg_phases = leg_phases + self.default_gait
        
        # Expand to joint phases with intra-leg offsets
        joint_phases = leg_phases.repeat_interleave(3, dim=1) + self.intra_leg_offsets
        
        # Couple amplitudes
        amplitudes = self._couple_diagonal_amplitudes(amplitudes)
        
        # Step each oscillator and collect outputs
        outputs = []
        for i, osc in enumerate(self.oscillators):
            # Each oscillator gets the shared frequency
            # and its corresponding amplitude (3 joints per leg)
            leg_amps = amplitudes[:, i*3:(i+1)*3].mean(dim=1, keepdim=True)  # Average for oscillator
            leg_phase = joint_phases[:, i*3]  # Use coxa phase for oscillator
            
            output = osc.step(frequency, leg_amps, leg_phase.unsqueeze(1))
            outputs.append(output)
        
        # Stack outputs (N,4) and expand to joints (N,12)
        leg_outputs = torch.cat(outputs, dim=1)  # (N,4)
        
        # Scale each joint by its specific amplitude
        # Each leg output is distributed to 3 joints with their specific amps
        joint_outputs = torch.zeros(self.num_envs, 12, device=self.device)
        for i in range(4):
            for j in range(3):
                joint_idx = i * 3 + j
                joint_outputs[:, joint_idx] = (
                    leg_outputs[:, i] * amplitudes[:, joint_idx]
                )
        
        return joint_outputs
    
    @torch.no_grad()
    def get_phase_features(self) -> torch.Tensor:
        """Get phase features for observation.
        
        Returns:
            features: (N,8) [sin_FL, cos_FL, sin_FR, cos_FR, sin_RL, cos_RL, sin_RR, cos_RR]
        """
        features = []
        for osc in self.oscillators:
            phase = osc.phase_angle()
            sin_phase = torch.sin(phase)
            cos_phase = torch.cos(phase)
            
            # Safety: ensure finite
            sin_phase = torch.nan_to_num(sin_phase, 0.0)
            cos_phase = torch.nan_to_num(cos_phase, 1.0)  # Default to cos(0)=1
            
            features.append(sin_phase)
            features.append(cos_phase)
        
        result = torch.cat(features, dim=1)  # (N,8)
        
        # Final safety check
        result = torch.nan_to_num(result, 0.0)
        
        return result