import torch
import math

class HopfCPG:
    def __init__(self, num_envs: int, dt: float, device: str, freq_floor_hz: float = 1e-3):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.freq_floor = freq_floor_hz
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
        """
        Generate oscillator output with per-joint phase and amplitude.
        
        Args:
            frequency: (N,1) - global frequency for all joints
            amplitude: (N,12) - per-joint amplitude scaling
            phase: (N,12) - per-joint phase offset
        
        Returns:
            (N,12) - joint target deltas
        """
        omega = 2.0 * math.pi * torch.clamp(frequency, min=self.freq_floor)

        r2 = self.x * self.x + self.y * self.y
        mu = 1.0
        dx_dt = self.alpha * (mu - r2) * self.x - omega * self.y
        dy_dt = self.alpha * (mu - r2) * self.y + omega * self.x

        self.x += dx_dt * self.dt
        self.y += dy_dt * self.dt

        # Broadcast single oscillator to all joints with their phases
        cos_phase = torch.cos(phase)  # (N,12)
        sin_phase = torch.sin(phase)  # (N,12)
        x_shifted = self.x * cos_phase - self.y * sin_phase
        
        return amplitude * x_shifted


class SpiderCPG:
    """
    Spider locomotion CPG with joint-specific amplitude control.
    
    Key insight: Different joint types need different movement magnitudes:
    - Coxa (hip): Large amplitude for leg sweep (forward/back or left/right)
    - Femur (shoulder): Medium amplitude for ground clearance  
    - Tibia (knee): Small amplitude for step adjustment
    """
    
    def __init__(self, num_envs: int, dt: float, device: str):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device

        # Single-oscillator internal CPG
        self.cpg = HopfCPG(num_envs, dt=dt, device=device)

        # ====================================================================
        # TROT GAIT: Diagonal legs move together (FL+RR, FR+RL)
        # ====================================================================
        # Base phases for 4 legs
        leg_phases = torch.tensor([0.0, math.pi, 0.0, math.pi], device=device)
        
        # Expand to 12 joints (each leg has 3 joints: coxa, femur, tibia)
        self.default_joint_phases = leg_phases.repeat_interleave(3).unsqueeze(0)  # (1,12)
        
        # ====================================================================
        # JOINT-SPECIFIC AMPLITUDE SCALING
        # ====================================================================
        # This is the KEY FIX for direction control
        # Different joint types should oscillate with different magnitudes
        
        # Create amplitude scale pattern: [coxa, femur, tibia] × 4 legs
        # EXPERIMENT with these values to control motion direction!
        
        # Option A: Forward motion (assuming coxa controls forward/back)
        joint_type_scales = torch.tensor([
            1.0, 0.3, 0.2,  # FL: high coxa, low femur/tibia
            1.0, 0.3, 0.2,  # FR
            1.0, 0.3, 0.2,  # RL  
            1.0, 0.3, 0.2,  # RR
        ], device=device).unsqueeze(0)  # (1,12)
        
        self.joint_type_scales = joint_type_scales
        
        # ====================================================================
        # ALTERNATIVE SCALING PATTERNS (uncomment to try)
        # ====================================================================
        
        # # Option B: If coxa controls lateral (left/right), use femur for forward
        # self.joint_type_scales = torch.tensor([
        #     0.3, 1.0, 0.2,  # FL: low coxa, high femur
        #     0.3, 1.0, 0.2,  # FR
        #     0.3, 1.0, 0.2,  # RL
        #     0.3, 1.0, 0.2,  # RR
        # ], device=device).unsqueeze(0)
        
        # # Option C: Equal all joints (your current behavior)
        # self.joint_type_scales = torch.ones(1, 12, device=device)
        
        # # Option D: Only coxa moves (pure leg sweep)
        # self.joint_type_scales = torch.tensor([
        #     1.0, 0.0, 0.0,  # FL
        #     1.0, 0.0, 0.0,  # FR
        #     1.0, 0.0, 0.0,  # RL
        #     1.0, 0.0, 0.0,  # RR
        # ], device=device).unsqueeze(0)

    def reset(self, env_ids: torch.Tensor):
        self.cpg.reset(env_ids)

    def compute_joint_targets(
        self, 
        frequency: torch.Tensor,        # (N,1)
        amplitudes: torch.Tensor,       # (N,12) - raw amplitudes from policy
        leg_phase_offsets: torch.Tensor  # (N,4) - per-leg phase offsets
    ):
        """
        Compute joint position targets using CPG.
        
        Args:
            frequency: Global oscillation frequency (Hz)
            amplitudes: Per-joint amplitude from RL policy  
            leg_phase_offsets: Additional phase offset per leg (for turning/gait variation)
        
        Returns:
            (N,12) joint position deltas
        """
        # Expand leg phases to joint phases
        joint_phases = leg_phase_offsets.repeat_interleave(3, dim=1)  # (N,12)
        joint_phases = joint_phases + self.default_joint_phases        # Add trot pattern
        
        # Apply joint-type-specific scaling to amplitudes
        # This makes coxa move more than femur/tibia (or vice versa)
        scaled_amplitudes = amplitudes * self.joint_type_scales  # (N,12) * (1,12) -> (N,12)
        
        # Generate oscillator output
        return self.cpg.step(frequency, scaled_amplitudes, joint_phases)


# ============================================================================
# ADVANCED: Per-joint-type amplitude control (better but needs env changes)
# ============================================================================

class SpiderCPG_Advanced:
    """
    Advanced CPG with separate amplitude control per joint type.
    
    This requires changing the action space in spiderbot_env_cfg.py:
    - OLD: action_space = 17  (1 freq + 12 amp + 4 phase)
    - NEW: action_space = 17  (1 freq + 12 amp_raw + 4 phase)
    
    But internally, we interpret the 12 amplitudes as:
    - 4 coxa amplitudes (one per leg)
    - 4 femur amplitudes  
    - 4 tibia amplitudes
    
    Then rearrange them to match joint order.
    """
    
    def __init__(self, num_envs: int, dt: float, device: str):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.cpg = HopfCPG(num_envs, dt=dt, device=device)
        
        # Trot gait phases
        leg_phases = torch.tensor([0.0, math.pi, 0.0, math.pi], device=device)
        self.default_joint_phases = leg_phases.repeat_interleave(3).unsqueeze(0)

    def reset(self, env_ids: torch.Tensor):
        self.cpg.reset(env_ids)

    def compute_joint_targets(
        self,
        frequency: torch.Tensor,           # (N,1)  
        amplitude_params: torch.Tensor,    # (N,12) - interpreted as [4 coxa, 4 femur, 4 tibia]
        leg_phase_offsets: torch.Tensor    # (N,4)
    ):
        """
        Compute targets with per-joint-type amplitude control.
        
        amplitude_params layout:
        [coxa_FL, coxa_FR, coxa_RL, coxa_RR,
         femur_FL, femur_FR, femur_RL, femur_RR,
         tibia_FL, tibia_FR, tibia_RL, tibia_RR]
        
        We need to rearrange to joint order:
        [coxa_FL, femur_FL, tibia_FL,
         coxa_FR, femur_FR, tibia_FR,
         coxa_RL, femur_RL, tibia_RL,
         coxa_RR, femur_RR, tibia_RR]
        """
        # Extract amplitude per joint type and leg
        coxa_amps = amplitude_params[:, 0:4]   # (N,4)
        femur_amps = amplitude_params[:, 4:8]   # (N,4)
        tibia_amps = amplitude_params[:, 8:12]  # (N,4)
        
        # Rearrange to match joint order (interleave by leg)
        amplitudes = torch.stack([coxa_amps, femur_amps, tibia_amps], dim=2)  # (N,4,3)
        amplitudes = amplitudes.reshape(self.num_envs, 12)  # (N,12)
        
        # Compute phases
        joint_phases = leg_phase_offsets.repeat_interleave(3, dim=1)
        joint_phases = joint_phases + self.default_joint_phases
        
        return self.cpg.step(frequency, amplitudes, joint_phases)