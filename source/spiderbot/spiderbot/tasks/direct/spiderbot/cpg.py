import torch
import math


class HopfCPG:
    def __init__(self, num_envs: int, num_oscillators: int, dt: float, device: str):
        self.num_envs = num_envs
        self.num_oscillators = num_oscillators
        self.dt = dt
        self.device = device
        
        self.x = torch.zeros(num_envs, num_oscillators, device=device)
        self.y = torch.zeros(num_envs, num_oscillators, device=device)
        
        self.x += torch.randn_like(self.x) * 0.1
        self.y += torch.randn_like(self.y) * 0.1
        
        self.alpha = 8.0
        
    def reset(self, env_ids: torch.Tensor):
        self.x[env_ids] = torch.randn(len(env_ids), self.num_oscillators, device=self.device) * 0.1
        self.y[env_ids] = torch.randn(len(env_ids), self.num_oscillators, device=self.device) * 0.1
    
    def step(
        self, 
        frequency: torch.Tensor,
        amplitude: torch.Tensor,
        phase: torch.Tensor
    ) -> torch.Tensor:
        omega = frequency.expand(-1, self.num_oscillators)
        
        r_squared = self.x**2 + self.y**2
        
        mu = 1.0
        
        dx_dt = self.alpha * (mu - r_squared) * self.x - omega * self.y
        
        dy_dt = self.alpha * (mu - r_squared) * self.y + omega * self.x
        
        self.x = self.x + dx_dt * self.dt
        self.y = self.y + dy_dt * self.dt
        
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        x_shifted = self.x * cos_phase - self.y * sin_phase
        
        output = amplitude * x_shifted
        
        return output


class SpiderCPG:
    def __init__(self, num_envs: int, dt: float, device: str):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        
        self.cpg = HopfCPG(num_envs, num_oscillators=12, dt=dt, device=device)
        
        self.default_leg_phases = torch.tensor([
            0.0,
            math.pi,
            math.pi,
            0.0,
        ], device=device)
        
    def reset(self, env_ids: torch.Tensor):
        self.cpg.reset(env_ids)
    
    def compute_joint_targets(
        self,
        frequency: torch.Tensor,
        amplitudes: torch.Tensor,
        leg_phase_offsets: torch.Tensor
    ) -> torch.Tensor:
        joint_phases = leg_phase_offsets.repeat_interleave(3, dim=1)
        
        default_phases = self.default_leg_phases.repeat_interleave(3).unsqueeze(0)
        joint_phases = joint_phases + default_phases
        
        joint_deltas = self.cpg.step(frequency, amplitudes, joint_phases)
        
        return joint_deltas