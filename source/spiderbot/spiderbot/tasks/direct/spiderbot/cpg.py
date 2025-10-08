"""
Central Pattern Generator (CPG) Module for Spider Robot Locomotion
==================================================================

This module implements Hopf oscillator-based CPGs for bio-inspired locomotion.

WHY CPGs?
---------
Real spiders don't control each leg joint independently. They have neural oscillators
(CPGs) in their nervous system that generate rhythmic patterns. These oscillators are
coupled together to coordinate legs.

HOPF OSCILLATOR:
----------------
The Hopf oscillator is a nonlinear dynamical system that produces stable limit cycles
(smooth oscillations). It's described by:

    ẋ = α(μ - r²)x - ωy
    ẏ = α(μ - r²)y + ωx
    
where:
    - x, y: oscillator state (forms a 2D oscillation)
    - r² = x² + y²: distance from origin
    - α: convergence rate (how fast it reaches stable oscillation)
    - μ: target amplitude (typically 1.0)
    - ω: frequency (how fast it oscillates)

The key property: No matter where you start (x, y), the system converges to a stable
circular oscillation with radius √μ and frequency ω.

ARCHITECTURE:
-------------
    RL Policy → [freq, amp1...amp12, phase1...phase4] → CPG Layer → Joint Angles → Robot
    
    - RL learns WHAT rhythm to use (parameters)
    - CPG generates HOW to move rhythmically (oscillations)
    - Coordination emerges from phase coupling
"""

import torch
import math


class HopfCPG:
    """
    Hopf oscillator-based Central Pattern Generator.
    
    Each oscillator produces smooth sinusoidal patterns that can be phase-coupled
    to other oscillators for coordinated movement.
    """
    
    def __init__(self, num_envs: int, num_oscillators: int, dt: float, device: str):
        """
        Initialize Hopf CPG network.
        
        Args:
            num_envs: Number of parallel simulation environments
            num_oscillators: Number of oscillators (12 for 4 legs × 3 joints)
            dt: Integration timestep (should match physics timestep)
            device: 'cuda' or 'cpu'
        """
        self.num_envs = num_envs
        self.num_oscillators = num_oscillators
        self.dt = dt
        self.device = device
        
        # Oscillator states: each oscillator has (x, y) coordinates
        # Shape: [num_envs, num_oscillators]
        self.x = torch.zeros(num_envs, num_oscillators, device=device)
        self.y = torch.zeros(num_envs, num_oscillators, device=device)
        
        # Initialize with small random values to break symmetry
        # (otherwise all oscillators start at origin and stay synchronized)
        self.x += torch.randn_like(self.x) * 0.1
        self.y += torch.randn_like(self.y) * 0.1
        
        # Convergence rate: controls how quickly oscillator reaches limit cycle
        # Higher α = faster convergence but more aggressive
        # Biological value: 20-100 (we use 50 as moderate)
        self.alpha = 50.0
        
    def reset(self, env_ids: torch.Tensor):
        """
        Reset CPG states for specific environments (called after episode ends).
        
        Args:
            env_ids: Indices of environments to reset
        """
        self.x[env_ids] = torch.randn(len(env_ids), self.num_oscillators, device=self.device) * 0.1
        self.y[env_ids] = torch.randn(len(env_ids), self.num_oscillators, device=self.device) * 0.1
    
    def step(
        self, 
        frequency: torch.Tensor, 
        amplitude: torch.Tensor, 
        phase: torch.Tensor
    ) -> torch.Tensor:
        """
        Step the CPG forward one timestep using Hopf oscillator dynamics.
        
        Args:
            frequency: [num_envs, 1] - Angular frequency ω (rad/s) for all oscillators
            amplitude: [num_envs, num_oscillators] - Amplitude A for each oscillator
            phase: [num_envs, num_oscillators] - Phase offset φ for each oscillator (rad)
            
        Returns:
            Joint angle deltas: [num_envs, num_oscillators] - to add to default joint positions
            
        How it works:
        -------------
        1. Integrate Hopf equations to get new (x, y) states
        2. Apply phase shifts to coordinate oscillators
        3. Scale by amplitude to control movement range
        4. Output is the oscillator's x-component (which oscillates smoothly)
        """
        # Broadcast frequency to all oscillators
        # [num_envs, 1] → [num_envs, num_oscillators]
        omega = frequency.expand(-1, self.num_oscillators)
        
        # Compute squared radius: r² = x² + y²
        # This is used in the Hopf equations to create a stable limit cycle
        r_squared = self.x**2 + self.y**2
        
        # Hopf oscillator differential equations
        # These equations create a circular attractor in the (x, y) plane
        mu = 1.0  # Target radius (will scale with amplitude later)
        
        # dx/dt = α(μ - r²)x - ωy
        # This term pulls toward (or pushes away from) radius μ
        dx_dt = self.alpha * (mu - r_squared) * self.x - omega * self.y
        
        # dy/dt = α(μ - r²)y + ωx
        # This term creates rotation at frequency ω
        dy_dt = self.alpha * (mu - r_squared) * self.y + omega * self.x
        
        # Euler integration: x(t+dt) = x(t) + dx/dt * dt
        # For more accuracy, could use Runge-Kutta, but Euler is sufficient for small dt
        self.x = self.x + dx_dt * self.dt
        self.y = self.y + dy_dt * self.dt
        
        # Apply phase shifting by rotating the (x, y) vector
        # This is how we coordinate oscillators: shift their phase relative to each other
        # Rotation formula: x' = x*cos(φ) - y*sin(φ)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        x_shifted = self.x * cos_phase - self.y * sin_phase
        
        # Scale by amplitude to control movement range
        # Output is the x-component of the oscillator (smoothly oscillates)
        output = amplitude * x_shifted
        
        return output


class SpiderCPG:
    """
    Spider-specific CPG controller.
    
    Architecture:
    -------------
    - 4 legs: Front-Left (FL), Front-Right (FR), Back-Left (BL), Back-Right (BR)
    - 3 joints per leg: Hip, Knee, Ankle
    - Total: 12 oscillators (4 × 3)
    
    Phase Coordination:
    -------------------
    Default gait is TETRAPOD (alternating diagonal pairs):
    - FL + BR move together (phase 0)
    - FR + BL move together (phase π)
    
    This creates a stable, energy-efficient gait used by many quadrupeds.
    The RL policy can modify these phases to adapt to terrain or speed.
    """
    
    def __init__(self, num_envs: int, dt: float, device: str):
        """
        Initialize spider CPG controller.
        
        Args:
            num_envs: Number of parallel environments
            dt: Physics timestep
            device: 'cuda' or 'cpu'
        """
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        
        # Create 12 Hopf oscillators (one per joint)
        self.cpg = HopfCPG(num_envs, num_oscillators=12, dt=dt, device=device)
        
        # Default phase offsets for tetrapod gait
        # Tetrapod = diagonal legs move together
        # Legs: [FL, FR, BL, BR]
        self.default_leg_phases = torch.tensor([
            0.0,      # Front-Left: phase 0
            math.pi,  # Front-Right: phase π (opposite)
            math.pi,  # Back-Left: phase π (opposite)  
            0.0,      # Back-Right: phase 0
        ], device=device)
        
    def reset(self, env_ids: torch.Tensor):
        """Reset CPG for specific environments."""
        self.cpg.reset(env_ids)
    
    def compute_joint_targets(
        self,
        frequency: torch.Tensor,           # [num_envs, 1]
        amplitudes: torch.Tensor,          # [num_envs, 12]
        leg_phase_offsets: torch.Tensor    # [num_envs, 4]
    ) -> torch.Tensor:
        """
        Compute target joint angles from CPG parameters.
        
        Args:
            frequency: Single frequency for all oscillators (shared rhythm)
            amplitudes: Individual amplitude for each joint (controls stride)
            leg_phase_offsets: Phase offset for each leg (modifies gait pattern)
            
        Returns:
            Joint angle deltas [num_envs, 12]: to add to default joint positions
            
        How phase works:
        ----------------
        Each leg gets a base phase (tetrapod) + learned offset:
        - Base: FL=0, FR=π, BL=π, BR=0 (tetrapod gait)
        - RL adds offsets to adapt gait to terrain/speed
        - All 3 joints in a leg share the same phase (move as a unit)
        """
        # Expand leg phases to all joints (3 joints per leg)
        # [num_envs, 4] → [num_envs, 12]
        # Each leg's phase is repeated 3 times for hip, knee, ankle
        joint_phases = leg_phase_offsets.repeat_interleave(3, dim=1)
        
        # Add default tetrapod pattern as a starting point
        # This biases toward a known stable gait
        default_phases = self.default_leg_phases.repeat_interleave(3).unsqueeze(0)  # [1, 12]
        joint_phases = joint_phases + default_phases
        
        # Step CPG to get joint angle deltas
        joint_deltas = self.cpg.step(frequency, amplitudes, joint_phases)
        
        return joint_deltas