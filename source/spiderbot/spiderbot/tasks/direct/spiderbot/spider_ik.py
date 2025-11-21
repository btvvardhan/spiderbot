"""Inverse Kinematics for Spider Robot with correct URDF dimensions."""

import torch
import math
import numpy as np

class SpiderIK:
    """
    Analytical IK for spider robot legs using URDF-extracted dimensions.
    All measurements in meters and radians.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
        # Link lengths from URDF and ik_dims.json
        # Coxa lengths (hip to knee joint)
        self.L_coxa = torch.tensor([
            0.06004539929919694,  # FL (from ik_dims.json)
            0.06004539929919694,  # FR
            0.06004539929919694,  # RL
            0.05963555966870773,  # RR
        ], device=device)
        
        # Femur lengths (knee to ankle joint)
        self.L_femur = torch.tensor([
            0.07999923279632123,  # FL (from ik_dims.json)
            0.08001330320640437,  # FR
            0.07999923279632123,  # RL
            0.07999923279632123,  # RR
        ], device=device)
        
        # Tibia lengths (ankle to foot tip)
        self.L_tibia = torch.tensor([
            0.15,  # FL (from ik_dims.json)
            0.15,  # FR
            0.15,  # RL
            0.15,  # RR
        ], device=device)
        
        # Hip offsets from base center (from URDF joint origins)
        # These define where each coxa joint is relative to base_link
        self.hip_offset_x = torch.tensor([
             0.0802,   # FL: positive x (forward)
            -0.1202,   # FR: negative x (forward, but on right side)
             0.0802,   # RL: positive x (rear left)
            -0.1202,   # RR: negative x (rear right)
        ], device=device)
        
        self.hip_offset_y = torch.tensor([
            -0.1049,   # FL: negative y (left side)
            -0.10365,  # FR: negative y (right side from robot's perspective)
             0.0649,   # RL: positive y (rear left)
            -0.0649,   # RR: negative y (rear right) 
        ], device=device)
        
        self.hip_offset_z = torch.tensor([
             0.0717,   # FL
            -0.07235,  # FR  
             0.0717,   # RL
            -0.0717,   # RR
        ], device=device)
        
        # Leg configuration (which legs are on which side)
        # This affects the sign of coxa angles
        self.leg_side_sign = torch.tensor([
            1.0,   # FL: left side
            -1.0,  # FR: right side (mirror coxa)
            1.0,   # RL: left side  
            -1.0,  # RR: right side (mirror coxa)
        ], device=device)
        
        # Default stance parameters
        self.default_height = 0.18  # Standing height from ground
        self.stride_length = 0.06   # Step length for walking
        self.step_height = 0.03     # Foot lift height during swing
        
        # Neutral foot positions (relative to hip)
        # These create a stable rectangular stance
        self.neutral_foot_x = torch.tensor([
            0.12,   # FL: forward
            0.12,   # FR: forward  
            -0.12,  # RL: backward
            -0.12,  # RR: backward
        ], device=device)
        
        self.neutral_foot_y = torch.tensor([
            0.08,   # FL: outward left
            -0.08,  # FR: outward right
            0.08,   # RL: outward left
            -0.08,  # RR: outward right
        ], device=device)
        
    def compute_leg_ik(self, leg_idx: int, target_x: torch.Tensor, 
                       target_y: torch.Tensor, target_z: torch.Tensor) -> tuple:
        """
        Compute IK for a single leg.
        
        Args:
            leg_idx: Which leg (0=FL, 1=FR, 2=RL, 3=RR)
            target_x, target_y, target_z: Target foot position relative to hip joint (N,)
            
        Returns:
            (coxa_angle, femur_angle, tibia_angle) in radians (N,)
        """
        L1 = self.L_coxa[leg_idx]
        L2 = self.L_femur[leg_idx]
        L3 = self.L_tibia[leg_idx]
        side_sign = self.leg_side_sign[leg_idx]
        
        # Coxa angle (yaw rotation around Z axis)
        # Account for leg side (right legs need mirrored angles)
        coxa_angle = torch.atan2(target_y * side_sign, target_x)
        
        # Distance from coxa axis to target in XY plane
        xy_dist = torch.sqrt(target_x**2 + target_y**2)
        
        # Effective reach after accounting for coxa length
        reach = xy_dist - L1
        reach = torch.clamp(reach, min=0.01)  # Prevent negative reach
        
        # 3D distance from femur joint to target
        dist_3d = torch.sqrt(reach**2 + target_z**2)
        
        # Clamp to reachable workspace
        max_reach = L2 + L3 - 0.01  # Leave small margin
        min_reach = abs(L2 - L3) + 0.01
        dist_3d = torch.clamp(dist_3d, min=min_reach, max=max_reach)
        
        # Use law of cosines for femur and tibia angles
        # Femur angle (from horizontal plane)
        cos_alpha = (L2**2 + dist_3d**2 - L3**2) / (2 * L2 * dist_3d + 1e-8)
        cos_alpha = torch.clamp(cos_alpha, -1.0, 1.0)
        alpha = torch.acos(cos_alpha)
        
        # Angle from horizontal to target
        gamma = torch.atan2(-target_z, reach)
        
        # Femur angle combines both
        femur_angle = gamma + alpha
        
        # Tibia angle (relative to femur)
        cos_beta = (L2**2 + L3**2 - dist_3d**2) / (2 * L2 * L3 + 1e-8)
        cos_beta = torch.clamp(cos_beta, -1.0, 1.0)
        tibia_angle = torch.acos(cos_beta) - math.pi
        
        return coxa_angle, femur_angle, tibia_angle
    
    def compute_foot_trajectory(self, phase: torch.Tensor, cmd_x: torch.Tensor, 
                               cmd_y: torch.Tensor, leg_idx: int) -> tuple:
        """
        Generate foot trajectory based on gait phase and velocity commands.
        
        Args:
            phase: Gait phase [0, 2π] for each env (N,)
            cmd_x, cmd_y: Velocity commands in m/s (N,)
            leg_idx: Which leg (0=FL, 1=FR, 2=RL, 3=RR)
            
        Returns:
            (x, y, z) foot positions relative to hip joint (N,)
        """
        N = phase.shape[0]
        device = phase.device
        
        # Normalize phase to [0, 1]
        norm_phase = phase / (2 * math.pi)
        
        # Determine stance vs swing (0.0-0.5 = stance, 0.5-1.0 = swing)
        is_stance = norm_phase < 0.5
        
        # Swing phase normalized to [0, 1]
        swing_phase = torch.where(is_stance, 
                                  torch.zeros_like(norm_phase),
                                  (norm_phase - 0.5) * 2.0)
        
        # Stance phase normalized to [0, 1]
        stance_phase = torch.where(is_stance,
                                   norm_phase * 2.0,
                                   torch.ones_like(norm_phase))
        
        # Base neutral position for this leg
        neutral_x = self.neutral_foot_x[leg_idx]
        neutral_y = self.neutral_foot_y[leg_idx]
        
        # Velocity-based stride adjustment
        stride_x = cmd_x * self.stride_length
        stride_y = cmd_y * self.stride_length * 0.5  # Less lateral movement
        
        # Stance phase: foot moves backward relative to body
        stance_x = neutral_x - stride_x * stance_phase
        stance_y = neutral_y - stride_y * stance_phase
        stance_z = -self.default_height * torch.ones(N, device=device)
        
        # Swing phase: foot lifts and moves forward
        # Use smooth sine curve for vertical movement
        lift = torch.sin(swing_phase * math.pi) * self.step_height
        
        # Foot moves from back to front during swing
        # swing_phase=0: start at backward position (where stance ended)
        # swing_phase=1: end at forward position (where next stance starts)
        swing_x = neutral_x - stride_x * (1.0 - 2.0 * swing_phase)  # Fixed: changed sign
        swing_y = neutral_y - stride_y * (1.0 - 2.0 * swing_phase)  # Fixed: changed sign
        swing_z = -self.default_height + lift
        
        # Combine based on phase
        x = torch.where(is_stance, stance_x, swing_x)
        y = torch.where(is_stance, stance_y, swing_y)
        z = torch.where(is_stance, stance_z, swing_z)
        
        return x, y, z
    
    def compute_all_legs_ik(self, phases: torch.Tensor, cmd_x: torch.Tensor,
                            cmd_y: torch.Tensor, cmd_yaw: torch.Tensor) -> torch.Tensor:
        """
        Compute IK targets for all legs based on gait phases and commands.
        
        Args:
            phases: Gait phases for each leg (N, 4) in radians [0, 2π]
            cmd_x, cmd_y: Linear velocity commands in m/s (N,)
            cmd_yaw: Angular velocity command in rad/s (N,)
            
        Returns:
            Joint angles for all 12 joints (N, 12) in radians
        """
        N = phases.shape[0]
        joint_angles = torch.zeros(N, 12, device=self.device)
        
        for leg in range(4):
            # Adjust commands for turning
            # Legs on opposite sides move differently during turns
            if leg in [0, 2]:  # Left legs (FL, RL)
                # Add turning component
                turn_offset_x = -cmd_yaw * self.hip_offset_y[leg] * 0.5
                turn_offset_y = cmd_yaw * self.hip_offset_x[leg] * 0.5
            else:  # Right legs (FR, RR)
                turn_offset_x = -cmd_yaw * self.hip_offset_y[leg] * 0.5
                turn_offset_y = cmd_yaw * self.hip_offset_x[leg] * 0.5
            
            # Get foot trajectory with turn adjustment
            fx, fy, fz = self.compute_foot_trajectory(
                phases[:, leg], 
                cmd_x + turn_offset_x, 
                cmd_y + turn_offset_y,
                leg
            )
            
            # Compute IK for this leg
            coxa, femur, tibia = self.compute_leg_ik(leg, fx, fy, fz)
            
            # Store in joint array [coxa, femur, tibia] for each leg
            joint_angles[:, leg*3] = coxa
            joint_angles[:, leg*3 + 1] = femur
            joint_angles[:, leg*3 + 2] = tibia
        
        # Apply joint limits
        joint_angles = torch.clamp(joint_angles, -1.57, 1.57)  # ±90 degrees
        
        return joint_angles
    
    def get_trot_phases(self, t: torch.Tensor, frequency: torch.Tensor) -> torch.Tensor:
        """
        Generate trot gait phases for all legs.
        Diagonal pairs move together: (FL, RR) and (FR, RL)
        
        Args:
            t: Time or base phase (N,)
            frequency: Gait frequency in Hz (N,)
            
        Returns:
            Phases for all 4 legs (N, 4) in radians [0, 2π]
        """
        N = t.shape[0]
        
        # Base phase
        base_phase = (2 * math.pi * frequency * t) % (2 * math.pi)
        
        # Trot pattern: diagonal legs in phase, opposite pairs π out of phase
        phases = torch.zeros(N, 4, device=self.device)
        phases[:, 0] = base_phase  # FL
        phases[:, 1] = (base_phase + math.pi) % (2 * math.pi)  # FR (opposite)
        phases[:, 2] = (base_phase + math.pi) % (2 * math.pi)  # RL (with FR)
        phases[:, 3] = base_phase  # RR (with FL)
        
        return phases
    
    def get_home_position(self) -> torch.Tensor:
        """
        Get home/neutral joint positions for all legs.
        
        Returns:
            Joint angles for neutral stance (12,) in radians
        """
        # Create a single sample
        phases = torch.zeros(1, 4, device=self.device)
        cmd_x = torch.zeros(1, device=self.device)
        cmd_y = torch.zeros(1, device=self.device)
        cmd_yaw = torch.zeros(1, device=self.device)
        
        # Get IK for neutral stance
        joint_angles = self.compute_all_legs_ik(phases, cmd_x, cmd_y, cmd_yaw)
        
        return joint_angles[0]  # Return single set of angles