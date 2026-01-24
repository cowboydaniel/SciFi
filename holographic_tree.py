#!/usr/bin/env python3
"""
Holographic Tree - GPU Accelerated with OpenGL Instancing
Uses pyglet + moderngl for instanced branches and shader-based glow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from array import array
import math
import random
import time
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import moderngl
import pyglet
import numpy as np

def check_gl_error(gl, location: str) -> bool:
    """Check for OpenGL errors and print them if found."""
    error = gl.glGetError()
    if error != gl.GL_NO_ERROR:
        error_str = {
            gl.GL_INVALID_ENUM: "GL_INVALID_ENUM",
            gl.GL_INVALID_VALUE: "GL_INVALID_VALUE",
            gl.GL_INVALID_OPERATION: "GL_INVALID_OPERATION",
            gl.GL_OUT_OF_MEMORY: "GL_OUT_OF_MEMORY",
        }.get(error, f"Unknown error ({error})")
        print(f"OpenGL error at {location}: {error_str}")
        return False
    return True

def print_shader_compile_log(shader, shader_type: str):
    """Print shader compilation log if there are any messages."""
    if shader is None:
        print(f"Error: {shader_type} shader is None")
        return
    
    log = shader.info_log
    if log and log.strip():
        print(f"{shader_type} shader log:")
        print(log)

def print_program_link_log(program):
    """Print program link log if there are any messages."""
    if program is None:
        print("Error: Program is None")
        return
    
    log = program.info_log
    if log and log.strip():
        print("Program link log:")
        print(log)


# ============================================================================
# PROCEDURAL TEXTURE GENERATION MODULE
# ============================================================================

def _fbm_noise(x, y, octaves=4, persistence=0.5, scale=1.0, seed=0):
    """Fast multi-octave value noise using numpy."""
    rng = np.random.RandomState(seed)
    total = np.zeros_like(x, dtype=np.float32)
    amplitude = 1.0
    frequency = scale
    max_value = 0.0

    for octave in range(octaves):
        # Simple hash-based value noise
        xi = (x * frequency).astype(np.int32)
        yi = (y * frequency).astype(np.int32)

        # Hash function using int64 for intermediate calculations to prevent overflow
        xi_64 = xi.astype(np.int64)
        yi_64 = yi.astype(np.int64)

        h = (xi_64 * 374761393 + yi_64 * 668265263 + (seed + octave) * 1274126177)
        h = h & 0x7FFFFFFF  # Keep in positive int32 range
        h = ((h ^ (h >> 13)) * 1274126177) & 0x7FFFFFFF
        noise = ((h & 0xFFFFFF).astype(np.float32)) / 0xFFFFFF

        total += noise * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= 2.0

    return total / max_value if max_value > 0 else total


def _smoothstep(edge0, edge1, x):
    """Smooth interpolation between edge0 and edge1."""
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def generate_leaf_rgba(size=128, seed=0, variant=0):
    """
    Generate a procedural leaf texture with alpha cutout.

    Returns:
        numpy array of shape (size, size, 4) with uint8 values
    """
    rng = np.random.RandomState(seed + variant * 1000)

    # Create coordinate grids centered at 0.5
    y, x = np.meshgrid(
        np.linspace(0, 1, size, dtype=np.float32),
        np.linspace(0, 1, size, dtype=np.float32),
        indexing='ij'
    )

    # Center coordinates
    cx, cy = x - 0.5, y - 0.5

    # Create elliptical leaf shape (tapered)
    # Make it slightly elongated vertically
    aspect = rng.uniform(0.45, 0.55)
    taper = rng.uniform(0.15, 0.25)

    # Distance from center with tapering at ends
    dist_y = np.abs(cy) / (0.5 - taper * np.abs(cy))
    dist_x = np.abs(cx) / (aspect * (0.5 - taper * 0.3 * np.abs(cy)))
    ellipse_dist = np.sqrt(dist_x**2 + dist_y**2)

    # Base leaf shape
    base_alpha = 1.0 - _smoothstep(0.7, 1.0, ellipse_dist)

    # Add central vein
    vein_width = 0.015
    vein_mask = np.abs(cx) < vein_width
    vein_alpha = _smoothstep(vein_width, 0, np.abs(cx)) * base_alpha

    # Add jagged edges using noise
    edge_noise = _fbm_noise(x * size * 0.15, y * size * 0.15, octaves=3, persistence=0.6, seed=seed + variant)
    edge_noise = (edge_noise - 0.5) * 0.15

    # Apply edge noise to alpha
    alpha = np.clip(base_alpha + edge_noise, 0, 1)

    # Strengthen vein in alpha
    alpha = np.maximum(alpha, vein_alpha * 0.6)

    # Create green color with variation
    base_green = rng.uniform(0.35, 0.45)
    base_brightness = rng.uniform(0.7, 0.85)

    # Color variation based on position (yellowing at edges)
    edge_factor = _smoothstep(0.3, 0.9, ellipse_dist)
    yellow_tint = edge_factor * rng.uniform(0.15, 0.25)

    # Add subtle color noise
    color_noise = _fbm_noise(x * size * 0.08, y * size * 0.08, octaves=2, persistence=0.5, seed=seed + variant + 100)
    color_variation = (color_noise - 0.5) * 0.15

    # Build RGB channels
    r = np.clip(base_green + yellow_tint + color_variation, 0, 1) * base_brightness
    g = np.clip(base_brightness + color_variation * 0.5, 0, 1)
    b = np.clip(base_green * 0.6 + color_variation, 0, 1) * base_brightness

    # Brighten vein slightly
    vein_brighten = vein_alpha * 0.15
    r = np.minimum(r + vein_brighten, 1.0)
    g = np.minimum(g + vein_brighten, 1.0)

    # Convert to uint8
    rgba = np.stack([r, g, b, alpha], axis=-1)
    rgba = (rgba * 255).astype(np.uint8)

    return rgba


def generate_bark_albedo(size=256, seed=0):
    """
    Generate procedural bark albedo (color) texture with enhanced realism.

    Returns:
        numpy array of shape (size, size, 3) with uint8 values
    """
    rng = np.random.RandomState(seed)

    y, x = np.meshgrid(
        np.linspace(0, 1, size, dtype=np.float32),
        np.linspace(0, 1, size, dtype=np.float32),
        indexing='ij'
    )

    # Base bark noise with multiple octaves for rich detail
    bark_noise = _fbm_noise(x * size * 0.06, y * size * 0.06, octaves=6, persistence=0.6, seed=seed)

    # Fine detail layer for micro-texture
    fine_detail = _fbm_noise(x * size * 0.2, y * size * 0.2, octaves=4, persistence=0.5, seed=seed + 10)

    # Vertical grain pattern (more pronounced)
    grain = _fbm_noise(x * size * 0.4, y * size * 0.018, octaves=4, persistence=0.45, seed=seed + 50)
    grain = grain * 0.5 + 0.5

    # Stronger vertical crevices with variation
    crevice = _fbm_noise(x * size * 0.18, y * size * 0.035, octaves=5, persistence=0.55, seed=seed + 100)
    crevice = _smoothstep(0.35, 0.65, crevice)
    deep_crevice = _smoothstep(0.45, 0.55, crevice)
    darken = 1.0 - crevice * 0.5 - deep_crevice * 0.25

    # Add horizontal bark plates/scales
    plates = _fbm_noise(x * size * 0.12, y * size * 0.25, octaves=3, persistence=0.4, seed=seed + 150)
    plates = _smoothstep(0.4, 0.6, plates)
    plate_darken = 1.0 - plates * 0.15

    # Bark knots and imperfections
    knots = _fbm_noise(x * size * 0.08, y * size * 0.08, octaves=3, persistence=0.5, seed=seed + 200)
    knots = _smoothstep(0.6, 0.8, knots)

    # Combine all patterns
    brightness = bark_noise * 0.6 + fine_detail * 0.25 + knots * 0.15
    brightness = brightness * grain * darken * plate_darken
    brightness = brightness * 0.55 + 0.2  # Remap to natural bark range

    # More varied and realistic brown colors
    base_r = rng.uniform(0.38, 0.48)
    base_g = rng.uniform(0.25, 0.35)
    base_b = rng.uniform(0.16, 0.24)

    # Color variation across the texture
    color_var = _fbm_noise(x * size * 0.04, y * size * 0.04, octaves=2, persistence=0.5, seed=seed + 250)
    r_mult = 1.0 + color_var * 0.2
    g_mult = 1.0 + color_var * 0.15
    b_mult = 1.0 + color_var * 0.1

    # Apply colors with variation
    r = np.clip(brightness * base_r * r_mult * 1.15, 0, 1)
    g = np.clip(brightness * base_g * g_mult * 1.05, 0, 1)
    b = np.clip(brightness * base_b * b_mult, 0, 1)

    # Add subtle reddish-brown highlights in lighter areas
    highlight_mask = _smoothstep(0.5, 0.7, brightness)
    r = r + highlight_mask * 0.08
    g = g + highlight_mask * 0.03

    # Ensure proper range
    r = np.clip(r, 0, 1)
    g = np.clip(g, 0, 1)
    b = np.clip(b, 0, 1)

    # Convert to uint8
    rgb = np.stack([r, g, b], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)

    return rgb


def generate_bark_normal(size=256, seed=0):
    """
    Generate procedural bark normal map (tangent space) with enhanced detail.

    Returns:
        numpy array of shape (size, size, 3) with uint8 values (0-255 maps to -1 to 1)
    """
    rng = np.random.RandomState(seed)

    y, x = np.meshgrid(
        np.linspace(0, 1, size, dtype=np.float32),
        np.linspace(0, 1, size, dtype=np.float32),
        indexing='ij'
    )

    # Generate height map with enhanced detail
    height = _fbm_noise(x * size * 0.06, y * size * 0.06, octaves=6, persistence=0.6, seed=seed)

    # Fine detail layer
    fine_detail = _fbm_noise(x * size * 0.2, y * size * 0.2, octaves=4, persistence=0.5, seed=seed + 10)

    # Stronger vertical grain bumps
    grain = _fbm_noise(x * size * 0.4, y * size * 0.018, octaves=4, persistence=0.45, seed=seed + 50)

    # Deep crevices
    crevice = _fbm_noise(x * size * 0.18, y * size * 0.035, octaves=5, persistence=0.55, seed=seed + 100)
    crevice = _smoothstep(0.35, 0.65, crevice) * -0.4  # Negative for depth

    # Bark plates
    plates = _fbm_noise(x * size * 0.12, y * size * 0.25, octaves=3, persistence=0.4, seed=seed + 150)
    plates = _smoothstep(0.4, 0.6, plates) * 0.2

    # Combine all height patterns
    height = height * 0.4 + fine_detail * 0.15 + grain * 0.35 + crevice + plates * 0.1

    # Scale height for more pronounced bumps
    height = height * 0.45

    # Compute gradients (derivatives)
    dx = np.zeros_like(height)
    dy = np.zeros_like(height)

    dx[:, 1:] = height[:, 1:] - height[:, :-1]
    dy[1:, :] = height[1:, :] - height[:-1, :]

    # Scale gradients
    strength = 2.0
    dx *= strength
    dy *= strength

    # Build normal: N = normalize(-dx, -dy, 1)
    nx = -dx
    ny = -dy
    nz = np.ones_like(height)

    # Normalize
    length = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= length
    ny /= length
    nz /= length

    # Map from [-1,1] to [0,255]
    nx = (nx * 0.5 + 0.5)
    ny = (ny * 0.5 + 0.5)
    nz = (nz * 0.5 + 0.5)

    rgb = np.stack([nx, ny, nz], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)

    return rgb


def generate_bark_roughness(size=256, seed=0):
    """
    Generate procedural bark roughness texture.

    Returns:
        numpy array of shape (size, size, 1) with uint8 values
    """
    rng = np.random.RandomState(seed)

    y, x = np.meshgrid(
        np.linspace(0, 1, size, dtype=np.float32),
        np.linspace(0, 1, size, dtype=np.float32),
        indexing='ij'
    )

    # Base roughness variation
    roughness = _fbm_noise(x * size * 0.05, y * size * 0.05, octaves=4, persistence=0.5, seed=seed)

    # Higher roughness in crevices
    crevice = _fbm_noise(x * size * 0.15, y * size * 0.04, octaves=4, persistence=0.5, seed=seed + 100)
    crevice = _smoothstep(0.3, 0.7, crevice)

    # Combine: higher in crevices, lower on ridges
    roughness = roughness * 0.3 + crevice * 0.5 + 0.3
    roughness = np.clip(roughness, 0, 1)

    # Convert to uint8 grayscale
    roughness = (roughness * 255).astype(np.uint8)
    roughness = roughness[..., np.newaxis]  # Add channel dimension

    return roughness


def generate_grass_albedo(size=512, seed=0):
    """
    Generate procedural grass albedo texture with enhanced realism.

    Returns:
        numpy array of shape (size, size, 3) with uint8 values
    """
    rng = np.random.RandomState(seed)

    y, x = np.meshgrid(
        np.linspace(0, 1, size, dtype=np.float32),
        np.linspace(0, 1, size, dtype=np.float32),
        indexing='ij'
    )

    # Multi-scale grass noise with more octaves
    grass_noise = _fbm_noise(x * size * 0.12, y * size * 0.12, octaves=6, persistence=0.52, seed=seed)

    # Add finer detail for grass blade texture
    detail = _fbm_noise(x * size * 0.45, y * size * 0.45, octaves=4, persistence=0.6, seed=seed + 200)
    grass_noise = grass_noise * 0.65 + detail * 0.35

    # Very fine micro-detail
    micro_detail = _fbm_noise(x * size * 0.8, y * size * 0.8, octaves=2, persistence=0.5, seed=seed + 300)
    grass_noise = grass_noise * 0.85 + micro_detail * 0.15

    # Patchy variation (clumps of grass)
    patches = _fbm_noise(x * size * 0.05, y * size * 0.05, octaves=3, persistence=0.5, seed=seed + 100)
    patches = _smoothstep(0.3, 0.7, patches)

    # Color variation with patches
    base_brightness = (grass_noise * 0.5 + patches * 0.3) + 0.35

    # More varied grass green with richer tones
    base_r = rng.uniform(0.22, 0.32)
    base_g = rng.uniform(0.48, 0.58)
    base_b = rng.uniform(0.18, 0.28)

    # Add color variation for different grass types
    color_var = _fbm_noise(x * size * 0.08, y * size * 0.08, octaves=2, persistence=0.5, seed=seed + 150)
    r_mult = 1.0 + color_var * 0.25
    g_mult = 1.0 + color_var * 0.15
    b_mult = 1.0 + color_var * 0.2

    # Apply colors with variation
    r = np.clip(base_brightness * base_r * r_mult * 1.15, 0, 1)
    g = np.clip(base_brightness * base_g * g_mult * 1.05, 0, 1)
    b = np.clip(base_brightness * base_b * b_mult, 0, 1)

    # Add some yellowish dried grass patches
    dry_patches = _fbm_noise(x * size * 0.06, y * size * 0.06, octaves=2, persistence=0.5, seed=seed + 250)
    dry_patches = _smoothstep(0.65, 0.75, dry_patches)
    r = r + dry_patches * 0.12
    g = g + dry_patches * 0.08
    b = b - dry_patches * 0.02

    # Ensure proper range
    r = np.clip(r, 0, 1)
    g = np.clip(g, 0, 1)
    b = np.clip(b, 0, 1)

    # Convert to uint8
    rgb = np.stack([r, g, b], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)

    return rgb


def generate_grass_roughness(size=512, seed=0):
    """
    Generate procedural grass roughness texture.

    Returns:
        numpy array of shape (size, size, 1) with uint8 values
    """
    rng = np.random.RandomState(seed)

    y, x = np.meshgrid(
        np.linspace(0, 1, size, dtype=np.float32),
        np.linspace(0, 1, size, dtype=np.float32),
        indexing='ij'
    )

    # Moderate roughness with variation
    roughness = _fbm_noise(x * size * 0.1, y * size * 0.1, octaves=3, persistence=0.5, seed=seed)
    roughness = roughness * 0.3 + 0.5  # Range 0.5-0.8
    roughness = np.clip(roughness, 0, 1)

    # Convert to uint8 grayscale
    roughness = (roughness * 255).astype(np.uint8)
    roughness = roughness[..., np.newaxis]  # Add channel dimension

    return roughness


@dataclass
class FallingLeaf:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    rotation: float
    rotation_speed: float
    size: float
    wobble_phase: float
    wobble_speed: float
    alpha: float
    lifetime: float
    atlas_index: int
    uv_offset: tuple[float, float]
    uv_scale: tuple[float, float]
    color_variance: tuple[float, float, float]
    variance_amount: float


@dataclass
class Branch:
    parent_index: int
    base_angle: float
    length: float
    thickness: float
    depth: int
    z_depth: float
    stiffness: float
    phase_offset: float
    current_angle: float = 0.0
    start_x: float = 0.0
    start_y: float = 0.0
    start_z: float = 0.0
    end_x: float = 0.0
    end_y: float = 0.0
    end_z: float = 0.0
    segment_fracs: List[float] = field(default_factory=list)
    segment_jitter: List[float] = field(default_factory=list)
    segment_sway: List[float] = field(default_factory=list)
    segment_points: List[tuple] = field(default_factory=list)
    wind_dir: tuple[float, float] = (0.0, 0.0)
    variation_seed: float = 0.0
    bark_tint: tuple[float, float, float] = (1.0, 1.0, 1.0)
    bark_roughness: float = 0.6


@dataclass
class CanopyLeaf:
    branch_index: int
    offset: tuple[float, float, float]
    axis_x: tuple[float, float, float]
    axis_y: tuple[float, float, float]
    axis_z: tuple[float, float, float]
    base_color: tuple[float, float, float, float]
    atlas_index: int
    uv_offset: tuple[float, float]
    uv_scale: tuple[float, float]
    color_variance: tuple[float, float, float]
    variance_amount: float


@dataclass
class LeafInstance:
    position: tuple[float, float, float]
    axis_x: tuple[float, float, float]
    axis_y: tuple[float, float, float]
    axis_z: tuple[float, float, float]
    uv_offset: tuple[float, float]
    uv_scale: tuple[float, float]
    atlas_index: int
    color: tuple[float, float, float, float]
    color_variance: tuple[float, float, float]
    variance_amount: float


@dataclass
class Bird:
    x: float
    y: float
    vx: float
    vy: float
    state: str
    timer: float
    wing_phase: float
    wing_speed: float
    facing: int
    perch_index: int = -1
    hop_count: int = 0
    hops_before_depart: int = 2
    hop_offset: tuple[float, float] = (0.0, 0.0)
    perch_wait: float = 3.0


class WindSystem:
    def __init__(self):
        self.time = 0.0
        self.base_strength = 15.0
        self.gust_strength = 0.0
        self.gust_target = 0.0
        self.gust_timer = 0.0
        self._cache = 0.0

    def update(self, dt: float):
        self.time += dt
        self.gust_timer -= dt
        if self.gust_timer <= 0:
            self.gust_target = random.uniform(-25, 35)
            self.gust_timer = random.uniform(2.0, 5.0)
        self.gust_strength += (self.gust_target - self.gust_strength) * dt * 0.5
        self._cache = math.sin(self.time * 0.8) * 0.5 + math.sin(self.time * 0.5) * 0.4

    def get_force(self, y: float, height: float) -> float:
        hf = 0.3 + (1.0 - y / height) * 0.7
        return (self._cache * self.base_strength + self.gust_strength * hf) * hf


class HolographicTree:
    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self.time = 0.0
        self.flicker = 1.0
        self.flicker_pulse = 0.0

        self.wind = WindSystem()

        self.branches: List[Branch] = []
        self.falling_leaves: List[FallingLeaf] = []
        self.max_leaves = 18
        self.leaf_atlas_cols = 4
        self.leaf_atlas_rows = 4
        self.bird_sprite_cols = 4
        self.bird_sprite_rows = 1

        self.sorted_branches: List[Branch] = []
        self.tip_branches: List[Branch] = []
        self.tip_indices: List[int] = []
        self.canopy_leaves: List[CanopyLeaf] = []
        self.canopy_center_y = 0.0
        self.canopy_half_height = 1.0
        self.canopy_min_y = 0.0
        self.canopy_max_y = 0.0

        self.root_x = width / 2
        self.root_y = 80

        self.bird = Bird(
            x=-120,
            y=height * 0.35,
            vx=0.0,
            vy=0.0,
            state="approach",
            timer=0.0,
            wing_phase=0.0,
            wing_speed=12.0,
            facing=1,
            hops_before_depart=random.randint(2, 4),
            perch_wait=random.uniform(2.5, 4.5),
        )

        self.regenerate_tree()

    def regenerate_tree(self):
        self.branches.clear()
        self._gen_branch(-1, 90, 180, 0, 9, 0.0)
        self.sorted_branches = sorted(self.branches, key=lambda b: b.z_depth)
        self.tip_indices = [i for i, branch in enumerate(self.branches) if branch.depth >= 6]
        self.tip_branches = [self.branches[i] for i in self.tip_indices]
        self._update_branch_positions()
        self._build_canopy_leaves()
        self._reset_bird()

    def _reset_bird(self):
        self.bird.state = "approach"
        self.bird.timer = 0.0
        self.bird.wing_phase = 0.0
        self.bird.wing_speed = 12.0
        self.bird.hop_count = 0
        self.bird.hops_before_depart = random.randint(2, 4)
        self.bird.perch_wait = random.uniform(2.5, 4.5)
        self.bird.hop_offset = (0.0, 0.0)
        self._assign_perch()
        target = self._get_perch_point()
        if target:
            tx, ty, _ = target
            self.bird.x = -120 if tx > self.w * 0.5 else self.w + 120
            self.bird.y = ty - 120
            self.bird.facing = 1 if tx > self.bird.x else -1
        else:
            self.bird.x = -120
            self.bird.y = self.h * 0.35
            self.bird.facing = 1

    def _assign_perch(self):
        if not self.tip_indices:
            self.bird.perch_index = -1
            return
        self.bird.perch_index = random.choice(self.tip_indices)

    def _get_perch_point(self) -> tuple[float, float, float] | None:
        if self.bird.perch_index < 0 or self.bird.perch_index >= len(self.branches):
            return None
        branch = self.branches[self.bird.perch_index]
        return branch.end_x, branch.end_y, branch.current_angle

    def _update_bird(self, dt: float):
        target = self._get_perch_point()
        if not target:
            return
        tx, ty, angle = target
        perch_y = ty - 6
        self.bird.timer += dt

        if self.bird.state == "approach":
            self.bird.wing_speed = 12.0
            dx = tx - self.bird.x
            dy = perch_y - self.bird.y
            dist = math.hypot(dx, dy)
            if dist > 1:
                step = min(dist, 220 * dt)
                self.bird.x += dx / dist * step
                self.bird.y += dy / dist * step
                self.bird.facing = 1 if dx >= 0 else -1
            if dist < 12:
                self.bird.state = "land"
                self.bird.timer = 0.0

        elif self.bird.state == "land":
            self.bird.wing_speed = 6.0
            dx = tx - self.bird.x
            dy = perch_y - self.bird.y
            dist = math.hypot(dx, dy)
            if dist > 0.5:
                step = min(dist, 140 * dt)
                self.bird.x += dx / dist * step
                self.bird.y += dy / dist * step
            if self.bird.timer > 0.45:
                self.bird.state = "perched"
                self.bird.timer = 0.0
                self.bird.hop_offset = (0.0, 0.0)

        elif self.bird.state == "perched":
            self.bird.wing_speed = 0.0
            bob = math.sin(self.time * 4.5) * 1.2
            self.bird.x = tx
            self.bird.y = perch_y + bob
            if self.bird.timer > self.bird.perch_wait:
                if self.bird.hop_count < self.bird.hops_before_depart:
                    self.bird.state = "hop"
                    self.bird.timer = 0.0
                    self.bird.hop_count += 1
                    hop_dx = math.cos(math.radians(angle)) * -8
                    hop_dy = math.sin(math.radians(angle)) * -8
                    self.bird.hop_offset = (hop_dx, hop_dy)
                else:
                    self.bird.state = "depart"
                    self.bird.timer = 0.0
                    self.bird.vx = self.bird.facing * 240
                    self.bird.vy = -120

        elif self.bird.state == "hop":
            hop_time = 0.45
            t = min(1.0, self.bird.timer / hop_time)
            arc = math.sin(math.pi * t) * 6
            hx, hy = self.bird.hop_offset
            self.bird.x = tx + hx * t
            self.bird.y = perch_y + hy * t - arc
            if self.bird.timer >= hop_time:
                self.bird.state = "perched"
                self.bird.timer = 0.0
                self.bird.perch_wait = random.uniform(1.2, 2.2)

        elif self.bird.state == "depart":
            self.bird.wing_speed = 13.0
            self.bird.x += self.bird.vx * dt
            self.bird.y += self.bird.vy * dt
            self.bird.vy -= 30 * dt
            if self.bird.x < -200 or self.bird.x > self.w + 200 or self.bird.y < -200:
                self._reset_bird()

        if self.bird.wing_speed > 0:
            self.bird.wing_phase += self.bird.wing_speed * dt * 2 * math.pi

    def _gen_branch(self, parent: int, angle: float, length: float, depth: int, max_d: int, z: float):
        if depth >= max_d or length < 10:
            return
        stiff = 1.0 - (depth / max_d) * 0.85
        nz = max(-1, min(1, z + random.uniform(-0.1, 0.1)))

        # Improved tapering: exponential taper with depth
        # Base thickness for trunk (depth 0)
        base_thickness = 22.0
        thickness = base_thickness * (0.68 ** depth)
        # Clamp minimum thickness for visibility
        thickness = max(1.5, thickness)

        b = Branch(parent, angle, length, thickness,
                   depth, nz, stiff, depth * 0.3 + random.random() * 0.5)
        b.variation_seed = random.random() * 1000.0
        rng = random.Random(b.variation_seed)
        b.bark_tint = (
            rng.uniform(0.75, 1.05),
            rng.uniform(0.7, 1.0),
            rng.uniform(0.65, 0.95),
        )
        b.bark_roughness = rng.uniform(0.35, 0.75)
        wind_angle = rng.uniform(0.0, math.tau)
        b.wind_dir = (math.cos(wind_angle), math.sin(wind_angle))
        point_count = random.randint(3, 6)
        segment_count = max(2, point_count - 1)
        weights = [random.uniform(0.6, 1.4) for _ in range(segment_count)]
        total = sum(weights)
        b.segment_fracs = [w / total for w in weights]
        b.segment_jitter = [random.uniform(-8, 8) for _ in range(segment_count)]
        b.segment_sway = [random.uniform(0.4, 1.0) for _ in range(segment_count)]
        idx = len(self.branches)
        self.branches.append(b)

        # Oak-like branching: wider spread and slight horizontal curve for main boughs
        spread = 24 + depth * 2
        ratio = 0.72

        # Add slight curve to main branches (depth < 3) to spread horizontally
        branch_curve = 0.0
        if depth < 3:
            # Main boughs curve outward more for oak silhouette
            branch_curve = random.uniform(3, 8) * (1 if random.random() > 0.5 else -1)

        self._gen_branch(idx, angle - spread + branch_curve, length * ratio, depth + 1, max_d, nz - 0.05)
        self._gen_branch(idx, angle + spread + branch_curve, length * ratio, depth + 1, max_d, nz + 0.05)
        if depth < 4 and random.random() > 0.5:
            self._gen_branch(idx, angle + random.uniform(-12, 12), length * 0.55, depth + 2, max_d, nz)

    @staticmethod
    def _rotate_point(x: float, y: float, angle_deg: float) -> tuple[float, float]:
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)

    def _make_leaf_style(
        self, seed: float
    ) -> tuple[int, tuple[float, float], tuple[float, float], tuple[float, float, float], float]:
        rng = random.Random(seed)
        atlas_count = max(1, self.leaf_atlas_cols * self.leaf_atlas_rows)
        atlas_index = rng.randrange(atlas_count)
        uv_scale = (rng.uniform(0.72, 0.96), rng.uniform(0.72, 0.96))
        uv_offset = (
            rng.uniform(0.0, max(0.0, 1.0 - uv_scale[0])),
            rng.uniform(0.0, max(0.0, 1.0 - uv_scale[1])),
        )
        color_variance = (
            rng.uniform(-0.24, 0.28),
            rng.uniform(-0.18, 0.22),
            rng.uniform(-0.26, 0.24),
        )
        variance_amount = rng.uniform(0.35, 0.85)
        return atlas_index, uv_offset, uv_scale, color_variance, variance_amount

    @staticmethod
    def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @staticmethod
    def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    @staticmethod
    def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
        mag = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        if mag <= 0.00001:
            return (0.0, 0.0, 0.0)
        return (v[0] / mag, v[1] / mag, v[2] / mag)

    @staticmethod
    def _rotate_around_axis(
        v: tuple[float, float, float],
        axis: tuple[float, float, float],
        angle: float,
    ) -> tuple[float, float, float]:
        ax = HolographicTree._normalize(axis)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        dot = v[0] * ax[0] + v[1] * ax[1] + v[2] * ax[2]
        cross = HolographicTree._cross(ax, v)
        return (
            v[0] * cos_a + cross[0] * sin_a + ax[0] * dot * (1 - cos_a),
            v[1] * cos_a + cross[1] * sin_a + ax[1] * dot * (1 - cos_a),
            v[2] * cos_a + cross[2] * sin_a + ax[2] * dot * (1 - cos_a),
        )

    @staticmethod
    def _build_transform(
        position: tuple[float, float, float],
        axis_x: tuple[float, float, float],
        axis_y: tuple[float, float, float],
        axis_z: tuple[float, float, float],
        scale_x: float,
        scale_y: float,
        scale_z: float,
    ) -> List[float]:
        return [
            axis_x[0] * scale_x, axis_x[1] * scale_x, axis_x[2] * scale_x, 0.0,
            axis_y[0] * scale_y, axis_y[1] * scale_y, axis_y[2] * scale_y, 0.0,
            axis_z[0] * scale_z, axis_z[1] * scale_z, axis_z[2] * scale_z, 0.0,
            position[0], position[1], position[2], 1.0,
        ]

    def _update_branch_positions(self):
        for b in self.branches:
            if b.parent_index < 0:
                b.start_x, b.start_y, b.start_z = self.root_x, self.root_y, 0.0
            else:
                p = self.branches[b.parent_index]
                b.start_x, b.start_y, b.start_z = p.end_x, p.end_y, p.end_z

            flex = 1.0 - b.stiffness
            wave = math.sin(self.time * 2.5 - b.phase_offset) * 0.3
            scale = 1.0 + b.z_depth * 0.15
            length = b.length * scale
            base_angle = b.base_angle + math.sin(self.time * 1.8 + b.phase_offset * 2) * 2 * flex

            points = [(b.start_x, b.start_y, b.start_z)]
            x, y, z = b.start_x, b.start_y, b.start_z
            current_angle = base_angle
            segment_fracs = b.segment_fracs or [1.0]
            segment_jitter = b.segment_jitter or [0.0] * len(segment_fracs)
            segment_sway = b.segment_sway or [1.0] * len(segment_fracs)
            for i, frac in enumerate(segment_fracs):
                seg_len = length * frac
                local_wind = self.wind.get_force(y, self.h)
                sway = local_wind * flex * segment_sway[i] * (1 + wave)
                wobble = math.sin(self.time * 2.5 - b.phase_offset + i * 0.5) * 1.2
                current_angle = base_angle + segment_jitter[i] + sway + wobble
                rad = math.radians(current_angle)
                wind_dx, wind_dz = b.wind_dir
                lateral = math.sin(math.radians(sway)) * seg_len * 0.6
                x += seg_len * math.cos(rad) + lateral * wind_dx
                y += seg_len * math.sin(rad)
                z += lateral * wind_dz
                points.append((x, y, z))

            if len(points) >= 2:
                p0, p1 = points[-2], points[-1]
                b.current_angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
            else:
                b.current_angle = current_angle
            b.segment_points = points
            b.end_x, b.end_y, b.end_z = points[-1]

    def _build_canopy_leaves(self):
        """
        Build dense oak-like canopy with 12K-25K tiny leaves in clusters:
        - Ellipsoidal canopy shell relative to trunk height
        - Clusters biased to outer shell
        - Drooping edges for realistic oak shape
        - Negative space (random gaps)
        - 40-110 tiny leaf cards per cluster
        """
        self.canopy_leaves = []
        self.canopy_min_y = float("inf")
        self.canopy_max_y = float("-inf")

        # No ellipsoid needed - spawn clusters DIRECTLY at branch tips!
        # This ensures leaves are ON branches, not floating in space
        if not self.tip_indices:
            print("WARNING: No tip branches found, cannot generate canopy")
            return

        print(f"Spawning leaf clusters on {len(self.tip_indices)} branch tips")

        # ONE cluster per branch tip to stay in 12K-25K leaf range
        # 826 tips × 1 cluster × ~22 leaves = ~18,000 leaves
        cluster_count = 0
        for tip_idx in self.tip_indices:
            branch = self.branches[tip_idx]

            # Skip ~8% of tips randomly for negative space
            if random.random() < 0.08:
                continue

            cluster_count += 1

            # Base position at branch tip
            base_x = branch.end_x
            base_y = branch.end_y
            base_z = branch.end_z

            # Small random offset from exact tip position
            offset_range = 10.0  # Small offset in pixels
            cluster_x = base_x + random.uniform(-offset_range, offset_range)
            cluster_y = base_y + random.uniform(-offset_range, offset_range)
            cluster_z = base_z + random.uniform(-offset_range * 0.7, offset_range * 0.7)

            zf = (branch.z_depth + 1) / 2

            # Green color with subtle variation
            base_color = (
                0.35 + zf * 0.10,  # R: subtle warmth
                0.68 + zf * 0.15,  # G: dominant green
                0.28 + zf * 0.10,  # B: less blue
                1.0                # A: opaque (alpha cutout handles transparency)
            )

            # Cluster radius for spreading individual leaves
            cluster_radius = 12.0  # Slightly larger cluster to fill gaps

            # Increase leaves per cluster for fuller canopy
            leaves_per_cluster = random.randint(40, 110)
            if branch.depth >= 7:
                leaves_per_cluster = int(leaves_per_cluster * 1.35)
            for card_idx in range(leaves_per_cluster):
                atlas_index, uv_offset, uv_scale, color_variance, variance_amount = (
                    self._make_leaf_style(branch.variation_seed + cluster_count * 3.17 + card_idx * 1.23)
                )

                # Random offset within cluster sphere
                leaf_theta = random.uniform(0.0, math.tau)
                leaf_phi = math.acos(random.uniform(-1.0, 1.0))
                leaf_r = random.uniform(0.0, cluster_radius)
                leaf_offset_x = leaf_r * math.sin(leaf_phi) * math.cos(leaf_theta)
                leaf_offset_y = leaf_r * math.sin(leaf_phi) * math.sin(leaf_theta)
                leaf_offset_z = leaf_r * math.cos(leaf_phi)

                card_offset_x = cluster_x - branch.end_x + leaf_offset_x
                card_offset_y = cluster_y - branch.end_y + leaf_offset_y
                card_offset_z = cluster_z - branch.end_z + leaf_offset_z

                # Random orientation: yaw 0..2pi, pitch/roll +-25 degrees
                yaw = random.uniform(0.0, math.tau)
                pitch = random.uniform(-0.436, 0.436)  # +-25 degrees in radians
                roll = random.uniform(-0.436, 0.436)

                # Build rotation axes
                right = (math.cos(yaw), math.sin(yaw), 0.0)
                up = (-math.sin(yaw), math.cos(yaw), 0.0)
                normal = (0.0, 0.0, 1.0)

                # Apply pitch rotation
                up = self._rotate_around_axis(up, right, pitch)
                normal = self._rotate_around_axis(normal, right, pitch)

                # Apply roll rotation (rotate around forward axis)
                forward = self._normalize(self._cross(right, up))
                up = self._rotate_around_axis(up, forward, roll)
                normal = self._rotate_around_axis(normal, forward, roll)

                # TINY leaf size (critical for realism)
                # Small leaves: 2-5 pixels each
                leaf_size = random.uniform(2.0, 5.0)

                # Per-leaf color jitter (+-10% hue/value variation)
                color_jitter_r = random.uniform(-0.10, 0.10)
                color_jitter_g = random.uniform(-0.10, 0.10)
                color_jitter_b = random.uniform(-0.10, 0.10)

                jittered_color = (
                    max(0.0, min(1.0, base_color[0] + color_jitter_r)),
                    max(0.0, min(1.0, base_color[1] + color_jitter_g)),
                    max(0.0, min(1.0, base_color[2] + color_jitter_b)),
                    1.0
                )

                axis_x = (right[0] * leaf_size, right[1] * leaf_size, right[2] * leaf_size)
                axis_y = (up[0] * leaf_size, up[1] * leaf_size, up[2] * leaf_size)
                axis_z = self._normalize(normal)

                leaf_y = branch.end_y + card_offset_y
                self.canopy_min_y = min(self.canopy_min_y, leaf_y)
                self.canopy_max_y = max(self.canopy_max_y, leaf_y)

                def append_leaf(ax_x, ax_y, ax_z):
                    self.canopy_leaves.append(CanopyLeaf(
                        branch_index=tip_idx,
                        offset=(card_offset_x, card_offset_y, card_offset_z),
                        axis_x=ax_x,
                        axis_y=ax_y,
                        axis_z=ax_z,
                        base_color=jittered_color,
                        atlas_index=atlas_index,
                        uv_offset=uv_offset,
                        uv_scale=uv_scale,
                        color_variance=color_variance,
                        variance_amount=variance_amount,
                    ))

                append_leaf(axis_x, axis_y, axis_z)

                # Cross card: same position/scale/UV, yaw rotated +90 degrees with slight tilt
                yaw_cross = yaw + (math.pi * 0.5)
                right_cross = (math.cos(yaw_cross), math.sin(yaw_cross), 0.0)
                up_cross = (-math.sin(yaw_cross), math.cos(yaw_cross), 0.0)
                normal_cross = (0.0, 0.0, 1.0)

                up_cross = self._rotate_around_axis(up_cross, right_cross, pitch)
                normal_cross = self._rotate_around_axis(normal_cross, right_cross, pitch)

                forward_cross = self._normalize(self._cross(right_cross, up_cross))
                up_cross = self._rotate_around_axis(up_cross, forward_cross, roll)
                normal_cross = self._rotate_around_axis(normal_cross, forward_cross, roll)

                tilt = random.uniform(-0.1745, 0.1745)
                up_cross = self._rotate_around_axis(up_cross, right_cross, tilt)
                normal_cross = self._rotate_around_axis(normal_cross, right_cross, tilt)

                axis_x_cross = (
                    right_cross[0] * leaf_size,
                    right_cross[1] * leaf_size,
                    right_cross[2] * leaf_size,
                )
                axis_y_cross = (
                    up_cross[0] * leaf_size,
                    up_cross[1] * leaf_size,
                    up_cross[2] * leaf_size,
                )
                axis_z_cross = self._normalize(normal_cross)

                append_leaf(axis_x_cross, axis_y_cross, axis_z_cross)

        # Debug output
        if self.canopy_leaves:
            self.canopy_min_y = min(self.canopy_min_y, self.canopy_max_y)
            self.canopy_max_y = max(self.canopy_max_y, self.canopy_min_y)
            self.canopy_center_y = (self.canopy_min_y + self.canopy_max_y) * 0.5
            self.canopy_half_height = max((self.canopy_max_y - self.canopy_min_y) * 0.5, 1.0)
        else:
            self.canopy_min_y = self.root_y
            self.canopy_max_y = self.root_y
            self.canopy_center_y = self.root_y
            self.canopy_half_height = 1.0
        print(f"Generated {len(self.canopy_leaves)} canopy leaves")
        print(f"Spawned {cluster_count} clusters on {len(self.tip_indices)} branch tips")
        print(f"Average {len(self.canopy_leaves) / cluster_count:.1f} leaves per cluster")

    def update(self, dt: float):
        self.time += dt
        self.wind.update(dt)
        self._update_branch_positions()

        base_flicker = 0.93 + 0.06 * math.sin(self.time * 1.1) + 0.01 * math.sin(self.time * 0.35)
        if random.random() > 0.994:
            self.flicker_pulse = max(self.flicker_pulse, random.uniform(0.08, 0.18))
        self.flicker_pulse *= math.exp(-dt * 1.6)
        self.flicker = max(0.78, min(1.06, base_flicker + self.flicker_pulse))

        if len(self.falling_leaves) < self.max_leaves and self.tip_branches and random.random() > 0.988:
            tb = random.choice(self.tip_branches)
            atlas_index, uv_offset, uv_scale, color_variance, variance_amount = self._make_leaf_style(
                tb.variation_seed + self.time + random.random()
            )
            self.falling_leaves.append(FallingLeaf(
                tb.end_x, tb.end_y, tb.end_z,
                random.uniform(-0.4, 0.4), random.uniform(-1.5, -0.5),
                random.uniform(0, 360), random.uniform(-4, 4),
                random.uniform(12, 20), random.uniform(0, 6.28),
                random.uniform(2, 4), 1.0, 0,
                atlas_index, uv_offset, uv_scale, color_variance, variance_amount,
            ))

        ground = 60
        new_l = []
        for lf in self.falling_leaves:
            lf.lifetime += 1
            lf.vx += self.wind.get_force(lf.y, self.h) * 0.003
            lf.vx *= 0.98
            lf.vy = max(lf.vy - 0.03, -2.5)
            lf.x += lf.vx + math.sin(lf.wobble_phase) * 1.5
            lf.wobble_phase += lf.wobble_speed * 0.016
            lf.y += lf.vy
            lf.rotation += lf.rotation_speed

            if lf.y <= ground:
                lf.y = ground
                lf.vy = 0
                lf.rotation_speed *= 0.92
                lf.alpha -= 0.01

            if lf.alpha > 0 and 0 < lf.x < self.w:
                new_l.append(lf)
        self.falling_leaves = new_l

        self._update_bird(dt)

    def build_branch_instances(self) -> List[float]:
        instances: List[float] = []
        f = self.flicker
        for b in self.sorted_branches:
            zf = (b.z_depth + 1) / 2
            pulse = (0.88 + 0.12 * math.sin(self.time * 3 + b.phase_offset)) * f
            # Use neutral color to let bark texture show through
            # Slight variation for depth but keep it neutral
            brightness = 0.95 + zf * 0.05
            color = (brightness * pulse, brightness * pulse, brightness * pulse, 0.7)
            th = max(1, b.thickness * (0.85 + zf * 0.3))
            points = b.segment_points or [(b.start_x, b.start_y, b.start_z), (b.end_x, b.end_y, b.end_z)]
            if len(points) < 2:
                continue
            for i in range(len(points) - 1):
                x0, y0, z0 = points[i]
                x1, y1, z1 = points[i + 1]
                dx = x1 - x0
                dy = y1 - y0
                dz = z1 - z0
                seg_len = math.sqrt(dx * dx + dy * dy + dz * dz)
                if seg_len <= 0.0001:
                    continue
                direction = self._normalize((x1 - x0, y1 - y0, z1 - z0))
                up_ref = (0.0, 0.0, 1.0)
                if abs(self._dot(direction, up_ref)) > 0.92:
                    up_ref = (0.0, 1.0, 0.0)
                right = self._normalize(self._cross(up_ref, direction))
                normal_axis = self._normalize(self._cross(direction, right))
                mid = ((x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5)
                transform = self._build_transform(
                    mid,
                    right,
                    direction,
                    normal_axis,
                    th,
                    seg_len,
                    th,
                )
                instances.extend([
                    *transform,
                    *color,
                    *b.bark_tint, b.bark_roughness,
                ])
        return instances

    def build_leaf_instances(self) -> List[LeafInstance]:
        instances: List[LeafInstance] = []
        f = self.flicker
        for leaf in self.canopy_leaves:
            if leaf.branch_index >= len(self.branches):
                continue
            branch = self.branches[leaf.branch_index]
            position = (
                branch.end_x + leaf.offset[0],
                branch.end_y + leaf.offset[1],
                branch.end_z + leaf.offset[2],
            )
            color = (
                leaf.base_color[0],
                leaf.base_color[1],
                leaf.base_color[2],
                leaf.base_color[3] * f,
            )
            instances.append(LeafInstance(
                position=position,
                axis_x=leaf.axis_x,
                axis_y=leaf.axis_y,
                axis_z=leaf.axis_z,
                uv_offset=leaf.uv_offset,
                uv_scale=leaf.uv_scale,
                atlas_index=leaf.atlas_index,
                color=color,
                color_variance=leaf.color_variance,
                variance_amount=leaf.variance_amount,
            ))

        for lf in self.falling_leaves:
            size = lf.size * (0.8 + (lf.z + 1) * 0.2)
            color = (0.6, 0.9, 1.0, lf.alpha * f)
            rot = math.radians(lf.rotation)
            right = (math.cos(rot), math.sin(rot), 0.0)
            up = (-math.sin(rot), math.cos(rot), 0.0)
            axis_x = (right[0] * size, right[1] * size, right[2] * size)
            axis_y = (up[0] * size, up[1] * size, up[2] * size)
            axis_z = (0.0, 0.0, 1.0)
            instances.append(LeafInstance(
                position=(lf.x, lf.y, lf.z),
                axis_x=axis_x,
                axis_y=axis_y,
                axis_z=axis_z,
                uv_offset=lf.uv_offset,
                uv_scale=lf.uv_scale,
                atlas_index=lf.atlas_index,
                color=color,
                color_variance=lf.color_variance,
                variance_amount=lf.variance_amount,
            ))

        return instances

    def build_bird_instances(self) -> List[float]:
        target = self._get_perch_point()
        if not target:
            return []
        bird = self.bird
        flap = math.sin(bird.wing_phase) if bird.wing_speed > 0 else 0.2
        size = 18.0
        color = (0.43, 0.92, 1.0, 0.95 * self.flicker)
        wing_color = (0.27, 0.78, 1.0, 0.85 * self.flicker)
        beak_color = (1.0, 0.82, 0.35, 0.9 * self.flicker)
        frame_count = max(1, self.bird_sprite_cols * self.bird_sprite_rows)
        if bird.wing_speed > 0:
            phase = (bird.wing_phase / (2 * math.pi)) % 1.0
            frame_index = int(phase * frame_count) % frame_count
        else:
            frame_index = 0
        return [
            bird.x, bird.y, size, float(bird.facing), flap,
            0.0, 0.0, 1.0, 1.0, float(frame_index),
            *color, *wing_color, *beak_color,
        ]


class DebugQuad:
    """Simple debug quad for testing basic rendering."""
    def __init__(self, ctx):
        self.ctx = ctx
        self.program = self.ctx.program(
            vertex_shader="""
                #version 120
                attribute vec2 in_vert;
                attribute vec2 in_uv;
                varying vec2 v_uv;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    v_uv = in_uv;
                }
            """,
            fragment_shader="""
                #version 120
                varying vec2 v_uv;
                void main() {
                    // Draw a simple gradient for visibility
                    vec2 uv = v_uv * 2.0 - 1.0;
                    float d = length(uv);
                    float a = smoothstep(1.0, 0.9, d);
                    gl_FragColor = vec4(0.2, 0.6, 1.0, 0.8) * a;
                }
            """
        )
        
        # Vertex data for a fullscreen quad
        vertices = np.array([
            # x, y, u, v
            -1, -1,  0, 0,
             1, -1,  1, 0,
            -1,  1,  0, 1,
             1,  1,  1, 1,
        ], dtype='f4')
        
        # Create a single buffer with interleaved vertex data
        vbo = self.ctx.buffer(vertices)
        
        # Create the vertex array
        self.vao = self.ctx.simple_vertex_array(
            self.program,
            vbo,
            'in_vert',
            'in_uv',
        )  
        
    def render(self):
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func(moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.vao.render(moderngl.TRIANGLE_STRIP)


class OpenGLRenderer:
    def _create_texture_from_array(self, data: np.ndarray, repeat=True, mipmaps=True) -> moderngl.Texture:
        """Create a moderngl texture from a numpy array."""
        if data.ndim == 2:
            # Grayscale
            height, width = data.shape
            components = 1
        elif data.ndim == 3:
            height, width, components = data.shape
        else:
            raise ValueError(f"Invalid array shape: {data.shape}")

        # Ensure data is contiguous and uint8
        data = np.ascontiguousarray(data, dtype=np.uint8)

        texture = self.ctx.texture((width, height), components, data.tobytes())
        texture.repeat_x = repeat
        texture.repeat_y = repeat

        if mipmaps and (width > 1 or height > 1):
            texture.build_mipmaps()
            texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        else:
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        return texture

    @staticmethod
    def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
        mag = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        if mag <= 0.00001:
            return (0.0, 0.0, 0.0)
        return (v[0] / mag, v[1] / mag, v[2] / mag)

    @staticmethod
    def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    @staticmethod
    def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    def _look_at(
        self,
        eye: tuple[float, float, float],
        target: tuple[float, float, float],
        up: tuple[float, float, float],
    ) -> tuple[float, ...]:
        forward = self._normalize((target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]))
        side = self._normalize(self._cross(forward, up))
        up_dir = self._cross(side, forward)
        return (
            side[0], up_dir[0], -forward[0], 0.0,
            side[1], up_dir[1], -forward[1], 0.0,
            side[2], up_dir[2], -forward[2], 0.0,
            -self._dot(side, eye), -self._dot(up_dir, eye), self._dot(forward, eye), 1.0,
        )

    @staticmethod
    def _perspective(fov_y: float, aspect: float, near: float, far: float) -> tuple[float, ...]:
        f = 1.0 / math.tan(fov_y / 2.0)
        nf = 1.0 / (near - far)
        return (
            f / aspect, 0.0, 0.0, 0.0,
            0.0, f, 0.0, 0.0,
            0.0, 0.0, (far + near) * nf, -1.0,
            0.0, 0.0, (2.0 * far * near) * nf, 0.0,
        )

    @staticmethod
    def _extract_frustum_planes(view_proj_matrix: tuple[float, ...]):
        """Extract 6 frustum planes from view-projection matrix.
        Each plane is (a, b, c, d) where ax + by + cz + d = 0
        Returns: [left, right, bottom, top, near, far]
        """
        m = view_proj_matrix
        planes = []

        # Left plane: m3 + m0
        planes.append((m[3] + m[0], m[7] + m[4], m[11] + m[8], m[15] + m[12]))
        # Right plane: m3 - m0
        planes.append((m[3] - m[0], m[7] - m[4], m[11] - m[8], m[15] - m[12]))
        # Bottom plane: m3 + m1
        planes.append((m[3] + m[1], m[7] + m[5], m[11] + m[9], m[15] + m[13]))
        # Top plane: m3 - m1
        planes.append((m[3] - m[1], m[7] - m[5], m[11] - m[9], m[15] - m[13]))
        # Near plane: m3 + m2
        planes.append((m[3] + m[2], m[7] + m[6], m[11] + m[10], m[15] + m[14]))
        # Far plane: m3 - m2
        planes.append((m[3] - m[2], m[7] - m[6], m[11] - m[10], m[15] - m[14]))

        # Normalize planes
        normalized_planes = []
        for a, b, c, d in planes:
            length = math.sqrt(a * a + b * b + c * c)
            if length > 0:
                normalized_planes.append((a / length, b / length, c / length, d / length))
            else:
                normalized_planes.append((a, b, c, d))

        return normalized_planes

    @staticmethod
    def _aabb_in_frustum(frustum_planes, min_x: float, min_z: float, max_x: float, max_z: float, min_y: float = 0.0, max_y: float = 25.0) -> bool:
        """Test if axis-aligned bounding box intersects frustum.
        Uses separating axis theorem - if AABB is completely outside any plane, it's culled.
        """
        for a, b, c, d in frustum_planes:
            # Find the positive vertex (vertex furthest along plane normal)
            px = max_x if a >= 0 else min_x
            py = max_y if b >= 0 else min_y
            pz = max_z if c >= 0 else min_z

            # If positive vertex is outside plane, entire AABB is outside
            if a * px + b * py + c * pz + d < 0:
                return False

        return True

    @staticmethod
    def _ortho(left: float, right: float, bottom: float, top: float, near: float, far: float) -> tuple[float, ...]:
        rl = right - left
        tb = top - bottom
        fn = far - near
        return (
            2.0 / rl, 0.0, 0.0, 0.0,
            0.0, 2.0 / tb, 0.0, 0.0,
            0.0, 0.0, -2.0 / fn, 0.0,
            -(right + left) / rl, -(top + bottom) / tb, -(far + near) / fn, 1.0,
        )

    @staticmethod
    def _mul_mat4(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
        out = [0.0] * 16
        for col in range(4):
            for row in range(4):
                out[col * 4 + row] = sum(a[k * 4 + row] * b[col * 4 + k] for k in range(4))
        return tuple(out)

    @staticmethod
    def _build_cylinder_mesh(segments: int = 18) -> List[float]:
        data: List[float] = []
        radius = 0.5
        for i in range(segments):
            angle0 = (i / segments) * math.tau
            angle1 = ((i + 1) / segments) * math.tau
            cos0, sin0 = math.cos(angle0), math.sin(angle0)
            cos1, sin1 = math.cos(angle1), math.sin(angle1)
            u0 = i / segments
            u1 = (i + 1) / segments

            def add_vertex(x: float, y: float, z: float, nx: float, nz: float, u: float, v: float):
                data.extend([
                    x, y, z,
                    nx, 0.0, nz,
                    u, v,
                    -nz, 0.0, nx,
                ])

            y0 = -0.5
            y1 = 0.5
            x0, z0 = cos0 * radius, sin0 * radius
            x1, z1 = cos1 * radius, sin1 * radius

            add_vertex(x0, y0, z0, cos0, sin0, u0, 0.0)
            add_vertex(x0, y1, z0, cos0, sin0, u0, 1.0)
            add_vertex(x1, y1, z1, cos1, sin1, u1, 1.0)

            add_vertex(x0, y0, z0, cos0, sin0, u0, 0.0)
            add_vertex(x1, y1, z1, cos1, sin1, u1, 1.0)
            add_vertex(x1, y0, z1, cos1, sin1, u1, 0.0)
        return data

    @staticmethod
    def _build_grass_blades(area_size: float = 2000.0, num_blades: int = 50000, seed: int = 42, center_x: float = 0.0) -> List[float]:
        """Generate individual grass blades with varying heights, widths, and positions.

        Each blade is a thin quad (2 triangles, 6 vertices).
        Vertex format: position (3f), normal (3f), uv (2f)
        """
        import random
        random.seed(seed)
        data: List[float] = []

        for _ in range(num_blades):
            # Random position within area (centered at center_x, 0)
            x = random.uniform(-area_size / 2, area_size / 2) + center_x
            z = random.uniform(-area_size / 2, area_size / 2)

            # Random blade properties
            height = random.uniform(8.0, 25.0)  # Varying heights
            width = random.uniform(0.8, 2.5)    # Varying widths
            rotation = random.uniform(0, math.tau)  # Random rotation around Y axis

            # Slight bend/lean
            lean_x = random.uniform(-0.15, 0.15) * height
            lean_z = random.uniform(-0.15, 0.15) * height

            # Rotation matrix for Y-axis rotation
            cos_rot = math.cos(rotation)
            sin_rot = math.sin(rotation)

            # Base vertices of blade (thin quad, oriented along local X axis)
            # Bottom left, bottom right, top left, top right
            half_width = width / 2.0

            # Local space vertices (before rotation)
            local_verts = [
                (-half_width, 0.0, 0.0),      # Bottom left
                (half_width, 0.0, 0.0),       # Bottom right
                (-half_width + lean_x / height * half_width, height, lean_z),  # Top left
                (half_width + lean_x / height * half_width, height, lean_z),   # Top right
            ]

            # Transform vertices: rotate around Y axis, then translate to position
            world_verts = []
            for lx, ly, lz in local_verts:
                # Rotate around Y axis
                wx = lx * cos_rot - lz * sin_rot
                wy = ly
                wz = lx * sin_rot + lz * cos_rot
                # Translate to world position
                wx += x
                wz += z
                world_verts.append((wx, wy, wz))

            # Calculate normal (pointing perpendicular to blade surface)
            # Normal in local space points in Z direction, rotate it
            nx = -sin_rot
            ny = 0.0
            nz = cos_rot

            # Create two triangles (6 vertices) for the blade
            # Triangle 1: bottom-left, bottom-right, top-left
            # Triangle 2: bottom-right, top-right, top-left

            # Vertex 0: bottom-left
            data.extend(world_verts[0])  # position
            data.extend([nx, ny, nz])    # normal
            data.extend([0.0, 0.0])      # uv

            # Vertex 1: bottom-right
            data.extend(world_verts[1])  # position
            data.extend([nx, ny, nz])    # normal
            data.extend([1.0, 0.0])      # uv

            # Vertex 2: top-left
            data.extend(world_verts[2])  # position
            data.extend([nx, ny, nz])    # normal
            data.extend([0.0, 1.0])      # uv

            # Triangle 2
            # Vertex 3: bottom-right
            data.extend(world_verts[1])  # position
            data.extend([nx, ny, nz])    # normal
            data.extend([1.0, 0.0])      # uv

            # Vertex 4: top-right
            data.extend(world_verts[3])  # position
            data.extend([nx, ny, nz])    # normal
            data.extend([1.0, 1.0])      # uv

            # Vertex 5: top-left
            data.extend(world_verts[2])  # position
            data.extend([nx, ny, nz])    # normal
            data.extend([0.0, 1.0])      # uv

        return data

    @staticmethod
    def _terrain_height_vectorized(x_arr: np.ndarray, z_arr: np.ndarray, seed: int = 42) -> np.ndarray:
        """Calculate terrain height at given world positions (vectorized for speed).

        Uses multiple octaves of noise for natural rolling hills:
        - Large scale: rolling hills (amplitude ~80 units)
        - Medium scale: terrain variation (amplitude ~30 units)
        - Small scale: subtle ground detail (amplitude ~5 units)

        Args:
            x_arr: numpy array of x positions
            z_arr: numpy array of z positions
            seed: random seed

        Returns:
            numpy array of heights at each position
        """
        # Large rolling hills (low frequency, high amplitude)
        hills = _fbm_noise(x_arr * 0.0008, z_arr * 0.0008, octaves=3, persistence=0.5, scale=1.0, seed=seed)
        hills = (hills - 0.5) * 160.0  # Range: -80 to +80

        # Medium terrain features
        terrain = _fbm_noise(x_arr * 0.003, z_arr * 0.003, octaves=4, persistence=0.5, scale=1.0, seed=seed + 100)
        terrain = (terrain - 0.5) * 60.0  # Range: -30 to +30

        # Small ground detail
        detail = _fbm_noise(x_arr * 0.02, z_arr * 0.02, octaves=3, persistence=0.4, scale=1.0, seed=seed + 200)
        detail = (detail - 0.5) * 10.0  # Range: -5 to +5

        return hills + terrain + detail

    @staticmethod
    def _should_have_grass_vectorized(x_arr: np.ndarray, z_arr: np.ndarray, seed: int = 42) -> np.ndarray:
        """Determine grass density at locations (vectorized).

        Creates RDR2-style dirt patches, paths, and natural grass variation.

        Returns:
            numpy array of grass density values (0.0 = no grass/dirt, 1.0 = full grass)
        """
        # Large dirt patches (low frequency)
        dirt_patches = _fbm_noise(x_arr * 0.002, z_arr * 0.002, octaves=3, persistence=0.6, scale=1.0, seed=seed + 300)
        # Bias toward grass (80% grass coverage overall)
        dirt_patches = (dirt_patches - 0.2) * 1.5
        dirt_patches = np.clip(dirt_patches, 0.0, 1.0)

        # Medium variation (grass density variation)
        variation = _fbm_noise(x_arr * 0.008, z_arr * 0.008, octaves=4, persistence=0.5, scale=1.0, seed=seed + 400)

        # Combine: dirt patches reduce grass, variation modulates density
        grass_density = dirt_patches * (0.3 + 0.7 * variation)

        return grass_density

    @staticmethod
    def _terrain_height(x: float, z: float, seed: int = 42) -> float:
        """Calculate terrain height at given world position (single value wrapper)."""
        x_arr = np.array([x], dtype=np.float32)
        z_arr = np.array([z], dtype=np.float32)
        return float(OpenGLRenderer._terrain_height_vectorized(x_arr, z_arr, seed)[0])

    @staticmethod
    def _terrain_normal(x: float, z: float, seed: int = 42, sample_dist: float = 2.0):
        """Calculate terrain normal at given position using finite differences."""
        # Sample height at center and neighbors
        h_c = OpenGLRenderer._terrain_height(x, z, seed)
        h_r = OpenGLRenderer._terrain_height(x + sample_dist, z, seed)
        h_l = OpenGLRenderer._terrain_height(x - sample_dist, z, seed)
        h_u = OpenGLRenderer._terrain_height(x, z + sample_dist, seed)
        h_d = OpenGLRenderer._terrain_height(x, z - sample_dist, seed)

        # Calculate tangent vectors
        dx = (h_r - h_l) / (2.0 * sample_dist)
        dz = (h_u - h_d) / (2.0 * sample_dist)

        # Normal is cross product of tangents
        # Tangent along X: (1, dx, 0)
        # Tangent along Z: (0, dz, 1)
        # Normal: cross product = (-dx, 1, -dz)
        nx = -dx
        ny = 1.0
        nz = -dz

        # Normalize
        length = math.sqrt(nx*nx + ny*ny + nz*nz)
        if length > 0:
            return (nx/length, ny/length, nz/length)
        return (0.0, 1.0, 0.0)


    @staticmethod
    def _build_grass_blade_mesh() -> List[float]:
        """Create a single grass blade mesh (base geometry for instancing).

        Returns a thin quad (2 triangles, 6 vertices).
        Vertex format: position (3f), normal (3f), uv (2f)
        Position is in local space: bottom at y=0, top at y=1, centered at x=0
        """
        data: List[float] = []

        # Base blade properties (will be scaled by instance data)
        half_width = 0.5
        height = 1.0

        # Local space vertices (Y-up, centered at origin)
        # Bottom left, bottom right, top left, top right
        local_verts = [
            (-half_width, 0.0, 0.0),      # Bottom left
            (half_width, 0.0, 0.0),       # Bottom right
            (-half_width, height, 0.0),   # Top left
            (half_width, height, 0.0),    # Top right
        ]

        # Normal points in +Z direction (will be rotated by instance)
        nx, ny, nz = 0.0, 0.0, 1.0

        # Triangle 1: bottom-left, bottom-right, top-left
        data.extend(local_verts[0]); data.extend([nx, ny, nz]); data.extend([0.0, 0.0])
        data.extend(local_verts[1]); data.extend([nx, ny, nz]); data.extend([1.0, 0.0])
        data.extend(local_verts[2]); data.extend([nx, ny, nz]); data.extend([0.0, 1.0])

        # Triangle 2: bottom-right, top-right, top-left
        data.extend(local_verts[1]); data.extend([nx, ny, nz]); data.extend([1.0, 0.0])
        data.extend(local_verts[3]); data.extend([nx, ny, nz]); data.extend([1.0, 1.0])
        data.extend(local_verts[2]); data.extend([nx, ny, nz]); data.extend([0.0, 1.0])

        return data

    @staticmethod
    def _build_grass_instance_data_patches(area_size: float = 2000.0, num_blades: int = 500000, seed: int = 42, center_x: float = 0.0, grid_size: int = 20):
        """Generate instance data for grass blades with RDR2-style terrain following (VECTORIZED).

        Returns a list of patches, where each patch contains:
        - bounds: (min_x, min_z, max_x, max_z)
        - instance_data: List[float] with format position_xyz (3f), rotation (1f), height (1f), width (1f), lean_xz (2f)
        - instance_count: number of instances in this patch
        """
        np.random.seed(seed)

        print(f"Generating terrain-following grass ({num_blades} blades)...")

        # Calculate patch dimensions
        patch_size = area_size / grid_size
        min_x = -area_size / 2 + center_x
        min_z = -area_size / 2

        # Create patches with bounds
        patches = []
        for row in range(grid_size):
            for col in range(grid_size):
                patch_min_x = min_x + col * patch_size
                patch_min_z = min_z + row * patch_size
                patch_max_x = patch_min_x + patch_size
                patch_max_z = patch_min_z + patch_size
                patches.append({
                    'bounds': (patch_min_x, patch_min_z, patch_max_x, patch_max_z),
                    'instance_data': [],
                })

        # VECTORIZED: Generate ALL grass positions at once per patch
        blades_per_patch = num_blades // (grid_size * grid_size)
        attempts_per_patch = int(blades_per_patch * 1.5)  # Over-sample for rejection

        for idx, patch in enumerate(patches):
            min_x, min_z, max_x, max_z = patch['bounds']

            # Generate candidate positions (vectorized)
            x_positions = np.random.uniform(min_x, max_x, attempts_per_patch).astype(np.float32)
            z_positions = np.random.uniform(min_z, max_z, attempts_per_patch).astype(np.float32)

            # Evaluate grass density at all positions (vectorized)
            grass_densities = OpenGLRenderer._should_have_grass_vectorized(x_positions, z_positions, seed)
            random_samples = np.random.uniform(0, 1, attempts_per_patch).astype(np.float32)

            # Filter positions based on grass density (rejection sampling)
            accepted_mask = random_samples <= grass_densities
            x_final = x_positions[accepted_mask][:blades_per_patch]
            z_final = z_positions[accepted_mask][:blades_per_patch]

            # Get terrain heights for accepted positions (vectorized)
            y_final = OpenGLRenderer._terrain_height_vectorized(x_final, z_final, seed)

            n = len(x_final)
            if n == 0:
                patch['instance_count'] = 0
                continue

            # Generate random properties (vectorized)
            height_factors = 1.0 - np.minimum(np.maximum(y_final, 0) / 150.0, 0.3)
            heights = np.random.uniform(8.0, 25.0, n).astype(np.float32) * height_factors
            widths = np.random.uniform(0.8, 2.5, n).astype(np.float32)
            rotations = np.random.uniform(0, math.tau, n).astype(np.float32)
            lean_x = np.random.uniform(-0.15, 0.15, n).astype(np.float32)
            lean_z = np.random.uniform(-0.15, 0.15, n).astype(np.float32)

            # Build instance data (interleave arrays)
            instance_data = np.empty(n * 8, dtype=np.float32)
            instance_data[0::8] = x_final
            instance_data[1::8] = y_final
            instance_data[2::8] = z_final
            instance_data[3::8] = rotations
            instance_data[4::8] = heights
            instance_data[5::8] = widths
            instance_data[6::8] = lean_x
            instance_data[7::8] = lean_z

            patch['instance_data'] = instance_data.tolist()
            patch['instance_count'] = n

            if (idx + 1) % 100 == 0:
                print(f"  Generated {idx + 1}/{len(patches)} patches...")

        print(f"Terrain grass generation complete!")
        return patches

    def __init__(self, window, tree):
        print("Initializing OpenGL renderer...")
        self.window = window
        self.tree = tree
        self.time = 0.0
        self.initialized = False
        self.debug_quad = None

        # Initialize camera with default values
        self.camera_position = (tree.w * 0.5, tree.h * 0.5, 1600.0)  # Positioned to see the whole tree
        self.camera_target = (tree.w * 0.5, tree.h * 0.48, 0.0)      # Slightly below center of tree
        self.camera_up = (0.0, 1.0, 0.0)  # Y is up

        # Camera settings
        self.fov = 40.0  # Field of view in degrees

        # Lighting settings
        self.light_direction = (0.5, -1.0, 0.5)  # Direction of the light source
        self.light_color = (1.0, 0.9, 0.8)      # Warm white light
        self.ambient_light = (0.2, 0.2, 0.2)    # Ambient light level

        # Rendering settings
        self.use_alpha_to_coverage = False
        self.alpha_to_coverage_flag = getattr(moderngl, 'SAMPLE_ALPHA_TO_COVERAGE', None)
        self.debug_view_mode = 0
        self.leaf_count = 0

        # Fog settings
        self.fog_color = (0.55, 0.68, 0.82)
        self.fog_density = 0.0001

        # Rendering state
        self.branch_instance_vbo = None
        self.leaf_instance_vbo = None
        self.branch_vao = None
        self.leaf_vao = None
        self.branch_vbo = None
        self.leaf_vbo = None
        self.branch_program = None
        self.leaf_program = None
        self.branch_shadow_program = None
        self.leaf_shadow_program = None
        self.branch_shadow_vao = None
        self.leaf_shadow_vao = None
        self.sky_program = None
        self.sky_vao = None
        self.grass_albedo = None
        self.ground_program = None
        self.grass_patches: List[Dict[str, Any]] = []
        self.bark_albedo = None
        self.bark_normal = None
        self.bark_roughness = None
        self.has_bark_normal = False
        self.has_bark_roughness = False
        self.leaf_atlas = None
        self.bird_instance_vbo = None
        self.bird_program = None
        self.bird_sprite = None
        self.bird_vao = None
        self.ui_line_instance_vbo = None
        self.ui_line_program = None
        self.ui_line_vao = None
        self.textures = {}
        self.initialized = False

        # Texture buffer objects for instanced rendering
        self.branch_position_tbo = None
        self.branch_direction_tbo = None
        self.leaf_position_tbo = None
        self.leaf_scale_tbo = None
        self.branch_position_tex = None

        # Rendering limits
        self.max_branches = 100000  # Maximum number of branches to render
        self.max_leaves = 200000    # Maximum number of leaves to render

        # Shadow mapping
        self.shadow_size = 2048
        self.shadow_fbo = None
        self.shadow_map = None
        self.shadow_bias = 0.001
        self.shadow_light_pos = (0, 0, 0)

        # Debug and state tracking
        self.draw_calls = 0
        self.triangles_drawn = 0
        self.last_frame_time = 0
        self.frame_times = []
        self.avg_frame_time = 0

        try:
            # Initialize ModernGL context
            self.ctx = moderngl.create_context()
            if not self.ctx:
                raise RuntimeError("Failed to create OpenGL context")
            
            # Print OpenGL info
            print("\n--- OpenGL Info ---")
            print(f"Version: {self.ctx.version_code}")
            
            # Safely get OpenGL info with fallbacks
            vendor = self.ctx.info.get('GL_VENDOR', 'Unknown')
            renderer = self.ctx.info.get('GL_RENDERER', 'Unknown')
            glsl_version = self.ctx.info.get('GL_SHADING_LANGUAGE_VERSION', 'Unknown')
            
            print(f"Vendor: {vendor}")
            print(f"Renderer: {renderer}")
            print(f"GLSL Version: {glsl_version}")
            
            # Print all available OpenGL info keys for debugging
            print("\nAvailable OpenGL Info:")
            for key in sorted(self.ctx.info.keys()):
                print(f"- {key}")
            print("------------------\n")
            
            # Initialize rendering resources
            try:
                # Create texture buffers for instanced rendering
                self.branch_position_tbo = self.ctx.buffer(reserve=self.max_branches * 16)  # vec4 per instance
                self.branch_direction_tbo = self.ctx.buffer(reserve=self.max_branches * 16)  # vec4 per instance
                self.leaf_position_tbo = self.ctx.buffer(reserve=self.max_leaves * 16)      # vec4 per instance
                self.leaf_scale_tbo = self.ctx.buffer(reserve=self.max_leaves * 4)         # float per instance
                
                self._init_branch_rendering()
                self._init_leaf_rendering()
                self._init_sky_rendering()
                self._init_ground_rendering()
                self._init_bark_textures()
                self._init_leaf_atlas()
                self._init_bird_rendering()
                self._init_ui_rendering()
                self._init_shadow_rendering()
                self.initialized = True
                print("OpenGL renderer initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize rendering: {e}")
                traceback.print_exc()
                print("Trying to continue with limited functionality...")
                self.initialized = False
            
        except Exception as e:
            print(f"Error initializing OpenGL renderer: {e}")
            traceback.print_exc()
            raise

    def _look_at(self, eye, target, up):
        """Create a view matrix looking from eye to target."""
        forward = np.array(target) - np.array(eye)
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        new_up = np.cross(right, forward)
        
        # Create rotation matrix
        rot = np.eye(4)
        rot[0, :3] = right
        rot[1, :3] = new_up
        rot[2, :3] = -forward
        
        # Create translation matrix
        trans = np.eye(4)
        trans[:3, 3] = -np.array(eye)
        
        return (rot @ trans).T.flatten().tolist()
    
    def _perspective(self, fov, aspect, near, far):
        """Create a perspective projection matrix."""
        f = 1.0 / math.tan(math.radians(fov) / 2.0)
        range_inv = 1.0 / (near - far)
        
        return [
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (near + far) * range_inv, -1,
            0, 0, near * far * range_inv * 2, 0
        ]
    
    def set_camera(self, position, target, up=None):
        """Set the camera position, target and up vector."""
        self.camera_position = position
        self.camera_target = target
        if up is not None:
            self.camera_up = up
    
    def _init_branch_rendering(self):
        """Initialize shaders and buffers for branch rendering."""
        try:
            # Simple branch shader
            self.branch_program = self.ctx.program(
                vertex_shader="""
                    #version 120
                    attribute vec3 in_position;
                    uniform mat4 u_mvp;
                    varying vec3 v_color;
                    
                    void main() {
                        gl_Position = u_mvp * vec4(in_position, 1.0);
                        v_color = vec3(0.5, 0.3, 0.1);  // Brown color for branches
                    }
                """,
                fragment_shader="""
                    #version 120
                    varying vec3 v_color;
                    
                    void main() {
                        gl_FragColor = vec4(v_color, 1.0);
                    }
                """
            )
            
            # Create a simple branch segment (a line)
            branch_vertices = np.array([
                0.0, 0.0, 0.0,  # Start point
                0.0, 1.0, 0.0   # End point (upwards)
            ], dtype='f4')
            
            # Create VBO for branch vertices
            branch_vbo = self.ctx.buffer(branch_vertices.tobytes())
            self.branch_vbo = branch_vbo
            
            # Create VAO for branches
            self.branch_vao = self.ctx.vertex_array(
                self.branch_program,
                [
                    (branch_vbo, '3f', 'in_position')
                ]
            )
            
            # Create instance VBO (will be updated each frame)
            self.branch_instance_vbo = self.ctx.buffer(reserve=1024*1024)  # 1MB initial size
            
        except Exception as e:
            print(f"Error initializing branch rendering: {e}")
            traceback.print_exc()
            raise
    
    def _init_leaf_rendering(self):
        """Initialize shaders and buffers for leaf rendering."""
        try:
            # Simple leaf shader
            self.leaf_program = self.ctx.program(
                vertex_shader="""
                    #version 120
                    attribute vec2 in_position;
                    attribute vec3 in_instance_pos;
                    attribute float in_instance_scale;
                    uniform mat4 u_mvp;
                    varying vec2 v_uv;
                    
                    void main() {
                        vec4 pos = u_mvp * vec4(in_instance_pos, 1.0);
                        pos.xy += in_position.xy * in_instance_scale * 0.1;
                        gl_Position = pos;
                        v_uv = in_position * 0.5 + 0.5;
                    }
                """,
                fragment_shader="""
                    #version 120
                    varying vec2 v_uv;
                    
                    void main() {
                        float r = length(v_uv - 0.5);
                        float alpha = smoothstep(0.5, 0.45, r);
                        if (alpha < 0.1) discard;
                        gl_FragColor = vec4(0.2, 0.8, 0.3, alpha);  // Green color for leaves
                    }
                """
            )
            
            # Create a simple quad for leaves
            leaf_vertices = np.array([
                -1, -1,  # bottom-left
                 1, -1,  # bottom-right
                -1,  1,  # top-left
                 1,  1,  # top-right
            ], dtype='f4')
            
            # Create VBO for leaf vertices
            leaf_vbo = self.ctx.buffer(leaf_vertices.tobytes())
            self.leaf_vbo = leaf_vbo
            
            # Create VAO for leaves
            self.leaf_vao = self.ctx.vertex_array(
                self.leaf_program,
                [
                    (leaf_vbo, '2f', 'in_position')
                ]
            )
            
            # Create instance VBO (will be updated each frame)
            self.leaf_instance_vbo = self.ctx.buffer(reserve=1024*1024)  # 1MB initial size
            
        except Exception as e:
            print(f"Error initializing leaf rendering: {e}")
            traceback.print_exc()
            raise

    def _init_sky_rendering(self):
        """Initialize sky rendering resources."""
        self.sky_program = None
        self.sky_vao = None
        try:
            self.sky_program = self.ctx.program(
                vertex_shader="""
                    #version 330
                    in vec2 in_position;
                    out vec2 v_uv;

                    void main() {
                        v_uv = in_position * 0.5 + 0.5;
                        gl_Position = vec4(in_position, 0.0, 1.0);
                    }
                """,
                fragment_shader="""
                    #version 330
                    uniform vec3 u_sky_top;
                    uniform vec3 u_sky_bottom;
                    in vec2 v_uv;
                    out vec4 f_color;

                    void main() {
                        float t = clamp(v_uv.y, 0.0, 1.0);
                        vec3 sky = mix(u_sky_bottom, u_sky_top, t);
                        f_color = vec4(sky, 1.0);
                    }
                """,
            )

            sky_vertices = np.array(
                [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
                dtype="f4",
            )
            sky_vbo = self.ctx.buffer(sky_vertices.tobytes())
            self.sky_vao = self.ctx.vertex_array(
                self.sky_program,
                [(sky_vbo, "2f", "in_position")],
            )
        except Exception as e:
            print(f"Warning: Could not initialize sky rendering: {e}")
            traceback.print_exc()
            self.sky_program = None
            self.sky_vao = None

    def _init_ground_rendering(self):
        """Initialize ground rendering resources."""
        self.grass_albedo = None
        self.ground_program = None
        self.grass_patches = []
        try:
            grass_albedo = generate_grass_albedo(size=512, seed=self.tree.seed + 11)
            self.grass_albedo = self._create_texture_from_array(grass_albedo, repeat=True, mipmaps=True)

            self.ground_program = self.ctx.program(
                vertex_shader="""
                    #version 330
                    uniform mat4 u_view;
                    uniform mat4 u_proj;
                    uniform mat4 u_light_space;
                    uniform vec3 u_light_dir;
                    uniform vec3 u_light_color;
                    uniform vec3 u_camera_pos;
                    uniform vec3 u_fog_color;
                    uniform float u_fog_density;
                    uniform vec2 u_tree_center;
                    uniform int u_debug_view;
                    uniform float u_time;

                    in vec3 in_position;
                    in vec3 in_normal;
                    in vec2 in_uv;

                    in vec3 in_instance_pos;
                    in float in_instance_rot;
                    in float in_instance_height;
                    in float in_instance_width;
                    in vec2 in_instance_lean;

                    out vec2 v_uv;
                    out vec3 v_normal;
                    out vec3 v_world_pos;
                    out vec4 v_light_pos;
                    flat out int v_debug_view;

                    mat3 rot_y(float angle) {
                        float c = cos(angle);
                        float s = sin(angle);
                        return mat3(
                            c, 0.0, -s,
                            0.0, 1.0, 0.0,
                            s, 0.0, c
                        );
                    }

                    void main() {
                        vec3 local = in_position;
                        local.x *= in_instance_width;
                        local.y *= in_instance_height;

                        float sway_phase = (in_instance_pos.x - u_tree_center.x) * 0.01 + u_time;
                        float sway = sin(sway_phase) * 0.35 * local.y;
                        local.x += sway + in_instance_lean.x * local.y;
                        local.z += in_instance_lean.y * local.y;

                        mat3 rot = rot_y(in_instance_rot);
                        vec3 world_pos = rot * local + in_instance_pos;
                        vec3 world_normal = normalize(rot * in_normal);

                        v_uv = in_uv;
                        v_normal = world_normal + normalize(u_light_dir) * 0.0001;
                        v_world_pos = world_pos + normalize(u_camera_pos) * 0.0001;
                        v_light_pos = u_light_space * vec4(world_pos, 1.0);
                        v_debug_view = u_debug_view;

                        gl_Position = u_proj * u_view * vec4(world_pos, 1.0);
                    }
                """,
                fragment_shader="""
                    #version 330
                    uniform sampler2D u_grass_albedo;
                    uniform sampler2DShadow u_shadow_map;
                    uniform vec2 u_shadow_texel;
                    uniform vec3 u_light_dir;
                    uniform vec3 u_light_color;
                    uniform vec3 u_camera_pos;
                    uniform vec3 u_fog_color;
                    uniform float u_fog_density;
                    uniform vec2 u_tree_center;
                    uniform int u_debug_view;
                    uniform float u_time;

                    in vec2 v_uv;
                    in vec3 v_normal;
                    in vec3 v_world_pos;
                    in vec4 v_light_pos;
                    flat in int v_debug_view;
                    out vec4 f_color;

                    float compute_shadow(vec4 light_pos) {
                        if (u_shadow_texel.x <= 0.0 || u_shadow_texel.y <= 0.0) {
                            return 1.0;
                        }

                        vec3 proj = light_pos.xyz / max(light_pos.w, 0.0001);
                        vec3 shadow_coord = proj * 0.5 + 0.5;
                        if (shadow_coord.z > 1.0) {
                            return 1.0;
                        }

                        vec2 jitter = vec2(
                            sin(v_world_pos.x * 0.01 + u_time),
                            cos(v_world_pos.z * 0.01 + u_time)
                        ) * u_shadow_texel * 1.5;
                        vec3 sample_coord = vec3(shadow_coord.xy + jitter, shadow_coord.z - 0.001);
                        return texture(u_shadow_map, sample_coord);
                    }

                    void main() {
                        vec3 albedo = texture(u_grass_albedo, v_uv * 4.0).rgb;
                        vec3 normal = normalize(v_normal);
                        vec3 light_dir = normalize(-u_light_dir);
                        float ndotl = max(dot(normal, light_dir), 0.0);
                        float shadow = compute_shadow(v_light_pos);
                        float lighting = (0.25 + ndotl * 0.75) * shadow;
                        vec3 lit = albedo * (lighting * u_light_color);

                        float tree_dist = length(v_world_pos.xz - u_tree_center);
                        float hue_shift = sin(tree_dist * 0.02 + u_time) * 0.04;
                        lit.g = clamp(lit.g + hue_shift, 0.0, 1.0);

                        float dist = length(v_world_pos - u_camera_pos);
                        float fog = 1.0 - exp(-u_fog_density * dist * dist);
                        vec3 color = mix(lit, u_fog_color, clamp(fog, 0.0, 1.0));

                        if (u_debug_view != 0 || v_debug_view != 0) {
                            color = mix(color, normal * 0.5 + 0.5, 0.6);
                        }

                        f_color = vec4(color, 1.0);
                    }
                """,
            )

            # Provide safe defaults when shadows are unavailable
            if "u_shadow_map" in self.ground_program:
                self.ground_program["u_shadow_map"].value = 0
            if "u_shadow_texel" in self.ground_program:
                self.ground_program["u_shadow_texel"].value = (0.0, 0.0)

            grass_mesh = np.array(self._build_grass_blade_mesh(), dtype="f4")
            grass_vbo = self.ctx.buffer(grass_mesh.tobytes())

            patches = self._build_grass_instance_data_patches(
                area_size=2000.0,
                num_blades=500000,
                seed=self.tree.seed + 29,
                center_x=self.tree.root_x,
                grid_size=20,
            )

            for patch in patches:
                instance_count = int(patch.get("instance_count", 0))
                patch["instance_count"] = instance_count
                if instance_count <= 0:
                    continue
                instance_data = np.array(patch["instance_data"], dtype="f4")
                instance_vbo = self.ctx.buffer(instance_data.tobytes())
                patch["instance_vbo"] = instance_vbo
                patch["vao"] = self.ctx.vertex_array(
                    self.ground_program,
                    [
                        (grass_vbo, "3f 3f 2f", "in_position", "in_normal", "in_uv"),
                        (
                            instance_vbo,
                            "3f 1f 1f 1f 2f /i",
                            "in_instance_pos",
                            "in_instance_rot",
                            "in_instance_height",
                            "in_instance_width",
                            "in_instance_lean",
                        ),
                    ],
                )

            self.grass_patches = patches
        except Exception as e:
            print(f"Warning: Could not initialize ground rendering: {e}")
            traceback.print_exc()
            self.grass_albedo = None
            self.ground_program = None
            self.grass_patches = []

    def _init_bark_textures(self):
        """Initialize bark textures and availability flags."""
        self.bark_albedo = None
        self.bark_normal = None
        self.bark_roughness = None
        self.has_bark_normal = False
        self.has_bark_roughness = False
        try:
            bark_albedo = generate_bark_albedo(size=256, seed=self.tree.seed + 3)
            bark_normal = generate_bark_normal(size=256, seed=self.tree.seed + 5)
            bark_roughness = generate_bark_roughness(size=256, seed=self.tree.seed + 7)

            self.bark_albedo = self._create_texture_from_array(bark_albedo, repeat=True, mipmaps=True)
            self.bark_normal = self._create_texture_from_array(bark_normal, repeat=True, mipmaps=True)
            self.bark_roughness = self._create_texture_from_array(bark_roughness, repeat=True, mipmaps=True)
            self.has_bark_normal = self.bark_normal is not None
            self.has_bark_roughness = self.bark_roughness is not None
        except Exception as e:
            print(f"Warning: Could not initialize bark textures: {e}")
            traceback.print_exc()
            self.bark_albedo = None
            self.bark_normal = None
            self.bark_roughness = None
            self.has_bark_normal = False
            self.has_bark_roughness = False

    def _init_leaf_atlas(self):
        """Initialize the procedural leaf atlas texture."""
        self.leaf_atlas = None
        try:
            cols = max(1, int(self.tree.leaf_atlas_cols))
            rows = max(1, int(self.tree.leaf_atlas_rows))
            cell_size = 128
            atlas = np.zeros((rows * cell_size, cols * cell_size, 4), dtype=np.uint8)
            variant = 0
            for row in range(rows):
                for col in range(cols):
                    leaf = generate_leaf_rgba(size=cell_size, seed=self.tree.seed + 101, variant=variant)
                    y0 = row * cell_size
                    y1 = y0 + cell_size
                    x0 = col * cell_size
                    x1 = x0 + cell_size
                    atlas[y0:y1, x0:x1, :] = leaf
                    variant += 1

            self.leaf_atlas = self._create_texture_from_array(atlas, repeat=False, mipmaps=True)
        except Exception as e:
            print(f"Warning: Could not initialize leaf atlas: {e}")
            traceback.print_exc()
            self.leaf_atlas = None

    def _init_bird_rendering(self):
        """Initialize bird sprite rendering resources."""
        self.bird_instance_vbo = None
        self.bird_program = None
        self.bird_sprite = None
        self.bird_vao = None
        try:
            sprite_size = 64
            cols = max(1, int(self.tree.bird_sprite_cols))
            rows = max(1, int(self.tree.bird_sprite_rows))
            atlas = np.zeros((rows * sprite_size, cols * sprite_size, 4), dtype=np.uint8)

            yy, xx = np.mgrid[0:sprite_size, 0:sprite_size].astype(np.float32)
            uv_x = (xx / max(sprite_size - 1, 1)) * 2.0 - 1.0
            uv_y = (yy / max(sprite_size - 1, 1)) * 2.0 - 1.0

            frame_count = cols * rows
            for frame in range(frame_count):
                flap = math.sin((frame / max(frame_count, 1)) * math.tau)
                body = np.clip(1.0 - (uv_x * uv_x + (uv_y * 1.2) ** 2), 0.0, 1.0)
                wing = np.clip(1.0 - ((uv_x * (1.4 + 0.3 * flap)) ** 2 + (uv_y * 0.8) ** 2), 0.0, 1.0)
                beak = np.clip(1.0 - ((uv_x - 0.55) ** 2 * 12.0 + (uv_y + 0.05) ** 2 * 40.0), 0.0, 1.0)

                alpha = np.clip(body * 0.9 + wing * 0.7 + beak * 0.8, 0.0, 1.0)
                color = np.zeros((sprite_size, sprite_size, 4), dtype=np.uint8)
                color[..., 0] = np.clip((0.2 + wing * 0.4 + beak * 0.8) * 255, 0, 255).astype(np.uint8)
                color[..., 1] = np.clip((0.6 + body * 0.35 + wing * 0.2) * 255, 0, 255).astype(np.uint8)
                color[..., 2] = np.clip((0.85 + body * 0.1) * 255, 0, 255).astype(np.uint8)
                color[..., 3] = np.clip(alpha * 255, 0, 255).astype(np.uint8)

                row = frame // cols
                col = frame % cols
                y0 = row * sprite_size
                y1 = y0 + sprite_size
                x0 = col * sprite_size
                x1 = x0 + sprite_size
                atlas[y0:y1, x0:x1, :] = color

            self.bird_sprite = self._create_texture_from_array(atlas, repeat=False, mipmaps=True)

            self.bird_program = self.ctx.program(
                vertex_shader="""
                    #version 330
                    uniform vec2 u_resolution;
                    uniform vec2 u_sprite_grid;

                    in vec2 in_position;
                    in vec2 in_center;
                    in float in_size;
                    in float in_facing;
                    in float in_flap;
                    in vec2 in_uv_offset;
                    in vec2 in_uv_scale;
                    in float in_frame_index;
                    in vec4 in_body_color;
                    in vec4 in_wing_color;
                    in vec4 in_beak_color;

                    out vec2 v_uv;
                    out vec4 v_color;
                    flat out float v_flap;

                    void main() {
                        vec2 scaled = in_position * in_size;
                        scaled.x *= in_facing;
                        vec2 world = in_center + scaled;
                        vec2 ndc = (world / max(u_resolution, vec2(1.0))) * 2.0 - 1.0;
                        gl_Position = vec4(ndc, 0.0, 1.0);

                        vec2 base_uv = in_uv_offset + (in_position * 0.5 + 0.5) * in_uv_scale;
                        float cols = max(u_sprite_grid.x, 1.0);
                        float rows = max(u_sprite_grid.y, 1.0);
                        float idx = mod(in_frame_index, cols * rows);
                        float col = mod(idx, cols);
                        float row = floor(idx / cols);
                        vec2 tile_origin = vec2(col, row) / vec2(cols, rows);
                        vec2 tile_scale = vec2(1.0 / cols, 1.0 / rows);
                        v_uv = tile_origin + base_uv * tile_scale;

                        float wing_mix = clamp(abs(in_position.y) * (0.5 + 0.5 * abs(in_flap)), 0.0, 1.0);
                        v_color = mix(in_body_color, in_wing_color, wing_mix);
                        v_color.rgb = mix(v_color.rgb, in_beak_color.rgb, step(0.35, in_position.x) * step(0.1, in_position.y));
                        v_color.a *= in_beak_color.a;
                        v_flap = in_flap;
                    }
                """,
                fragment_shader="""
                    #version 330
                    uniform sampler2D u_bird_sprite;
                    uniform vec2 u_resolution;
                    uniform vec2 u_sprite_grid;
                    uniform float u_alpha_cutoff;

                    in vec2 v_uv;
                    in vec4 v_color;
                    flat in float v_flap;
                    out vec4 f_color;

                    void main() {
                        vec4 tex = texture(u_bird_sprite, v_uv);
                        float alpha = tex.a * v_color.a * (0.9 + 0.1 * abs(v_flap));
                        if (alpha < u_alpha_cutoff) {
                            discard;
                        }
                        vec3 color = tex.rgb * v_color.rgb;
                        color += vec3(u_sprite_grid.x, u_sprite_grid.y, u_resolution.x / max(u_resolution.y, 1.0)) * 0.00001;
                        f_color = vec4(color, alpha);
                    }
                """,
            )

            bird_quad = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype="f4")
            bird_vbo = self.ctx.buffer(bird_quad.tobytes())
            self.bird_instance_vbo = self.ctx.buffer(reserve=22 * 4)
            self.bird_vao = self.ctx.vertex_array(
                self.bird_program,
                [
                    (bird_vbo, "2f", "in_position"),
                    (
                        self.bird_instance_vbo,
                        "2f 1f 1f 1f 2f 2f 1f 4f 4f 4f /i",
                        "in_center",
                        "in_size",
                        "in_facing",
                        "in_flap",
                        "in_uv_offset",
                        "in_uv_scale",
                        "in_frame_index",
                        "in_body_color",
                        "in_wing_color",
                        "in_beak_color",
                    ),
                ],
            )
        except Exception as e:
            print(f"Warning: Could not initialize bird rendering: {e}")
            traceback.print_exc()
            self.bird_instance_vbo = None
            self.bird_program = None
            self.bird_sprite = None
            self.bird_vao = None

    def _init_ui_rendering(self):
        """Initialize UI line rendering resources."""
        self.ui_line_instance_vbo = None
        self.ui_line_program = None
        self.ui_line_vao = None
        try:
            self.ui_line_program = self.ctx.program(
                vertex_shader="""
                    #version 330
                    uniform vec2 u_resolution;

                    in vec2 in_corner;
                    in vec2 in_start;
                    in vec2 in_end;
                    in float in_width;
                    in vec4 in_color;

                    out vec4 v_color;

                    void main() {
                        vec2 dir = in_end - in_start;
                        float len = length(dir);
                        vec2 dir_n = len > 0.0001 ? dir / len : vec2(1.0, 0.0);
                        vec2 normal = vec2(-dir_n.y, dir_n.x);

                        vec2 center = (in_start + in_end) * 0.5;
                        vec2 offset = dir_n * (in_corner.x * len * 0.5) + normal * (in_corner.y * in_width * 0.5);
                        vec2 world = center + offset;
                        vec2 ndc = (world / max(u_resolution, vec2(1.0))) * 2.0 - 1.0;
                        gl_Position = vec4(ndc, 0.0, 1.0);
                        v_color = in_color;
                    }
                """,
                fragment_shader="""
                    #version 330
                    in vec4 v_color;
                    out vec4 f_color;

                    void main() {
                        f_color = v_color;
                    }
                """,
            )

            ui_corners = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype="f4")
            ui_corner_vbo = self.ctx.buffer(ui_corners.tobytes())
            self.ui_line_instance_vbo = self.ctx.buffer(reserve=9 * 4 * 8)
            self.ui_line_vao = self.ctx.vertex_array(
                self.ui_line_program,
                [
                    (ui_corner_vbo, "2f", "in_corner"),
                    (
                        self.ui_line_instance_vbo,
                        "2f 2f 1f 4f /i",
                        "in_start",
                        "in_end",
                        "in_width",
                        "in_color",
                    ),
                ],
            )
        except Exception as e:
            print(f"Warning: Could not initialize UI rendering: {e}")
            traceback.print_exc()
            self.ui_line_instance_vbo = None
            self.ui_line_program = None
            self.ui_line_vao = None
    
    def _update_branch_buffers(self, positions):
        """Update branch vertex buffer with new positions."""
        if not hasattr(self, 'branch_instance_vbo') or not self.branch_instance_vbo:
            return 0
            
        # Convert positions to numpy array if needed
        if not isinstance(positions, np.ndarray):
            positions = np.array(positions, dtype='f4')
            
        # Limit the number of branches to render
        num_branches = min(len(positions) // 3, self.max_branches)
        if num_branches == 0:
            return 0
            
        # Update the VBO with new positions
        self.branch_instance_vbo.orphan()
        self.branch_instance_vbo.write(positions[:num_branches * 3].tobytes())
        return num_branches
        
    def _upload_static_leaf_data(self):
        """Build and upload static leaf instance data once at initialization."""
        leaf_data: List[float] = []
        for leaf in self.tree.canopy_leaves:
            # Static data: branch_index, offset, axes, UV, color
            leaf_data.extend([
                float(leaf.branch_index),      # branch index
                *leaf.offset,                   # offset from branch (x, y, z)
                *leaf.axis_x,                   # rotation axis x
                *leaf.axis_y,                   # rotation axis y
                *leaf.axis_z,                   # rotation axis z (normal)
                *leaf.uv_offset,                # UV offset
                *leaf.uv_scale,                 # UV scale
                float(leaf.atlas_index),        # atlas index
                *leaf.base_color,               # color (RGBA)
                *leaf.color_variance,           # color variance (RGB)
                leaf.variance_amount,           # variance amount
            ])

        if leaf_data:
            data = array('f', leaf_data)
            self.leaf_instance_vbo.orphan(len(data) * 4)
            self.leaf_instance_vbo.write(data)

    def _init_shadow_rendering(self):
        """Initialize shadow mapping resources (depth map, shaders, and VAOs)."""
        try:
            self.shadow_map = self.ctx.depth_texture((self.shadow_size, self.shadow_size))
            # Use hardware depth comparisons where available
            self.shadow_map.compare_func = "<="
            self.shadow_map.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.shadow_fbo = self.ctx.framebuffer(depth_attachment=self.shadow_map)

            if not self.branch_vbo or not self.leaf_vbo:
                raise RuntimeError("Branch/leaf vertex buffers must be initialized before shadows")

            self.branch_shadow_program = self.ctx.program(
                vertex_shader="""
                    #version 330
                    uniform mat4 u_light_space;
                    in vec3 in_position;
                    in vec4 in_model_0;
                    in vec4 in_model_1;
                    in vec4 in_model_2;
                    in vec4 in_model_3;

                    void main() {
                        mat4 model = mat4(in_model_0, in_model_1, in_model_2, in_model_3);
                        gl_Position = u_light_space * model * vec4(in_position, 1.0);
                    }
                """,
                fragment_shader="""
                    #version 330
                    void main() {
                        // Depth is written automatically to the shadow map
                    }
                """,
            )

            self.leaf_shadow_program = self.ctx.program(
                vertex_shader="""
                    #version 330
                    uniform mat4 u_light_space;
                    uniform samplerBuffer u_branch_positions;
                    uniform vec2 u_atlas_grid;
                    uniform float u_alpha_cutoff;

                    in vec2 in_position;
                    in float in_branch_index;
                    in vec3 in_offset;
                    in vec3 in_axis_x;
                    in vec3 in_axis_y;
                    in vec3 in_axis_z;
                    in vec2 in_uv_offset;
                    in vec2 in_uv_scale;
                    in float in_atlas_index;

                    out vec2 v_uv;
                    flat out float v_alpha_cutoff;

                    vec3 fetch_branch_pos(int idx) {
                        return texelFetch(u_branch_positions, idx).xyz;
                    }

                    void main() {
                        int branch_idx = int(in_branch_index + 0.5);
                        vec3 branch_pos = fetch_branch_pos(branch_idx);
                        vec3 local = in_axis_x * in_position.x + in_axis_y * in_position.y;
                        vec3 world_pos = branch_pos + in_offset + local;

                        vec2 base_uv = in_position * 0.5 + 0.5;
                        float atlas_cols = max(u_atlas_grid.x, 1.0);
                        float atlas_rows = max(u_atlas_grid.y, 1.0);
                        float atlas_idx = max(in_atlas_index, 0.0);
                        float col = mod(atlas_idx, atlas_cols);
                        float row = floor(atlas_idx / atlas_cols);
                        vec2 atlas_origin = vec2(col, row) / vec2(atlas_cols, atlas_rows);
                        vec2 atlas_scale = vec2(1.0 / atlas_cols, 1.0 / atlas_rows);
                        v_uv = atlas_origin + (in_uv_offset + base_uv * in_uv_scale) * atlas_scale;
                        v_alpha_cutoff = u_alpha_cutoff;

                        gl_Position = u_light_space * vec4(world_pos, 1.0);
                    }
                """,
                fragment_shader="""
                    #version 330
                    uniform sampler2D u_leaf_atlas;
                    in vec2 v_uv;
                    flat in float v_alpha_cutoff;

                    void main() {
                        float alpha = texture(u_leaf_atlas, v_uv).a;
                        if (alpha < v_alpha_cutoff) {
                            discard;
                        }
                    }
                """,
            )

            self.branch_shadow_vao = self.ctx.vertex_array(
                self.branch_shadow_program,
                [
                    (self.branch_vbo, "3f", "in_position"),
                    (
                        self.branch_instance_vbo,
                        "16f 8x /i",
                        "in_model_0",
                        "in_model_1",
                        "in_model_2",
                        "in_model_3",
                    ),
                ],
            )

            self.leaf_shadow_vao = self.ctx.vertex_array(
                self.leaf_shadow_program,
                [
                    (self.leaf_vbo, "2f", "in_position"),
                    (
                        self.leaf_instance_vbo,
                        "1f 3f 3f 3f 3f 2f 2f 1f 8x /i",
                        "in_branch_index",
                        "in_offset",
                        "in_axis_x",
                        "in_axis_y",
                        "in_axis_z",
                        "in_uv_offset",
                        "in_uv_scale",
                        "in_atlas_index",
                    ),
                ],
            )

            # Texture buffer for branch positions (vec4 per branch)
            self.branch_position_tex = self.ctx.texture_buffer(self.branch_position_tbo, components=4, dtype="f4")

        except Exception as e:
            print(f"Warning: Could not initialize shadow rendering: {e}")
            traceback.print_exc()
            self.shadow_fbo = None
            self.shadow_map = None
            self.branch_shadow_program = None
            self.leaf_shadow_program = None
            self.branch_shadow_vao = None
            self.leaf_shadow_vao = None
            self.branch_position_tex = None

    def render(self):
        width, height = self.window.get_framebuffer_size()
        aspect = width / max(height, 1)
        view = array('f', self._look_at(self.camera_position, self.camera_target, self.camera_up))
        proj = array('f', self._perspective(math.radians(40.0), aspect, 10.0, 2000.0))
        light_target = (self.tree.w * 0.5, self.tree.h * 0.4, 0.0)
        light_dir = self.light_direction
        light_distance = 1200.0
        light_pos = (
            light_target[0] - light_dir[0] * light_distance,
            light_target[1] - light_dir[1] * light_distance,
            light_target[2] - light_dir[2] * light_distance,
        )
        light_view = self._look_at(light_pos, light_target, (0.0, 1.0, 0.0))
        ortho_extent = 900.0
        light_proj = self._ortho(
            -ortho_extent, ortho_extent,
            -ortho_extent, ortho_extent,
            -1200.0, 1600.0,
        )
        light_space = array('f', self._mul_mat4(light_proj, light_view))

        # Extract frustum planes for culling grass patches
        view_proj = self._mul_mat4(tuple(proj), tuple(view))
        frustum_planes = self._extract_frustum_planes(view_proj)

        branch_instances = self.tree.build_branch_instances()
        if branch_instances:
            data = array('f', branch_instances)
            self.branch_instance_vbo.orphan(len(data) * 4)
            self.branch_instance_vbo.write(data)

        # GPU-driven: Update branch positions only (not all leaf data!)
        branch_positions: List[float] = []
        for branch in self.tree.branches:
            branch_positions.extend([branch.end_x, branch.end_y, branch.end_z, 1.0])

        if branch_positions:
            branch_count = min(len(branch_positions) // 4, self.max_branches)
            data = array('f', branch_positions[:branch_count * 4])
            self.branch_position_tbo.write(data.tobytes())

        # Alpha-to-coverage for leaves (if available)
        if self.use_alpha_to_coverage:
            self.ctx.enable(self.alpha_to_coverage_flag)
        else:
            if self.alpha_to_coverage_flag is not None:
                self.ctx.disable(self.alpha_to_coverage_flag)

        shadow_ready = bool(
            self.shadow_fbo
            and self.shadow_map
            and self.branch_shadow_program
            and self.branch_shadow_vao
        )
        leaf_shadow_ready = bool(
            shadow_ready
            and self.leaf_shadow_program
            and self.leaf_shadow_vao
            and self.branch_position_tex
            and self.leaf_atlas
        )

        if shadow_ready:
            self.shadow_fbo.use()
            self.ctx.viewport = (0, 0, self.shadow_size, self.shadow_size)
            self.ctx.clear(depth=1.0)
            if branch_instances:
                self.branch_shadow_program["u_light_space"].write(light_space)
                self.branch_shadow_vao.render(instances=len(branch_instances) // 24)
            if self.leaf_count > 0 and leaf_shadow_ready:
                # Bind branch position texture buffer for GPU lookup
                self.branch_position_tex.use(location=4)
                self.leaf_shadow_program["u_branch_positions"].value = 4
                self.leaf_shadow_program["u_light_space"].write(light_space)
                self.leaf_shadow_program["u_atlas_grid"].value = (self.tree.leaf_atlas_cols, self.tree.leaf_atlas_rows)
                self.leaf_shadow_program["u_alpha_cutoff"].value = 0.2
                self.leaf_atlas.use(location=0)
                self.leaf_shadow_program["u_leaf_atlas"].value = 0
                self.leaf_shadow_vao.render(instances=self.leaf_count)
        # Grass shadows disabled for performance (grass doesn't need to cast shadows)
        # self.ground_shadow_program["u_light_space"].write(light_space)
        # for patch in self.grass_patches:
        #     min_x, min_z, max_x, max_z = patch['bounds']
        #     if self._aabb_in_frustum(frustum_planes, min_x, min_z, max_x, max_z):
        #         patch['shadow_vao'].render(instances=patch['instance_count'])

        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.clear(0.015, 0.05, 0.11, depth=1.0)
        self.ctx.disable(moderngl.DEPTH_TEST)
        sky_ready = bool(
            getattr(self, "sky_program", None)
            and getattr(self, "sky_vao", None)
            and "u_sky_top" in self.sky_program
            and "u_sky_bottom" in self.sky_program
        )
        if sky_ready:
            self.sky_program["u_sky_top"].value = (0.18, 0.28, 0.5)
            self.sky_program["u_sky_bottom"].value = self.fog_color
            self.sky_vao.render()
        self.ctx.enable(moderngl.DEPTH_TEST)

        shadow_texel = (1.0 / self.shadow_size, 1.0 / self.shadow_size) if shadow_ready else None
        if shadow_ready:
            self.shadow_map.use(location=3)

        ground_required_uniforms = (
            "u_grass_albedo",
            "u_view",
            "u_proj",
            "u_light_space",
            "u_light_dir",
            "u_light_color",
            "u_camera_pos",
            "u_fog_color",
            "u_fog_density",
            "u_tree_center",
            "u_debug_view",
            "u_time",
        )
        ground_ready = bool(
            getattr(self, "grass_albedo", None)
            and getattr(self, "ground_program", None)
            and getattr(self, "grass_patches", None) is not None
            and all(name in self.ground_program for name in ground_required_uniforms)
        )
        if ground_ready:
            # Bind grass texture for ground
            self.grass_albedo.use(location=0)

            self.ground_program["u_grass_albedo"].value = 0
            self.ground_program["u_view"].write(view)
            self.ground_program["u_proj"].write(proj)
            self.ground_program["u_light_space"].write(light_space)
            if shadow_ready and "u_shadow_map" in self.ground_program and "u_shadow_texel" in self.ground_program:
                self.ground_program["u_shadow_map"].value = 3
                self.ground_program["u_shadow_texel"].value = shadow_texel
            self.ground_program["u_light_dir"].value = self.light_direction
            self.ground_program["u_light_color"].value = self.light_color
            self.ground_program["u_camera_pos"].value = self.camera_position
            self.ground_program["u_fog_color"].value = self.fog_color
            self.ground_program["u_fog_density"].value = self.fog_density
            self.ground_program["u_tree_center"].value = (self.tree.root_x, self.tree.root_y)
            self.ground_program["u_debug_view"].value = self.debug_view_mode
            self.ground_program["u_time"].value = self.tree.time

            # Render grass patches with frustum culling
            for patch in self.grass_patches:
                vao = patch.get("vao")
                instance_count = int(patch.get("instance_count", 0))
                if not vao or instance_count <= 0:
                    continue
                min_x, min_z, max_x, max_z = patch["bounds"]
                if self._aabb_in_frustum(frustum_planes, min_x, min_z, max_x, max_z):
                    vao.render(instances=instance_count)

        branch_required_uniforms = (
            "u_view",
            "u_proj",
            "u_light_space",
            "u_camera_pos",
            "u_bark_uv_scale",
            "u_has_normal",
            "u_has_roughness",
            "u_bark_albedo",
            "u_bark_normal",
            "u_bark_roughness",
            "u_light_dir",
            "u_light_color",
            "u_fog_color",
            "u_fog_density",
            "u_debug_view",
            "u_glow",
        )
        bark_ready = bool(getattr(self, "bark_albedo", None) and getattr(self, "bark_normal", None) and getattr(self, "bark_roughness", None))
        branch_ready = bool(
            branch_instances
            and getattr(self, "branch_program", None)
            and getattr(self, "branch_vao", None)
            and bark_ready
            and all(name in self.branch_program for name in branch_required_uniforms)
        )
        if branch_ready:
            self.branch_program["u_view"].write(view)
            self.branch_program["u_proj"].write(proj)
            self.branch_program["u_light_space"].write(light_space)
            self.branch_program["u_camera_pos"].value = self.camera_position
            self.branch_program["u_bark_uv_scale"].value = (0.08, 1.2)
            self.branch_program["u_has_normal"].value = 1 if self.has_bark_normal else 0
            self.branch_program["u_has_roughness"].value = 1 if self.has_bark_roughness else 0
            self.bark_albedo.use(location=0)
            self.bark_normal.use(location=1)
            self.bark_roughness.use(location=2)
            self.branch_program["u_bark_albedo"].value = 0
            self.branch_program["u_bark_normal"].value = 1
            self.branch_program["u_bark_roughness"].value = 2
            if shadow_ready:
                self.branch_program["u_shadow_map"].value = 3
                self.branch_program["u_shadow_texel"].value = shadow_texel
            self.branch_program["u_light_dir"].value = self.light_direction
            self.branch_program["u_light_color"].value = self.light_color
            self.branch_program["u_fog_color"].value = self.fog_color
            self.branch_program["u_fog_density"].value = self.fog_density
            self.branch_program["u_debug_view"].value = self.debug_view_mode

            self.branch_program["u_glow"].value = 0.0
            self.branch_vao.render(instances=len(branch_instances) // 24)

        leaf_required_uniforms = (
            "u_branch_positions",
            "u_view",
            "u_proj",
            "u_light_space",
            "u_atlas_grid",
            "u_alpha_cutoff",
            "u_leaf_atlas",
            "u_light_dir",
            "u_light_color",
            "u_camera_pos",
            "u_fog_color",
            "u_fog_density",
            "u_canopy_center_y",
            "u_canopy_half_height",
            "u_debug_view",
        )
        branch_position_source = getattr(self, "branch_position_tex", None) or getattr(self, "branch_position_tbo", None)
        leaf_ready = bool(
            self.leaf_count > 0
            and getattr(self, "leaf_program", None)
            and getattr(self, "leaf_vao", None)
            and getattr(self, "leaf_atlas", None)
            and branch_position_source
            and all(name in self.leaf_program for name in leaf_required_uniforms)
        )
        if leaf_ready:
            # Disable blending for alpha cutout rendering (depth write ON, blending OFF)
            self.ctx.disable(moderngl.BLEND)

            # Bind branch position TBO for GPU lookup
            branch_position_source.use(location=4)
            self.leaf_program["u_branch_positions"].value = 4
            self.leaf_program["u_view"].write(view)
            self.leaf_program["u_proj"].write(proj)
            self.leaf_program["u_light_space"].write(light_space)
            self.leaf_program["u_atlas_grid"].value = (self.tree.leaf_atlas_cols, self.tree.leaf_atlas_rows)
            self.leaf_program["u_alpha_cutoff"].value = 0.2
            self.leaf_atlas.use(location=0)
            self.leaf_program["u_leaf_atlas"].value = 0
            if shadow_ready:
                self.leaf_program["u_shadow_map"].value = 3
                self.leaf_program["u_shadow_texel"].value = shadow_texel
            self.leaf_program["u_light_dir"].value = self.light_direction
            self.leaf_program["u_light_color"].value = self.light_color
            self.leaf_program["u_camera_pos"].value = self.camera_position
            self.leaf_program["u_fog_color"].value = self.fog_color
            self.leaf_program["u_fog_density"].value = self.fog_density
            self.leaf_program["u_canopy_center_y"].value = self.tree.canopy_center_y
            self.leaf_program["u_canopy_half_height"].value = self.tree.canopy_half_height
            self.leaf_program["u_debug_view"].value = self.debug_view_mode
            self.leaf_vao.render(instances=self.leaf_count)
            if self.use_alpha_to_coverage:
                self.ctx.disable(self.alpha_to_coverage_flag)

            # Re-enable blending for subsequent rendering
            self.ctx.enable(moderngl.BLEND)

        self.ctx.disable(moderngl.DEPTH_TEST)
        bird_instances = self.tree.build_bird_instances()
        bird_required_uniforms = ("u_resolution", "u_sprite_grid", "u_alpha_cutoff", "u_bird_sprite")
        bird_ready = bool(
            bird_instances
            and getattr(self, "bird_instance_vbo", None)
            and getattr(self, "bird_program", None)
            and getattr(self, "bird_sprite", None)
            and getattr(self, "bird_vao", None)
            and all(name in self.bird_program for name in bird_required_uniforms)
        )
        if bird_ready:
            data = array('f', bird_instances)
            self.bird_instance_vbo.orphan(len(data) * 4)
            self.bird_instance_vbo.write(data)
            self.bird_program["u_resolution"].value = (width, height)
            self.bird_program["u_sprite_grid"].value = (
                self.tree.bird_sprite_cols,
                self.tree.bird_sprite_rows,
            )
            self.bird_program["u_alpha_cutoff"].value = 0.2
            self.bird_sprite.use(location=0)
            self.bird_program["u_bird_sprite"].value = 0
            self.bird_vao.render(instances=1)

        flicker = self.tree.flicker
        ui_color = (
            (120 / 255) * flicker,
            (235 / 255) * flicker,
            (255 / 255) * flicker,
            180 / 255,
        )
        ui_lines = [
            (20.0, 20.0, 100.0, 20.0),
            (20.0, 20.0, 20.0, 100.0),
            (width - 100.0, 20.0, width - 20.0, 20.0),
            (width - 20.0, 20.0, width - 20.0, 100.0),
        ]
        ui_instances = []
        for x1, y1, x2, y2 in ui_lines:
            ui_instances.extend([x1, y1, x2, y2, 2.0, *ui_color])
        ui_ready = bool(
            ui_instances
            and getattr(self, "ui_line_instance_vbo", None)
            and getattr(self, "ui_line_program", None)
            and getattr(self, "ui_line_vao", None)
            and "u_resolution" in self.ui_line_program
        )
        if ui_ready:
            data = array('f', ui_instances)
            self.ui_line_instance_vbo.orphan(len(data) * 4)
            self.ui_line_instance_vbo.write(data)
            self.ui_line_program["u_resolution"].value = (width, height)
            self.ui_line_vao.render(instances=len(ui_instances) // 9)


class HolographicWindow(pyglet.window.Window):
    def __init__(self):
        display = pyglet.display.get_display()
        screen = display.get_default_screen()
        
        # Try to create a window with multisampling, fallback to basic config if not supported
        try:
            config = pyglet.gl.Config(
                double_buffer=True,
                sample_buffers=1,
                samples=4,
                depth_size=24,
                stencil_size=8
            )
            super().__init__(
                width=min(1280, screen.width - 100),
                height=min(800, screen.height - 100),
                fullscreen=False,  # Start in windowed mode
                caption="Holographic Tree",
                config=config,
                vsync=True,  # Enable vsync for smoother rendering
                resizable=True
            )
        except pyglet.window.NoSuchConfigException:
            print("Warning: Multisampling not supported, falling back to basic config")
            config = pyglet.gl.Config(double_buffer=True, depth_size=24)
            super().__init__(
                width=min(1280, screen.width - 100),
                height=min(800, screen.height - 100),
                fullscreen=False,
                caption="Holographic Tree",
                config=config,
                vsync=True,
                resizable=True
            )
            
        self.set_mouse_visible(True)  # Make mouse visible for easier window management

        self.tree = HolographicTree(self.width, self.height)
        self.renderer = OpenGLRenderer(self, self.tree)
        self.info_batch = pyglet.graphics.Batch()
        self.avg_fps = 0.0
        self._init_labels()

        self.last_time = time.perf_counter()
        self.frame_times: List[float] = []
        pyglet.clock.schedule_interval(self.update, 1 / 165.0)

    def _init_labels(self):
        margin_x = 30
        top_start_y = self.height - 40
        line_gap = 22
        base_color = (120, 235, 255, 180)
        self.top_labels = {
            "status": pyglet.text.Label(
                "HOLOGRAPHIC PROJECTION ACTIVE",
                font_name="Arial",
                font_size=16,
                x=margin_x,
                y=top_start_y,
                color=base_color,
                batch=self.info_batch,
            ),
            "branches": pyglet.text.Label(
                "BRANCHES: 0",
                font_name="Arial",
                font_size=16,
                x=margin_x,
                y=top_start_y - line_gap,
                color=base_color,
                batch=self.info_batch,
            ),
            "wind": pyglet.text.Label(
                "WIND: 0.0",
                font_name="Arial",
                font_size=16,
                x=margin_x,
                y=top_start_y - line_gap * 2,
                color=base_color,
                batch=self.info_batch,
            ),
            "fps": pyglet.text.Label(
                "FPS: 0.0",
                font_name="Arial",
                font_size=16,
                x=margin_x,
                y=top_start_y - line_gap * 3,
                color=base_color,
                batch=self.info_batch,
            ),
        }
        bottom_start_y = 30
        self.bottom_labels = [
            pyglet.text.Label(
                "Holographic Tree",
                font_name="Arial",
                font_size=16,
                x=margin_x,
                y=bottom_start_y + line_gap,
                color=base_color,
                batch=self.info_batch,
            ),
            pyglet.text.Label(
                "Press R/Space to regenerate  •  ESC to quit",
                font_name="Arial",
                font_size=16,
                x=margin_x,
                y=bottom_start_y,
                color=base_color,
                batch=self.info_batch,
            ),
        ]

    def _update_labels(self):
        flicker = self.tree.flicker
        glow_color = (
            max(0, min(255, int(120 * flicker))),
            max(0, min(255, int(235 * flicker))),
            max(0, min(255, int(255 * flicker))),
            180,
        )
        for label in self.top_labels.values():
            label.color = glow_color
        for label in self.bottom_labels:
            label.color = glow_color

        self.top_labels["branches"].text = f"BRANCHES: {len(self.tree.branches)}"
        wind_strength = self.tree.wind.base_strength + self.tree.wind.gust_strength
        self.top_labels["wind"].text = f"WIND: {wind_strength:4.1f}"
        self.top_labels["fps"].text = f"FPS: {self.avg_fps:4.1f}"

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            self.close()
        elif symbol in (pyglet.window.key.R, pyglet.window.key.SPACE):
            self.tree.regenerate_tree()
        elif symbol == pyglet.window.key._0:
            # Final with fog
            self.renderer.debug_view_mode = 0
            print("View: Final shaded WITH fog")
        elif symbol == pyglet.window.key._1:
            # Debug view: albedo only
            self.renderer.debug_view_mode = 1
            print("View: Albedo only (no lighting)")
        elif symbol == pyglet.window.key._2:
            # Debug view: normals
            self.renderer.debug_view_mode = 2
            print("View: Normals visualization")
        elif symbol == pyglet.window.key._3:
            # Debug view: final shaded without fog (default)
            self.renderer.debug_view_mode = 3
            print("View: Final shaded WITHOUT fog (default)")

    def update(self, _dt):
        now = time.perf_counter()
        dt = now - self.last_time
        self.last_time = now
        self.frame_times.append(dt)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
        if self.frame_times:
            avg_dt = sum(self.frame_times) / len(self.frame_times)
            self.avg_fps = 1.0 / avg_dt if avg_dt > 0 else 0.0
        self.tree.update(dt)
        self._update_labels()

    def on_draw(self):
        self.clear()
        self.renderer.render()
        self._clear_gl_errors()
        self.info_batch.draw()

    @staticmethod
    def _clear_gl_errors():
        while pyglet.gl.glGetError() != pyglet.gl.GL_NO_ERROR:
            continue


def main():
    try:
        window = HolographicWindow()
        print("Window created successfully. Press ESC to exit.")
        pyglet.app.run()
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()
