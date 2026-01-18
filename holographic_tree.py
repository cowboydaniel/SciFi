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
from pathlib import Path
from typing import List

import moderngl
import pyglet
import numpy as np


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
    Generate procedural bark albedo (color) texture.

    Returns:
        numpy array of shape (size, size, 3) with uint8 values
    """
    rng = np.random.RandomState(seed)

    y, x = np.meshgrid(
        np.linspace(0, 1, size, dtype=np.float32),
        np.linspace(0, 1, size, dtype=np.float32),
        indexing='ij'
    )

    # Base bark noise (rough texture)
    bark_noise = _fbm_noise(x * size * 0.05, y * size * 0.05, octaves=5, persistence=0.55, seed=seed)

    # Vertical grain pattern
    grain = _fbm_noise(x * size * 0.3, y * size * 0.02, octaves=3, persistence=0.4, seed=seed + 50)
    grain = grain * 0.4 + 0.6

    # Darker crevices (vertical)
    crevice = _fbm_noise(x * size * 0.15, y * size * 0.04, octaves=4, persistence=0.5, seed=seed + 100)
    crevice = _smoothstep(0.4, 0.6, crevice)
    darken = 1.0 - crevice * 0.4

    # Combine noise patterns
    brightness = bark_noise * grain * darken
    brightness = brightness * 0.5 + 0.25  # Remap to darker range

    # Warm brown color
    base_r = rng.uniform(0.45, 0.55)
    base_g = rng.uniform(0.3, 0.38)
    base_b = rng.uniform(0.2, 0.28)

    r = np.clip(brightness * base_r * 1.1, 0, 1)
    g = np.clip(brightness * base_g, 0, 1)
    b = np.clip(brightness * base_b, 0, 1)

    # Convert to uint8
    rgb = np.stack([r, g, b], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)

    return rgb


def generate_bark_normal(size=256, seed=0):
    """
    Generate procedural bark normal map (tangent space).

    Returns:
        numpy array of shape (size, size, 3) with uint8 values (0-255 maps to -1 to 1)
    """
    rng = np.random.RandomState(seed)

    y, x = np.meshgrid(
        np.linspace(0, 1, size, dtype=np.float32),
        np.linspace(0, 1, size, dtype=np.float32),
        indexing='ij'
    )

    # Generate height map (same patterns as albedo)
    height = _fbm_noise(x * size * 0.05, y * size * 0.05, octaves=5, persistence=0.55, seed=seed)

    # Add vertical grain bumps
    grain = _fbm_noise(x * size * 0.3, y * size * 0.02, octaves=3, persistence=0.4, seed=seed + 50)
    height = height * 0.7 + grain * 0.3

    # Scale height
    height = height * 0.3

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
    Generate procedural grass albedo texture.

    Returns:
        numpy array of shape (size, size, 3) with uint8 values
    """
    rng = np.random.RandomState(seed)

    y, x = np.meshgrid(
        np.linspace(0, 1, size, dtype=np.float32),
        np.linspace(0, 1, size, dtype=np.float32),
        indexing='ij'
    )

    # Multi-scale grass noise
    grass_noise = _fbm_noise(x * size * 0.1, y * size * 0.1, octaves=5, persistence=0.5, seed=seed)

    # Add finer detail
    detail = _fbm_noise(x * size * 0.4, y * size * 0.4, octaves=3, persistence=0.6, seed=seed + 200)
    grass_noise = grass_noise * 0.7 + detail * 0.3

    # Color variation (green with some yellowing and dark patches)
    base_brightness = grass_noise * 0.4 + 0.4

    # Base grass green
    base_r = rng.uniform(0.25, 0.35)
    base_g = rng.uniform(0.45, 0.55)
    base_b = rng.uniform(0.2, 0.3)

    # Add color variation
    r = np.clip(base_brightness * base_r * 1.1, 0, 1)
    g = np.clip(base_brightness * base_g, 0, 1)
    b = np.clip(base_brightness * base_b, 0, 1)

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
            rng.uniform(-0.18, 0.22),
            rng.uniform(-0.12, 0.16),
            rng.uniform(-0.2, 0.18),
        )
        variance_amount = rng.uniform(0.25, 0.6)
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
        - 60-140 tiny leaf cards per cluster
        """
        self.canopy_leaves = []

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

            # Skip ~20% of tips randomly for negative space
            if random.random() < 0.2:
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

            # Reduce leaves per cluster for performance (was 70-120)
            # Target: 15-30 leaves per cluster
            leaves_per_cluster = random.randint(15, 30)
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

                self.canopy_leaves.append(CanopyLeaf(
                    branch_index=tip_idx,
                    offset=(card_offset_x, card_offset_y, card_offset_z),
                    axis_x=axis_x,
                    axis_y=axis_y,
                    axis_z=axis_z,
                    base_color=jittered_color,
                    atlas_index=atlas_index,
                    uv_offset=uv_offset,
                    uv_scale=uv_scale,
                    color_variance=color_variance,
                    variance_amount=variance_amount,
                ))

        # Debug output
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

    def __init__(self, window: pyglet.window.Window, tree: HolographicTree):
        self.window = window
        self.tree = tree
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.alpha_to_coverage_flag = getattr(moderngl, "SAMPLE_ALPHA_TO_COVERAGE", None)
        self.use_alpha_to_coverage = self.ctx.fbo.samples > 1 and self.alpha_to_coverage_flag is not None

        # Debug view mode: 0=final with fog, 1=albedo only, 2=normals only, 3=final no fog
        self.debug_view_mode = 0  # Final with fog for atmospheric depth

        # Camera positioned for photo-like framing (FOV 40 degrees)
        # Further back and lower to show full tree and ground
        self.camera_position = (tree.w * 0.5, tree.h * 0.5, 650.0)
        self.camera_target = (tree.w * 0.5, tree.h * 0.48, 0.0)
        self.camera_up = (0.0, 1.0, 0.0)
        # Directional sun: coming from upper-left-front
        # Direction vector points FROM surface TO light (negated for shader)
        # Raw direction: (-0.4, -1.0, -0.2) normalized
        light_dir_raw = (-0.4, -1.0, -0.2)
        length = math.sqrt(light_dir_raw[0]**2 + light_dir_raw[1]**2 + light_dir_raw[2]**2)
        self.light_direction = (light_dir_raw[0]/length, light_dir_raw[1]/length, light_dir_raw[2]/length)
        # Warm sunlight (slightly yellow)
        self.light_color = (1.0, 0.96, 0.88)
        # Sky blue fog color
        self.fog_color = (0.55, 0.68, 0.82)
        # Mild atmospheric fog for depth (not too heavy)
        self.fog_density = 0.0008

        self.shadow_size = 2048
        self.shadow_map = self.ctx.depth_texture((self.shadow_size, self.shadow_size))
        self.shadow_map.repeat_x = False
        self.shadow_map.repeat_y = False
        self.shadow_map.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.shadow_fbo = self.ctx.framebuffer(depth_attachment=self.shadow_map)

        self.branch_program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                in vec3 in_normal;
                in vec2 in_uv;
                in vec3 in_tangent;
                in vec4 in_transform_0;
                in vec4 in_transform_1;
                in vec4 in_transform_2;
                in vec4 in_transform_3;
                in vec4 in_color;
                in vec3 in_bark_tint;
                in float in_roughness;

                uniform mat4 u_view;
                uniform mat4 u_proj;
                uniform mat4 u_light_space;

                out vec2 v_local;
                out vec4 v_color;
                out vec2 v_uv;
                out vec3 v_bark_tint;
                out float v_roughness;
                out vec3 v_world_pos;
                out vec4 v_light_space_pos;
                out vec3 v_tangent;
                out vec3 v_bitangent;
                out vec3 v_normal;

                void main() {
                    mat4 transform = mat4(in_transform_0, in_transform_1, in_transform_2, in_transform_3);
                    vec4 world = transform * vec4(in_pos, 1.0);
                    gl_Position = u_proj * u_view * world;
                    v_local = vec2(in_uv.y, 0.0);
                    v_color = in_color;
                    v_uv = in_uv;
                    v_bark_tint = in_bark_tint;
                    v_roughness = in_roughness;
                    v_world_pos = world.xyz;
                    v_light_space_pos = u_light_space * world;
                    mat3 normal_mat = transpose(inverse(mat3(transform)));
                    v_normal = normalize(normal_mat * in_normal);
                    v_tangent = normalize(mat3(transform) * in_tangent);
                    v_bitangent = normalize(cross(v_normal, v_tangent));
                }
            """,
            fragment_shader="""
                #version 330
                in vec2 v_local;
                in vec4 v_color;
                in vec2 v_uv;
                in vec3 v_bark_tint;
                in float v_roughness;
                in vec3 v_world_pos;
                in vec4 v_light_space_pos;
                in vec3 v_tangent;
                in vec3 v_bitangent;
                in vec3 v_normal;

                uniform float u_glow;
                uniform sampler2D u_bark_albedo;
                uniform sampler2D u_bark_normal;
                uniform sampler2D u_bark_roughness;
                uniform vec2 u_bark_uv_scale;
                uniform int u_has_normal;
                uniform int u_has_roughness;
                uniform vec3 u_camera_pos;
                uniform sampler2D u_shadow_map;
                uniform vec2 u_shadow_texel;
                uniform vec3 u_light_dir;
                uniform vec3 u_light_color;
                uniform vec3 u_fog_color;
                uniform float u_fog_density;
                uniform int u_debug_view;

                out vec4 f_color;

                const float PI = 3.14159265;

                float distribution_ggx(vec3 n, vec3 h, float roughness) {
                    float a = roughness * roughness;
                    float a2 = a * a;
                    float n_dot_h = max(dot(n, h), 0.0);
                    float n_dot_h2 = n_dot_h * n_dot_h;
                    float denom = (n_dot_h2 * (a2 - 1.0) + 1.0);
                    return a2 / max(PI * denom * denom, 0.0001);
                }

                float geometry_schlick_ggx(float n_dot_v, float roughness) {
                    float r = roughness + 1.0;
                    float k = (r * r) / 8.0;
                    return n_dot_v / (n_dot_v * (1.0 - k) + k);
                }

                float geometry_smith(vec3 n, vec3 v, vec3 l, float roughness) {
                    float n_dot_v = max(dot(n, v), 0.0);
                    float n_dot_l = max(dot(n, l), 0.0);
                    float ggx_v = geometry_schlick_ggx(n_dot_v, roughness);
                    float ggx_l = geometry_schlick_ggx(n_dot_l, roughness);
                    return ggx_v * ggx_l;
                }

                vec3 fresnel_schlick(float cos_theta, vec3 f0) {
                    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
                }

                vec3 tone_map(vec3 color) {
                    return color / (color + vec3(1.0));
                }

                float shadow_pcf(vec4 light_space_pos, vec3 normal, vec3 light_dir) {
                    vec3 proj = light_space_pos.xyz / light_space_pos.w;
                    proj = proj * 0.5 + 0.5;
                    if (proj.z > 1.0 || proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0) {
                        return 1.0;
                    }
                    float bias = max(0.0015 * (1.0 - dot(normal, light_dir)), 0.0007);
                    float current = proj.z - bias;
                    float shadow = 0.0;
                    for (int x = -1; x <= 1; x++) {
                        for (int y = -1; y <= 1; y++) {
                            vec2 offset = vec2(x, y) * u_shadow_texel;
                            float depth = texture(u_shadow_map, proj.xy + offset).r;
                            shadow += current <= depth ? 1.0 : 0.0;
                        }
                    }
                    return shadow / 9.0;
                }

                void main() {
                    float edge = abs(v_local.y);
                    float core = smoothstep(1.0, 0.0, edge);
                    float glow = exp(-edge * 4.0) * u_glow;
                    vec2 bark_uv = vec2(v_uv.x * u_bark_uv_scale.x, v_uv.y * u_bark_uv_scale.y);
                    // Decode sRGB texture to linear
                    vec3 albedo = pow(texture(u_bark_albedo, bark_uv).rgb, vec3(2.2)) * v_bark_tint;
                    vec3 normal = normalize(v_normal);
                    if (u_has_normal == 1) {
                        vec3 tangent_normal = texture(u_bark_normal, bark_uv).xyz * 2.0 - 1.0;
                        mat3 tbn = mat3(normalize(v_tangent), normalize(v_bitangent), normalize(v_normal));
                        normal = normalize(tbn * tangent_normal);
                    }

                    // Debug view modes
                    if (u_debug_view == 1) {
                        // Albedo only (gamma corrected for display)
                        vec3 display = pow(albedo, vec3(1.0 / 2.2));
                        f_color = vec4(display, 1.0);
                        return;
                    } else if (u_debug_view == 2) {
                        // Normals visualization (map from [-1,1] to [0,1] for display)
                        vec3 normal_display = normal * 0.5 + 0.5;
                        f_color = vec4(normal_display, 1.0);
                        return;
                    }

                    float roughness = v_roughness;
                    if (u_has_roughness == 1) {
                        roughness = clamp(v_roughness * (0.6 + 0.8 * texture(u_bark_roughness, bark_uv).r), 0.05, 1.0);
                    }
                    float occlusion = mix(0.65, 1.0, core);
                    vec3 view_dir = normalize(u_camera_pos - v_world_pos);
                    vec3 light_dir = normalize(u_light_dir);
                    vec3 half_dir = normalize(view_dir + light_dir);
                    float n_dot_l = max(dot(normal, light_dir), 0.0);
                    float n_dot_v = max(dot(normal, view_dir), 0.0);
                    vec3 f0 = vec3(0.04);
                    vec3 fresnel = fresnel_schlick(max(dot(half_dir, view_dir), 0.0), f0);
                    float d = distribution_ggx(normal, half_dir, roughness);
                    float g = geometry_smith(normal, view_dir, light_dir, roughness);
                    vec3 specular = (d * g * fresnel) / max(4.0 * n_dot_v * n_dot_l, 0.001);
                    vec3 kd = vec3(1.0) - fresnel;
                    vec3 diffuse = kd * albedo / PI;
                    float shadow = shadow_pcf(v_light_space_pos, normal, light_dir);
                    vec3 radiance = u_light_color;
                    vec3 lighting = (diffuse + specular) * radiance * n_dot_l * shadow;
                    vec3 ambient = albedo * 0.18;
                    vec3 lit = (ambient + lighting) * occlusion;
                    lit *= v_color.rgb;

                    // Apply fog only if not in debug mode 3 (final no fog)
                    if (u_debug_view != 3) {
                        float dist = length(u_camera_pos - v_world_pos);
                        float fog = 1.0 - exp(-u_fog_density * dist);
                        lit = mix(lit, u_fog_color, clamp(fog, 0.0, 1.0));
                    }

                    lit = tone_map(lit);
                    lit = pow(lit, vec3(1.0 / 2.2));
                    float alpha = (v_color.a * core) + glow;
                    f_color = vec4(lit, alpha);
                }
            """,
        )

        # Generate all textures procedurally
        seed = random.randint(0, 100000)

        # Bark textures
        print("Generating bark albedo...")
        bark_albedo_data = generate_bark_albedo(size=256, seed=seed)
        # Convert RGB to RGBA for moderngl
        bark_albedo_rgba = np.concatenate([bark_albedo_data, np.full((256, 256, 1), 255, dtype=np.uint8)], axis=2)
        self.bark_albedo = self._create_texture_from_array(bark_albedo_rgba, repeat=True, mipmaps=True)

        print("Generating bark normal...")
        bark_normal_data = generate_bark_normal(size=256, seed=seed)
        # Convert RGB to RGBA for moderngl
        bark_normal_rgba = np.concatenate([bark_normal_data, np.full((256, 256, 1), 255, dtype=np.uint8)], axis=2)
        self.bark_normal = self._create_texture_from_array(bark_normal_rgba, repeat=True, mipmaps=True)

        print("Generating bark roughness...")
        bark_roughness_data = generate_bark_roughness(size=256, seed=seed)
        # Convert to RGBA (replicate grayscale to RGB, then add alpha)
        bark_roughness_rgb = np.repeat(bark_roughness_data, 3, axis=2)
        bark_roughness_rgba = np.concatenate([bark_roughness_rgb, np.full((256, 256, 1), 255, dtype=np.uint8)], axis=2)
        self.bark_roughness = self._create_texture_from_array(bark_roughness_rgba, repeat=True, mipmaps=True)

        self.has_bark_normal = True
        self.has_bark_roughness = True

        # Leaf atlas (2x2 grid with 2 variants)
        print("Generating leaf atlas...")
        leaf_size = 128
        atlas_size = 256  # 2x2 grid of 128x128 leaves
        leaf_atlas_data = np.zeros((atlas_size, atlas_size, 4), dtype=np.uint8)

        # Generate 4 leaf variants (2 primary variants, repeated)
        leaf1 = generate_leaf_rgba(size=leaf_size, seed=seed, variant=0)
        leaf2 = generate_leaf_rgba(size=leaf_size, seed=seed, variant=1)

        # Fill 2x2 atlas
        leaf_atlas_data[0:leaf_size, 0:leaf_size] = leaf1
        leaf_atlas_data[0:leaf_size, leaf_size:atlas_size] = leaf2
        leaf_atlas_data[leaf_size:atlas_size, 0:leaf_size] = leaf1  # Repeat variant 1
        leaf_atlas_data[leaf_size:atlas_size, leaf_size:atlas_size] = leaf2  # Repeat variant 2

        # Avoid mipmap alpha bleed that can erase tiny leaf cutouts at distance
        self.leaf_atlas = self._create_texture_from_array(leaf_atlas_data, repeat=False, mipmaps=False)
        # Update tree's atlas dimensions for shader uniforms
        tree.leaf_atlas_cols = 2
        tree.leaf_atlas_rows = 2

        # Grass textures
        print("Generating grass albedo...")
        grass_albedo_data = generate_grass_albedo(size=512, seed=seed)
        grass_albedo_rgba = np.concatenate([grass_albedo_data, np.full((512, 512, 1), 255, dtype=np.uint8)], axis=2)
        self.grass_albedo = self._create_texture_from_array(grass_albedo_rgba, repeat=True, mipmaps=True)

        print("Generating grass roughness...")
        grass_roughness_data = generate_grass_roughness(size=512, seed=seed)
        grass_roughness_rgb = np.repeat(grass_roughness_data, 3, axis=2)
        grass_roughness_rgba = np.concatenate([grass_roughness_rgb, np.full((512, 512, 1), 255, dtype=np.uint8)], axis=2)
        self.grass_roughness = self._create_texture_from_array(grass_roughness_rgba, repeat=True, mipmaps=True)

        # Bird sprite (simple fallback - not the focus)
        bird_fallback = np.full((4, 4, 4), [255, 255, 255, 0], dtype=np.uint8)
        self.bird_sprite = self._create_texture_from_array(bird_fallback, repeat=False, mipmaps=False)

        print("All textures generated successfully!")

        self.leaf_program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                in float in_branch_index;
                in vec3 in_offset;
                in vec3 in_axis_x;
                in vec3 in_axis_y;
                in vec3 in_axis_z;
                in vec2 in_uv_offset;
                in vec2 in_uv_scale;
                in float in_atlas_index;
                in vec4 in_color;
                in vec4 in_variance;

                uniform mat4 u_view;
                uniform mat4 u_proj;
                uniform mat4 u_light_space;
                uniform sampler2D u_branch_positions;  // GPU texture with branch positions

                out vec2 v_uv;
                out float v_atlas_index;
                out vec4 v_color;
                out vec4 v_variance;
                out vec3 v_normal;
                out vec3 v_world_pos;
                out vec4 v_light_space_pos;

                void main() {
                    // Look up branch position from GPU buffer
                    int branch_idx = int(in_branch_index);
                    ivec2 size = textureSize(u_branch_positions, 0);
                    int max_index = max(size.x - 1, 0);
                    int safe_idx = clamp(branch_idx, 0, max_index);
                    vec3 branch_pos = texelFetch(u_branch_positions, ivec2(safe_idx, 0), 0).xyz;

                    // Calculate final leaf position
                    vec3 leaf_center = branch_pos + in_offset;
                    vec3 offset = in_axis_x * in_pos.x + in_axis_y * in_pos.y;
                    vec4 world = vec4(leaf_center + offset, 1.0);

                    gl_Position = u_proj * u_view * world;
                    vec2 base_uv = in_pos * 0.5 + 0.5;
                    v_uv = in_uv_offset + base_uv * in_uv_scale;
                    v_atlas_index = in_atlas_index;
                    v_color = in_color;
                    v_variance = in_variance;
                    v_normal = normalize(in_axis_z);
                    v_world_pos = world.xyz;
                    v_light_space_pos = u_light_space * world;
                }
            """,
            fragment_shader="""
                #version 330
                in vec2 v_uv;
                in float v_atlas_index;
                in vec4 v_color;
                in vec4 v_variance;
                in vec3 v_normal;
                in vec3 v_world_pos;
                in vec4 v_light_space_pos;

                uniform sampler2D u_leaf_atlas;
                uniform vec2 u_atlas_grid;
                uniform float u_alpha_cutoff;
                uniform sampler2D u_shadow_map;
                uniform vec2 u_shadow_texel;
                uniform vec3 u_light_dir;
                uniform vec3 u_light_color;
                uniform vec3 u_camera_pos;
                uniform vec3 u_fog_color;
                uniform float u_fog_density;
                uniform int u_debug_view;

                out vec4 f_color;

                vec3 tone_map(vec3 color) {
                    // Reinhard tone mapping
                    return color / (color + vec3(1.0));
                }

                float shadow_pcf(vec4 light_space_pos, vec3 normal, vec3 light_dir) {
                    vec3 proj = light_space_pos.xyz / light_space_pos.w;
                    proj = proj * 0.5 + 0.5;
                    if (proj.z > 1.0 || proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0) {
                        return 1.0;
                    }
                    float bias = max(0.0015 * (1.0 - dot(normal, light_dir)), 0.0007);
                    float current = proj.z - bias;
                    float shadow = 0.0;
                    for (int x = -1; x <= 1; x++) {
                        for (int y = -1; y <= 1; y++) {
                            vec2 offset = vec2(x, y) * u_shadow_texel;
                            float depth = texture(u_shadow_map, proj.xy + offset).r;
                            shadow += current <= depth ? 1.0 : 0.0;
                        }
                    }
                    return shadow / 9.0;
                }

                void main() {
                    // Sample leaf texture
                    vec2 grid = max(u_atlas_grid, vec2(1.0));
                    float index = max(v_atlas_index, 0.0);
                    vec2 cell = vec2(mod(index, grid.x), floor(index / grid.x));
                    vec2 atlas_uv = (cell + clamp(v_uv, 0.0, 1.0)) / grid;
                    vec4 tex = texture(u_leaf_atlas, atlas_uv);

                    // Alpha cutout (discard if below threshold)
                    if (tex.a < u_alpha_cutoff) {
                        discard;
                    }

                    // Decode sRGB texture to linear
                    vec3 tex_linear = pow(tex.rgb, vec3(2.2));

                    // Apply color variance
                    vec3 variance = v_variance.xyz * v_variance.w;
                    vec3 tinted = clamp(v_color.rgb * (1.0 + variance), 0.0, 1.0);
                    vec3 albedo = tinted * tex_linear;

                    // Two-sided lighting: flip normal if facing away from camera
                    vec3 normal = normalize(v_normal);
                    vec3 view_dir = normalize(u_camera_pos - v_world_pos);
                    if (dot(normal, view_dir) < 0.0) {
                        normal = -normal;
                    }

                    // Debug view modes
                    if (u_debug_view == 1) {
                        // Albedo only (gamma corrected for display)
                        vec3 display = pow(albedo, vec3(1.0 / 2.2));
                        f_color = vec4(display, 1.0);
                        return;
                    } else if (u_debug_view == 2) {
                        // Normals visualization (map from [-1,1] to [0,1] for display)
                        vec3 normal_display = normal * 0.5 + 0.5;
                        f_color = vec4(normal_display, 1.0);
                        return;
                    }

                    vec3 light_dir = normalize(u_light_dir);
                    float shadow = shadow_pcf(v_light_space_pos, normal, light_dir);

                    // Diffuse lighting
                    float ndotl = dot(normal, light_dir);
                    float diff = max(ndotl, 0.0);

                    // Translucency/backlighting (when light is behind the leaf)
                    float backlight = clamp(-ndotl, 0.0, 1.0);
                    vec3 translucent = albedo * backlight * 0.35;

                    // Hemispheric ambient based on normal.y
                    float sky_factor = normal.y * 0.5 + 0.5;
                    vec3 ambient = mix(vec3(0.15, 0.12, 0.1), vec3(0.3, 0.35, 0.4), sky_factor);

                    // Combine lighting
                    vec3 direct = albedo * diff * shadow * u_light_color;
                    vec3 rgb = ambient * albedo + direct + translucent;

                    // Apply fog only if not in debug mode 3 (final no fog)
                    if (u_debug_view != 3) {
                        float dist = length(u_camera_pos - v_world_pos);
                        float fog = 1.0 - exp(-u_fog_density * dist);
                        rgb = mix(rgb, u_fog_color, clamp(fog, 0.0, 1.0));
                    }

                    // Tone mapping
                    rgb = tone_map(rgb);

                    // Gamma correction (linear to sRGB)
                    rgb = pow(rgb, vec3(1.0 / 2.2));

                    f_color = vec4(rgb, 1.0);
                }
            """,
        )

        self.ground_program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                in vec3 in_normal;
                in vec2 in_uv;

                uniform mat4 u_view;
                uniform mat4 u_proj;
                uniform mat4 u_light_space;

                out vec3 v_world_pos;
                out vec3 v_normal;
                out vec2 v_uv;
                out vec4 v_light_space_pos;

                void main() {
                    vec4 world = vec4(in_pos, 1.0);
                    gl_Position = u_proj * u_view * world;
                    v_world_pos = world.xyz;
                    v_normal = normalize(in_normal);
                    v_uv = in_uv;
                    v_light_space_pos = u_light_space * world;
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_world_pos;
                in vec3 v_normal;
                in vec2 v_uv;
                in vec4 v_light_space_pos;

                uniform sampler2D u_grass_albedo;
                uniform sampler2D u_shadow_map;
                uniform vec2 u_shadow_texel;
                uniform vec3 u_light_dir;
                uniform vec3 u_light_color;
                uniform vec3 u_camera_pos;
                uniform vec3 u_fog_color;
                uniform float u_fog_density;
                uniform vec2 u_tree_center;
                uniform int u_debug_view;

                out vec4 f_color;

                vec3 tone_map(vec3 color) {
                    return color / (color + vec3(1.0));
                }

                float shadow_pcf(vec4 light_space_pos, vec3 normal, vec3 light_dir) {
                    vec3 proj = light_space_pos.xyz / light_space_pos.w;
                    proj = proj * 0.5 + 0.5;
                    if (proj.z > 1.0 || proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0) {
                        return 1.0;
                    }
                    float bias = max(0.0015 * (1.0 - dot(normal, light_dir)), 0.0007);
                    float current = proj.z - bias;
                    float shadow = 0.0;
                    for (int x = -1; x <= 1; x++) {
                        for (int y = -1; y <= 1; y++) {
                            vec2 offset = vec2(x, y) * u_shadow_texel;
                            float depth = texture(u_shadow_map, proj.xy + offset).r;
                            shadow += current <= depth ? 1.0 : 0.0;
                        }
                    }
                    return shadow / 9.0;
                }

                void main() {
                    // Sample grass texture
                    vec2 grass_uv = v_uv * 8.0;
                    vec4 grass_tex = texture(u_grass_albedo, grass_uv);

                    // Decode sRGB to linear
                    vec3 albedo = pow(grass_tex.rgb, vec3(2.2));

                    vec3 normal = normalize(v_normal);

                    // Debug view modes
                    if (u_debug_view == 1) {
                        // Albedo only (gamma corrected for display)
                        vec3 display = pow(albedo, vec3(1.0 / 2.2));
                        f_color = vec4(display, 1.0);
                        return;
                    } else if (u_debug_view == 2) {
                        // Normals visualization (map from [-1,1] to [0,1] for display)
                        vec3 normal_display = normal * 0.5 + 0.5;
                        f_color = vec4(normal_display, 1.0);
                        return;
                    }

                    // Contact shadow under tree (soft radial falloff)
                    vec2 tree_offset = v_world_pos.xy - u_tree_center;
                    float dist_to_tree = length(tree_offset);
                    float contact_shadow = smoothstep(80.0, 200.0, dist_to_tree);
                    contact_shadow = mix(0.4, 1.0, contact_shadow);

                    // Lighting
                    vec3 light_dir = normalize(u_light_dir);
                    float diff = max(dot(normal, light_dir), 0.0);
                    float shadow = shadow_pcf(v_light_space_pos, normal, light_dir);

                    // Ambient
                    vec3 ambient = albedo * vec3(0.25, 0.28, 0.32);

                    // Direct lighting
                    vec3 direct = albedo * diff * shadow * u_light_color;

                    // Combine with contact shadow
                    vec3 rgb = (ambient + direct) * contact_shadow;

                    // Apply fog only if not in debug mode 3 (final no fog)
                    if (u_debug_view != 3) {
                        float dist = length(u_camera_pos - v_world_pos);
                        float fog = 1.0 - exp(-u_fog_density * dist);
                        rgb = mix(rgb, u_fog_color, clamp(fog, 0.0, 1.0));
                    }

                    // Tone mapping
                    rgb = tone_map(rgb);

                    // Gamma correction
                    rgb = pow(rgb, vec3(1.0 / 2.2));

                    f_color = vec4(rgb, 1.0);
                }
            """,
        )

        self.sky_program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                out vec2 v_uv;

                void main() {
                    gl_Position = vec4(in_pos, 0.0, 1.0);
                    v_uv = in_pos * 0.5 + 0.5;
                }
            """,
            fragment_shader="""
                #version 330
                in vec2 v_uv;
                uniform vec3 u_sky_top;
                uniform vec3 u_sky_bottom;
                out vec4 f_color;

                void main() {
                    float t = smoothstep(0.0, 1.0, v_uv.y);
                    vec3 color = mix(u_sky_bottom, u_sky_top, t);
                    f_color = vec4(color, 1.0);
                }
            """,
        )

        self.branch_shadow_program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                in vec4 in_transform_0;
                in vec4 in_transform_1;
                in vec4 in_transform_2;
                in vec4 in_transform_3;

                uniform mat4 u_light_space;

                void main() {
                    mat4 transform = mat4(in_transform_0, in_transform_1, in_transform_2, in_transform_3);
                    vec4 world = transform * vec4(in_pos, 1.0);
                    gl_Position = u_light_space * world;
                }
            """,
            fragment_shader="""
                #version 330
                void main() {
                }
            """,
        )

        self.leaf_shadow_program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                in float in_branch_index;
                in vec3 in_offset;
                in vec3 in_axis_x;
                in vec3 in_axis_y;
                in vec2 in_uv_offset;
                in vec2 in_uv_scale;
                in float in_atlas_index;

                uniform mat4 u_light_space;
                uniform sampler2D u_branch_positions;

                out vec2 v_uv;
                out float v_atlas_index;

                void main() {
                    // Look up branch position from GPU buffer
                    int branch_idx = int(in_branch_index);
                    ivec2 size = textureSize(u_branch_positions, 0);
                    int max_index = max(size.x - 1, 0);
                    int safe_idx = clamp(branch_idx, 0, max_index);
                    vec3 branch_pos = texelFetch(u_branch_positions, ivec2(safe_idx, 0), 0).xyz;

                    // Calculate final leaf position
                    vec3 leaf_center = branch_pos + in_offset;
                    vec3 offset = in_axis_x * in_pos.x + in_axis_y * in_pos.y;
                    vec4 world = vec4(leaf_center + offset, 1.0);

                    gl_Position = u_light_space * world;
                    vec2 base_uv = in_pos * 0.5 + 0.5;
                    v_uv = in_uv_offset + base_uv * in_uv_scale;
                    v_atlas_index = in_atlas_index;
                }
            """,
            fragment_shader="""
                #version 330
                in vec2 v_uv;
                in float v_atlas_index;

                uniform sampler2D u_leaf_atlas;
                uniform vec2 u_atlas_grid;
                uniform float u_alpha_cutoff;

                void main() {
                    vec2 grid = max(u_atlas_grid, vec2(1.0));
                    float index = max(v_atlas_index, 0.0);
                    vec2 cell = vec2(mod(index, grid.x), floor(index / grid.x));
                    vec2 atlas_uv = (cell + clamp(v_uv, 0.0, 1.0)) / grid;
                    float alpha = texture(u_leaf_atlas, atlas_uv).a;
                    if (alpha < u_alpha_cutoff) {
                        discard;
                    }
                }
            """,
        )

        self.ground_shadow_program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec3 in_pos;
                in vec3 in_normal;
                in vec2 in_uv;
                uniform mat4 u_light_space;

                void main() {
                    gl_Position = u_light_space * vec4(in_pos, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                void main() {
                }
            """,
        )

        self.bird_program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                in vec2 in_center;
                in float in_size;
                in float in_facing;
                in float in_flap;
                in vec2 in_uv_offset;
                in vec2 in_uv_scale;
                in float in_frame;
                in vec4 in_body_color;
                in vec4 in_wing_color;
                in vec4 in_beak_color;

                uniform vec2 u_resolution;

                out vec2 v_uv;
                out float v_facing;
                out float v_flap;
                out float v_frame;
                out vec4 v_body_color;
                out vec4 v_wing_color;
                out vec4 v_beak_color;

                void main() {
                    vec2 scaled = in_pos * vec2(in_size * 1.6, in_size);
                    vec2 world = in_center + scaled;
                    vec2 ndc = vec2((world.x / u_resolution.x) * 2.0 - 1.0,
                                    (world.y / u_resolution.y) * 2.0 - 1.0);
                    gl_Position = vec4(ndc, 0.0, 1.0);
                    v_facing = in_facing;
                    v_flap = in_flap;
                    vec2 base_uv = in_pos * 0.5 + 0.5;
                    if (in_facing < 0.0) {
                        base_uv.x = 1.0 - base_uv.x;
                    }
                    v_uv = in_uv_offset + base_uv * in_uv_scale;
                    v_frame = in_frame;
                    v_body_color = in_body_color;
                    v_wing_color = in_wing_color;
                    v_beak_color = in_beak_color;
                }
            """,
            fragment_shader="""
                #version 330
                in vec2 v_uv;
                in float v_facing;
                in float v_flap;
                in float v_frame;
                in vec4 v_body_color;
                in vec4 v_wing_color;
                in vec4 v_beak_color;

                uniform sampler2D u_bird_sprite;
                uniform vec2 u_sprite_grid;
                uniform float u_alpha_cutoff;

                out vec4 f_color;

                void main() {
                    vec2 grid = max(u_sprite_grid, vec2(1.0));
                    float index = max(v_frame, 0.0);
                    vec2 cell = vec2(mod(index, grid.x), floor(index / grid.x));
                    vec2 atlas_uv = (cell + clamp(v_uv, 0.0, 1.0)) / grid;
                    vec4 tex = texture(u_bird_sprite, atlas_uv);
                    float alpha = tex.a * v_body_color.a;
                    if (alpha < u_alpha_cutoff) {
                        discard;
                    }
                    vec3 normal = normalize(vec3((v_uv - 0.5) * vec2(0.6, 0.4), 1.0));
                    vec3 light_dir = normalize(vec3(0.35, 0.4, 0.85));
                    float diff = max(dot(normal, light_dir), 0.0);
                    float flap = clamp(abs(v_flap), 0.0, 1.0);
                    vec3 tint = mix(v_body_color.rgb, v_wing_color.rgb, 0.2 + flap * 0.1);
                    tint = mix(tint, v_beak_color.rgb, 0.08);
                    vec3 lit = tex.rgb * tint * (0.75 + diff * 0.25);
                    lit *= alpha;
                    f_color = vec4(lit, alpha);
                }
            """,
        )

        self.ui_line_program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                in vec2 in_start;
                in vec2 in_end;
                in float in_thickness;
                in vec4 in_color;

                uniform vec2 u_resolution;

                out vec2 v_local;
                out vec4 v_color;

                void main() {
                    vec2 dir = in_end - in_start;
                    float len = length(dir);
                    if (len < 0.0001) {
                        len = 0.0001;
                    }
                    vec2 dir_n = dir / len;
                    vec2 normal = vec2(-dir_n.y, dir_n.x);
                    float half_thick = in_thickness * 0.5;
                    float along = (in_pos.x + 1.0) * 0.5 * len;
                    vec2 world = in_start + dir_n * along + normal * (in_pos.y * half_thick);
                    vec2 ndc = vec2((world.x / u_resolution.x) * 2.0 - 1.0,
                                    (world.y / u_resolution.y) * 2.0 - 1.0);
                    gl_Position = vec4(ndc, 0.0, 1.0);
                    v_local = vec2((in_pos.x + 1.0) * 0.5, in_pos.y);
                    v_color = in_color;
                }
            """,
            fragment_shader="""
                #version 330
                in vec2 v_local;
                in vec4 v_color;

                out vec4 f_color;

                void main() {
                    float edge = abs(v_local.y);
                    float core = smoothstep(1.0, 0.0, edge);
                    float alpha = v_color.a * core;
                    f_color = vec4(v_color.rgb, alpha);
                }
            """,
        )

        quad = [
            -1.0, -1.0,
            1.0, -1.0,
            1.0, 1.0,
            -1.0, -1.0,
            1.0, 1.0,
            -1.0, 1.0,
        ]
        cylinder = self._build_cylinder_mesh()
        self.cylinder_vbo = self.ctx.buffer(data=array('f', cylinder))
        self.quad_vbo = self.ctx.buffer(data=array('f', quad))

        ground_size = 2200.0
        ground_y = 0.0
        ground = [
            -ground_size, ground_y, -ground_size, 0.0, 1.0, 0.0, 0.0, 0.0,
            ground_size, ground_y, -ground_size, 0.0, 1.0, 0.0, 1.0, 0.0,
            ground_size, ground_y, ground_size, 0.0, 1.0, 0.0, 1.0, 1.0,
            -ground_size, ground_y, -ground_size, 0.0, 1.0, 0.0, 0.0, 0.0,
            ground_size, ground_y, ground_size, 0.0, 1.0, 0.0, 1.0, 1.0,
            -ground_size, ground_y, ground_size, 0.0, 1.0, 0.0, 0.0, 1.0,
        ]
        self.ground_vbo = self.ctx.buffer(data=array('f', ground))

        self.branch_instance_vbo = self.ctx.buffer(reserve=4 * 1024 * 96)
        self.leaf_instance_vbo = self.ctx.buffer(reserve=4 * 1024 * 128)
        self.bird_instance_vbo = self.ctx.buffer(reserve=4 * 64)
        self.ui_line_instance_vbo = self.ctx.buffer(reserve=4 * 64)

        # GPU-driven: Branch position buffer (TBO) for leaves
        # Stores vec3 position for each branch
        self.max_branches = 2048
        # Create texture for shader access via texelFetch
        self.branch_position_tbo = self.ctx.texture(
            (self.max_branches, 1),
            components=3,  # RGB for XYZ
            dtype="f4",
        )

        self.branch_vao = self.ctx.vertex_array(
            self.branch_program,
            [
                (self.cylinder_vbo, "3f 3f 2f 3f", "in_pos", "in_normal", "in_uv", "in_tangent"),
                (self.branch_instance_vbo, "4f 4f 4f 4f 4f 3f f /i",
                 "in_transform_0", "in_transform_1", "in_transform_2", "in_transform_3", "in_color",
                 "in_bark_tint", "in_roughness"),
            ],
        )

        self.leaf_vao = self.ctx.vertex_array(
            self.leaf_program,
            [
                (self.quad_vbo, "2f", "in_pos"),
                (self.leaf_instance_vbo, "f 3f 3f 3f 3f 2f 2f f 4f 4f /i",
                 "in_branch_index", "in_offset", "in_axis_x", "in_axis_y", "in_axis_z",
                 "in_uv_offset", "in_uv_scale",
                 "in_atlas_index", "in_color", "in_variance"),
            ],
        )

        self.ground_vao = self.ctx.vertex_array(
            self.ground_program,
            [(self.ground_vbo, "3f 3f 2f", "in_pos", "in_normal", "in_uv")],
        )

        self.sky_vao = self.ctx.vertex_array(
            self.sky_program,
            [(self.quad_vbo, "2f", "in_pos")],
        )

        self.branch_shadow_vao = self.ctx.vertex_array(
            self.branch_shadow_program,
            [
                (self.cylinder_vbo, "3f 32x", "in_pos"),
                (self.branch_instance_vbo, "4f 4f 4f 4f 32x /i",
                 "in_transform_0", "in_transform_1", "in_transform_2", "in_transform_3"),
            ],
        )

        self.leaf_shadow_vao = self.ctx.vertex_array(
            self.leaf_shadow_program,
            [
                (self.quad_vbo, "2f", "in_pos"),
                (self.leaf_instance_vbo, "f 3f 3f 3f 12x 2f 2f f 32x /i",
                 "in_branch_index", "in_offset", "in_axis_x", "in_axis_y",
                 "in_uv_offset", "in_uv_scale",
                 "in_atlas_index"),
            ],
        )

        self.ground_shadow_vao = self.ctx.vertex_array(
            self.ground_shadow_program,
            [(self.ground_vbo, "3f 20x", "in_pos")],
        )

        self.bird_vao = self.ctx.vertex_array(
            self.bird_program,
            [
                (self.quad_vbo, "2f", "in_pos"),
                (self.bird_instance_vbo, "2f f f f 2f 2f f 4f 4f 4f /i",
                 "in_center", "in_size", "in_facing", "in_flap",
                 "in_uv_offset", "in_uv_scale", "in_frame",
                 "in_body_color", "in_wing_color", "in_beak_color"),
            ],
        )

        self.ui_line_vao = self.ctx.vertex_array(
            self.ui_line_program,
            [
                (self.quad_vbo, "2f", "in_pos"),
                (self.ui_line_instance_vbo, "2f 2f f 4f /i",
                 "in_start", "in_end", "in_thickness", "in_color"),
            ],
        )

        # Build static leaf data once (GPU-driven approach)
        self._upload_static_leaf_data()
        self.leaf_count = len(self.tree.canopy_leaves)
        print(f"Uploaded {self.leaf_count} static leaves to GPU")

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

        branch_instances = self.tree.build_branch_instances()
        if branch_instances:
            data = array('f', branch_instances)
            self.branch_instance_vbo.orphan(len(data) * 4)
            self.branch_instance_vbo.write(data)

        # GPU-driven: Update branch positions only (not all leaf data!)
        branch_positions: List[float] = []
        for branch in self.tree.branches:
            branch_positions.extend([branch.end_x, branch.end_y, branch.end_z])

        if branch_positions:
            branch_count = min(len(branch_positions) // 3, self.max_branches)
            data = array('f', branch_positions[:branch_count * 3])
            self.branch_position_tbo.write(data.tobytes(), viewport=(0, 0, branch_count, 1))

        # Alpha-to-coverage for leaves (if available)
        if self.use_alpha_to_coverage:
            self.ctx.enable(self.alpha_to_coverage_flag)
        else:
            if self.alpha_to_coverage_flag is not None:
                self.ctx.disable(self.alpha_to_coverage_flag)

        self.shadow_fbo.use()
        self.ctx.viewport = (0, 0, self.shadow_size, self.shadow_size)
        self.ctx.clear(depth=1.0)
        if branch_instances:
            self.branch_shadow_program["u_light_space"].write(light_space)
            self.branch_shadow_vao.render(instances=len(branch_instances) // 24)
        if self.leaf_count > 0:
            # Bind branch position TBO for GPU lookup
            self.branch_position_tbo.use(location=4)
            self.leaf_shadow_program["u_branch_positions"].value = 4
            self.leaf_shadow_program["u_light_space"].write(light_space)
            self.leaf_shadow_program["u_atlas_grid"].value = (self.tree.leaf_atlas_cols, self.tree.leaf_atlas_rows)
            self.leaf_shadow_program["u_alpha_cutoff"].value = 0.2
            self.leaf_atlas.use(location=0)
            self.leaf_shadow_program["u_leaf_atlas"].value = 0
            self.leaf_shadow_vao.render(instances=self.leaf_count)
        self.ground_shadow_program["u_light_space"].write(light_space)
        self.ground_shadow_vao.render()

        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.clear(0.015, 0.05, 0.11, depth=1.0)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.sky_program["u_sky_top"].value = (0.18, 0.28, 0.5)
        self.sky_program["u_sky_bottom"].value = self.fog_color
        self.sky_vao.render()
        self.ctx.enable(moderngl.DEPTH_TEST)

        self.shadow_map.use(location=3)
        shadow_texel = (1.0 / self.shadow_size, 1.0 / self.shadow_size)

        # Bind grass texture for ground
        self.grass_albedo.use(location=0)

        self.ground_program["u_grass_albedo"].value = 0
        self.ground_program["u_view"].write(view)
        self.ground_program["u_proj"].write(proj)
        self.ground_program["u_light_space"].write(light_space)
        self.ground_program["u_shadow_map"].value = 3
        self.ground_program["u_shadow_texel"].value = shadow_texel
        self.ground_program["u_light_dir"].value = self.light_direction
        self.ground_program["u_light_color"].value = self.light_color
        self.ground_program["u_camera_pos"].value = self.camera_position
        self.ground_program["u_fog_color"].value = self.fog_color
        self.ground_program["u_fog_density"].value = self.fog_density
        self.ground_program["u_tree_center"].value = (self.tree.root_x, self.tree.root_y)
        self.ground_program["u_debug_view"].value = self.debug_view_mode
        self.ground_vao.render()

        if branch_instances:
            self.branch_program["u_view"].write(view)
            self.branch_program["u_proj"].write(proj)
            self.branch_program["u_light_space"].write(light_space)
            self.branch_program["u_camera_pos"].value = self.camera_position
            self.branch_program["u_bark_uv_scale"].value = (0.05, 1.0)
            self.branch_program["u_has_normal"].value = 1 if self.has_bark_normal else 0
            self.branch_program["u_has_roughness"].value = 1 if self.has_bark_roughness else 0
            self.bark_albedo.use(location=0)
            self.bark_normal.use(location=1)
            self.bark_roughness.use(location=2)
            self.branch_program["u_bark_albedo"].value = 0
            self.branch_program["u_bark_normal"].value = 1
            self.branch_program["u_bark_roughness"].value = 2
            self.branch_program["u_shadow_map"].value = 3
            self.branch_program["u_shadow_texel"].value = shadow_texel
            self.branch_program["u_light_dir"].value = self.light_direction
            self.branch_program["u_light_color"].value = self.light_color
            self.branch_program["u_fog_color"].value = self.fog_color
            self.branch_program["u_fog_density"].value = self.fog_density
            self.branch_program["u_debug_view"].value = self.debug_view_mode

            self.branch_program["u_glow"].value = 0.0
            self.branch_vao.render(instances=len(branch_instances) // 24)

        if self.leaf_count > 0:
            # Disable blending for alpha cutout rendering (depth write ON, blending OFF)
            self.ctx.disable(moderngl.BLEND)

            # Bind branch position TBO for GPU lookup
            self.branch_position_tbo.use(location=4)
            self.leaf_program["u_branch_positions"].value = 4
            self.leaf_program["u_view"].write(view)
            self.leaf_program["u_proj"].write(proj)
            self.leaf_program["u_light_space"].write(light_space)
            self.leaf_program["u_atlas_grid"].value = (self.tree.leaf_atlas_cols, self.tree.leaf_atlas_rows)
            self.leaf_program["u_alpha_cutoff"].value = 0.2
            self.leaf_atlas.use(location=0)
            self.leaf_program["u_leaf_atlas"].value = 0
            self.leaf_program["u_shadow_map"].value = 3
            self.leaf_program["u_shadow_texel"].value = shadow_texel
            self.leaf_program["u_light_dir"].value = self.light_direction
            self.leaf_program["u_light_color"].value = self.light_color
            self.leaf_program["u_camera_pos"].value = self.camera_position
            self.leaf_program["u_fog_color"].value = self.fog_color
            self.leaf_program["u_fog_density"].value = self.fog_density
            self.leaf_program["u_debug_view"].value = self.debug_view_mode
            self.leaf_vao.render(instances=self.leaf_count)
            if self.use_alpha_to_coverage:
                self.ctx.disable(self.alpha_to_coverage_flag)

            # Re-enable blending for subsequent rendering
            self.ctx.enable(moderngl.BLEND)

        self.ctx.disable(moderngl.DEPTH_TEST)
        bird_instances = self.tree.build_bird_instances()
        if bird_instances:
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
        data = array('f', ui_instances)
        self.ui_line_instance_vbo.orphan(len(data) * 4)
        self.ui_line_instance_vbo.write(data)
        self.ui_line_program["u_resolution"].value = (width, height)
        self.ui_line_vao.render(instances=len(ui_instances) // 9)


class HolographicWindow(pyglet.window.Window):
    def __init__(self):
        display = pyglet.display.get_display()
        screen = display.get_default_screen()
        config = pyglet.gl.Config(double_buffer=True, sample_buffers=1, samples=4)
        super().__init__(
            width=screen.width,
            height=screen.height,
            fullscreen=True,
            caption="Holographic Tree",
            config=config,
        )
        self.set_mouse_visible(False)

        self.tree = HolographicTree(self.width, self.height)
        self.renderer = OpenGLRenderer(self, self.tree)
        self.info_batch = pyglet.graphics.Batch()
        self.avg_fps = 0.0
        self._init_labels()

        self.last_time = time.perf_counter()
        self.frame_times: List[float] = []
        pyglet.clock.schedule_interval(self.update, 1 / 120.0)

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
    window = HolographicWindow()
    pyglet.app.run()


if __name__ == "__main__":
    main()
