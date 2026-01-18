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
        b = Branch(parent, angle, length, max(2, (max_d - depth) * 2.5),
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
        spread = 24 + depth * 2
        ratio = 0.72
        self._gen_branch(idx, angle - spread, length * ratio, depth + 1, max_d, nz - 0.05)
        self._gen_branch(idx, angle + spread, length * ratio, depth + 1, max_d, nz + 0.05)
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
        self.canopy_leaves = []
        for idx in self.tip_indices:
            branch = self.branches[idx]
            zf = (branch.z_depth + 1) / 2
            base_color = (0.6 + zf * 0.4, 0.95, 1.0, 0.55)
            cluster_radius = (
                random.uniform(16, 26),
                random.uniform(12, 22),
                random.uniform(12, 24),
            )
            count = random.randint(20, 60)
            for i in range(count):
                atlas_index, uv_offset, uv_scale, color_variance, variance_amount = (
                    self._make_leaf_style(branch.variation_seed + i * 0.31)
                )
                theta = random.uniform(0.0, math.tau)
                phi = math.acos(random.uniform(-1.0, 1.0))
                radius = random.random() ** (1.0 / 3.0)
                dir_x = math.sin(phi) * math.cos(theta)
                dir_y = math.sin(phi) * math.sin(theta)
                dir_z = math.cos(phi)
                offset = (
                    dir_x * cluster_radius[0] * radius,
                    dir_y * cluster_radius[1] * radius,
                    dir_z * cluster_radius[2] * radius,
                )
                jitter = (
                    random.uniform(-2.0, 2.0),
                    random.uniform(-2.0, 2.0),
                    random.uniform(-1.5, 1.5),
                )
                offset = (
                    offset[0] + jitter[0],
                    offset[1] + jitter[1],
                    offset[2] + jitter[2],
                )
                yaw = random.uniform(0.0, math.tau)
                right = (math.cos(yaw), math.sin(yaw), 0.0)
                up = (-math.sin(yaw), math.cos(yaw), 0.0)
                normal = (0.0, 0.0, 1.0)
                droop = math.radians(random.uniform(8.0, 22.0))
                up = self._rotate_around_axis(up, right, droop)
                normal = self._rotate_around_axis(normal, right, droop)
                size = random.uniform(8, 16) * (0.85 + zf * 0.25)
                axis_x = (right[0] * size, right[1] * size, right[2] * size)
                axis_y = (up[0] * size, up[1] * size, up[2] * size)
                axis_z = self._normalize(normal)
                self.canopy_leaves.append(CanopyLeaf(
                    branch_index=idx,
                    offset=offset,
                    axis_x=axis_x,
                    axis_y=axis_y,
                    axis_z=axis_z,
                    base_color=base_color,
                    atlas_index=atlas_index,
                    uv_offset=uv_offset,
                    uv_scale=uv_scale,
                    color_variance=color_variance,
                    variance_amount=variance_amount,
                ))

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
            r = (60 + (1 - zf) * 50) * pulse / 255.0
            g = (200 + zf * 55) * pulse / 255.0
            blue = 255 * pulse / 255.0
            color = (r, g, blue, 0.7)
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
    def _load_texture(self, path: str, fallback_color: tuple[int, int, int, int]) -> moderngl.Texture:
        image = None
        file_path = Path(path)
        if file_path.exists():
            try:
                image = pyglet.image.load(str(file_path))
            except pyglet.image.codecs.ImageDecodeException:
                image = None
        if image:
            image_data = image.get_image_data()
            data = image_data.get_data("RGBA", image_data.width * 4)
            texture = self.ctx.texture((image_data.width, image_data.height), 4, data)
            texture.repeat_x = True
            texture.repeat_y = True
            if image_data.width > 1 or image_data.height > 1:
                texture.build_mipmaps()
                texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            else:
                texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            return texture
        texture = self.ctx.texture((1, 1), 4, bytes(fallback_color))
        texture.repeat_x = True
        texture.repeat_y = True
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
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.use_alpha_to_coverage = self.ctx.fbo.samples > 1

        self.camera_position = (tree.w * 0.5, tree.h * 0.6, 520.0)
        self.camera_target = (tree.w * 0.5, tree.h * 0.45, 0.0)
        self.camera_up = (0.0, 1.0, 0.0)

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

                out vec2 v_local;
                out vec4 v_color;
                out vec2 v_uv;
                out vec3 v_bark_tint;
                out float v_roughness;
                out vec3 v_world_pos;
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

                void main() {
                    float edge = abs(v_local.y);
                    float core = smoothstep(1.0, 0.0, edge);
                    float glow = exp(-edge * 4.0) * u_glow;
                    vec2 bark_uv = vec2(v_uv.x * u_bark_uv_scale.x, v_uv.y * u_bark_uv_scale.y);
                    vec3 albedo = texture(u_bark_albedo, bark_uv).rgb * v_bark_tint;
                    vec3 normal = normalize(v_normal);
                    if (u_has_normal == 1) {
                        vec3 tangent_normal = texture(u_bark_normal, bark_uv).xyz * 2.0 - 1.0;
                        mat3 tbn = mat3(normalize(v_tangent), normalize(v_bitangent), normalize(v_normal));
                        normal = normalize(tbn * tangent_normal);
                    }
                    float roughness = v_roughness;
                    if (u_has_roughness == 1) {
                        roughness = clamp(v_roughness * (0.6 + 0.8 * texture(u_bark_roughness, bark_uv).r), 0.05, 1.0);
                    }
                    float occlusion = mix(0.65, 1.0, core);
                    vec3 view_dir = normalize(u_camera_pos - v_world_pos);
                    vec3 light_dir = normalize(vec3(0.35, 0.55, 0.78));
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
                    vec3 radiance = vec3(1.0);
                    vec3 lighting = (diffuse + specular) * radiance * n_dot_l;
                    vec3 ambient = albedo * 0.18;
                    vec3 lit = (ambient + lighting) * occlusion;
                    lit *= v_color.rgb;
                    lit = tone_map(lit);
                    lit = pow(lit, vec3(1.0 / 2.2));
                    float alpha = (v_color.a * core) + glow;
                    f_color = vec4(lit, alpha);
                }
            """,
        )

        self.bark_albedo = self._load_texture("assets/bark_albedo.png", fallback_color=(94, 63, 45, 255))
        self.bark_normal = self._load_texture("assets/bark_normal.png", fallback_color=(128, 128, 255, 255))
        self.bark_roughness = self._load_texture("assets/bark_roughness.png", fallback_color=(180, 180, 180, 255))
        self.has_bark_normal = Path("assets/bark_normal.png").exists()
        self.has_bark_roughness = Path("assets/bark_roughness.png").exists()
        self.leaf_atlas = self._load_texture("assets/leaf_atlas.png", fallback_color=(255, 255, 255, 255))
        self.leaf_atlas.repeat_x = False
        self.leaf_atlas.repeat_y = False
        self.bird_sprite = self._load_texture("assets/bird_sprite.png", fallback_color=(255, 255, 255, 0))
        self.bird_sprite.repeat_x = False
        self.bird_sprite.repeat_y = False

        self.leaf_program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                in vec3 in_center;
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

                out vec2 v_uv;
                out float v_atlas_index;
                out vec4 v_color;
                out vec4 v_variance;
                out vec3 v_normal;

                void main() {
                    vec3 offset = in_axis_x * in_pos.x + in_axis_y * in_pos.y;
                    vec4 world = vec4(in_center + offset, 1.0);
                    gl_Position = u_proj * u_view * world;
                    vec2 base_uv = in_pos * 0.5 + 0.5;
                    v_uv = in_uv_offset + base_uv * in_uv_scale;
                    v_atlas_index = in_atlas_index;
                    v_color = in_color;
                    v_variance = in_variance;
                    v_normal = normalize(in_axis_z);
                }
            """,
            fragment_shader="""
                #version 330
                in vec2 v_uv;
                in float v_atlas_index;
                in vec4 v_color;
                in vec4 v_variance;
                in vec3 v_normal;

                uniform sampler2D u_leaf_atlas;
                uniform vec2 u_atlas_grid;
                uniform float u_alpha_cutoff;

                out vec4 f_color;

                vec3 tone_map(vec3 color) {
                    return color / (color + vec3(1.0));
                }

                void main() {
                    vec2 grid = max(u_atlas_grid, vec2(1.0));
                    float index = max(v_atlas_index, 0.0);
                    vec2 cell = vec2(mod(index, grid.x), floor(index / grid.x));
                    vec2 atlas_uv = (cell + clamp(v_uv, 0.0, 1.0)) / grid;
                    vec4 tex = texture(u_leaf_atlas, atlas_uv);
                    float alpha = v_color.a * tex.a;
                    alpha *= clamp(abs(v_normal.z), 0.0, 1.0);
                    if (alpha < u_alpha_cutoff) {
                        discard;
                    }
                    vec3 variance = v_variance.xyz * v_variance.w;
                    vec3 tinted = clamp(v_color.rgb * (1.0 + variance), 0.0, 1.0);
                    vec3 rgb = tinted * tex.rgb;
                    rgb *= alpha;
                    rgb = tone_map(rgb);
                    rgb = pow(rgb, vec3(1.0 / 2.2));
                    f_color = vec4(rgb, alpha);
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

        self.branch_instance_vbo = self.ctx.buffer(reserve=4 * 1024 * 96)
        self.leaf_instance_vbo = self.ctx.buffer(reserve=4 * 1024 * 128)
        self.bird_instance_vbo = self.ctx.buffer(reserve=4 * 64)
        self.ui_line_instance_vbo = self.ctx.buffer(reserve=4 * 64)

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
                (self.leaf_instance_vbo, "3f 3f 3f 3f 2f 2f f 4f 4f /i",
                 "in_center", "in_axis_x", "in_axis_y", "in_axis_z",
                 "in_uv_offset", "in_uv_scale",
                 "in_atlas_index", "in_color", "in_variance"),
            ],
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

    def render(self):
        width, height = self.window.get_framebuffer_size()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.clear(0.015, 0.05, 0.11)
        aspect = width / max(height, 1)
        view = array('f', self._look_at(self.camera_position, self.camera_target, self.camera_up))
        proj = array('f', self._perspective(math.radians(45.0), aspect, 10.0, 2000.0))

        branch_instances = self.tree.build_branch_instances()
        if branch_instances:
            data = array('f', branch_instances)
            self.branch_instance_vbo.orphan(len(data) * 4)
            self.branch_instance_vbo.write(data)

            self.branch_program["u_view"].write(view)
            self.branch_program["u_proj"].write(proj)
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

            self.branch_program["u_glow"].value = 0.0
            self.branch_vao.render(instances=len(branch_instances) // 24)

        leaf_instances = self.tree.build_leaf_instances()
        if leaf_instances:
            if self.use_alpha_to_coverage:
                self.ctx.enable(moderngl.SAMPLE_ALPHA_TO_COVERAGE)
            else:
                self.ctx.disable(moderngl.SAMPLE_ALPHA_TO_COVERAGE)
                cx, cy, cz = self.camera_position
                leaf_instances.sort(
                    key=lambda inst: (inst.position[0] - cx) ** 2
                    + (inst.position[1] - cy) ** 2
                    + (inst.position[2] - cz) ** 2,
                    reverse=True,
                )
            leaf_data: List[float] = []
            for leaf in leaf_instances:
                leaf_data.extend([
                    *leaf.position,
                    *leaf.axis_x,
                    *leaf.axis_y,
                    *leaf.axis_z,
                    *leaf.uv_offset,
                    *leaf.uv_scale,
                    float(leaf.atlas_index),
                    *leaf.color,
                    *leaf.color_variance,
                    leaf.variance_amount,
                ])
            data = array('f', leaf_data)
            self.leaf_instance_vbo.orphan(len(data) * 4)
            self.leaf_instance_vbo.write(data)
            self.leaf_program["u_view"].write(view)
            self.leaf_program["u_proj"].write(proj)
            self.leaf_program["u_atlas_grid"].value = (self.tree.leaf_atlas_cols, self.tree.leaf_atlas_rows)
            self.leaf_program["u_alpha_cutoff"].value = 0.2
            self.leaf_atlas.use(location=0)
            self.leaf_program["u_leaf_atlas"].value = 0
            self.leaf_vao.render(instances=len(leaf_data) // 25)
            if self.use_alpha_to_coverage:
                self.ctx.disable(moderngl.SAMPLE_ALPHA_TO_COVERAGE)

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
        self.info_batch.draw()


def main():
    window = HolographicWindow()
    pyglet.app.run()


if __name__ == "__main__":
    main()
