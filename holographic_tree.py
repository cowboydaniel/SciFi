#!/usr/bin/env python3
"""
Holographic Tree - GPU Accelerated with Pygame
Uses hardware-accelerated blitting for 60+ FPS
"""

import pygame
import pygame.gfxdraw
import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


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
    end_x: float = 0.0
    end_y: float = 0.0
    segment_fracs: List[float] = field(default_factory=list)
    segment_jitter: List[float] = field(default_factory=list)
    segment_sway: List[float] = field(default_factory=list)
    segment_points: List[tuple] = field(default_factory=list)


@dataclass
class AttachedLeaf:
    branch_index: int
    t: float
    side: int
    size: float
    angle_offset: float


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
    hop_offset: Tuple[float, float] = (0.0, 0.0)
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

        self.wind = WindSystem()

        self.branches: List[Branch] = []
        self.falling_leaves: List[FallingLeaf] = []
        self.max_leaves = 15

        self.sorted_branches: List[Branch] = []
        self.tip_branches: List[Branch] = []
        self.tip_indices: List[int] = []
        self.attached_leaves: List[AttachedLeaf] = []

        self.root_x = width / 2
        self.root_y = height - 50

        # Leaf surface cache
        self.leaf_surfaces = {}
        self.leaf_rotation_cache = {}
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
            perch_wait=random.uniform(2.5, 4.5)
        )

        self.regenerate_tree()

    @staticmethod
    def draw_thick_aaline(surface, color, x1, y1, x2, y2, thickness):
        """Draw a thick antialiased line using filled polygon with antialiased edges."""
        if thickness <= 1:
            pygame.gfxdraw.line(surface, int(x1), int(y1), int(x2), int(y2), color)
            return

        # Calculate perpendicular offset
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length == 0:
            return

        # Normalize and get perpendicular
        dx /= length
        dy /= length
        px = -dy * thickness / 2
        py = dx * thickness / 2

        # Four corners of the thick line
        points = [
            (int(x1 + px), int(y1 + py)),
            (int(x1 - px), int(y1 - py)),
            (int(x2 - px), int(y2 - py)),
            (int(x2 + px), int(y2 + py))
        ]

        # Draw filled polygon
        pygame.gfxdraw.filled_polygon(surface, points, color)
        # Draw antialiased outline
        pygame.gfxdraw.aapolygon(surface, points, color)

    def regenerate_tree(self):
        self.branches.clear()
        self.leaf_rotation_cache.clear()
        self._gen_branch(-1, -90, 180, 0, 9, 0.0)
        self.sorted_branches = sorted(self.branches, key=lambda b: b.z_depth)
        self.tip_indices = [i for i, branch in enumerate(self.branches) if branch.depth >= 6]
        self.tip_branches = [self.branches[i] for i in self.tip_indices]
        self.attached_leaves = []
        for idx in self.tip_indices:
            for _ in range(random.randint(1, 3)):
                self.attached_leaves.append(AttachedLeaf(
                    branch_index=idx,
                    t=random.uniform(0.82, 1.0),
                    side=random.choice([-1, 1]),
                    size=random.uniform(6, 12),
                    angle_offset=random.uniform(-12, 12)
                ))
        self._reset_bird()

    def _get_leaf_surface(self, size: int) -> tuple[pygame.Surface, tuple[float, float]]:
        size = max(4, int(size))
        if size in self.leaf_surfaces:
            return self.leaf_surfaces[size]

        surf = pygame.Surface((size * 4, size * 4), pygame.SRCALPHA)
        cx_l, cy_l = size * 2, size * 2
        body_w = int(size * 1.3)
        body_h = int(size * 1.9)
        body_rect = pygame.Rect(0, 0, body_w, body_h)
        body_rect.center = (cx_l, cy_l + int(size * 0.2))
        tip_height = int(size * 0.7)
        tip_top = body_rect.top - tip_height
        tip_left = body_rect.left + int(body_w * 0.15)
        tip_right = body_rect.right - int(body_w * 0.15)
        tip_points = [
            (cx_l, tip_top),
            (tip_right, body_rect.top + int(body_h * 0.2)),
            (tip_left, body_rect.top + int(body_h * 0.2))
        ]

        # Base volume
        base_color = (70, 200, 255, 160)
        mid_color = (110, 235, 255, 190)
        highlight_color = (190, 255, 255, 210)
        outline_color = (200, 255, 255, 230)
        shadow_color = (20, 90, 120, 140)
        rim_color = (220, 255, 255, 240)

        shadow_rect = body_rect.copy()
        shadow_rect.center = (shadow_rect.centerx + int(size * 0.2),
                              shadow_rect.centery + int(size * 0.25))
        pygame.draw.ellipse(surf, shadow_color, shadow_rect)
        pygame.draw.ellipse(surf, base_color, body_rect)
        pygame.draw.polygon(surf, base_color, tip_points)

        inner_rect = body_rect.inflate(-int(size * 0.3), -int(size * 0.3))
        pygame.draw.ellipse(surf, mid_color, inner_rect)

        highlight_rect = inner_rect.inflate(-int(size * 0.4), -int(size * 0.4))
        highlight_rect.center = (highlight_rect.centerx - int(size * 0.25),
                                 highlight_rect.centery - int(size * 0.25))
        pygame.draw.ellipse(surf, highlight_color, highlight_rect)
        rim_rect = body_rect.inflate(-int(size * 0.1), -int(size * 0.1))
        pygame.draw.ellipse(surf, rim_color, rim_rect, 1)
        vein_color = (180, 255, 255, 160)
        pygame.draw.line(surf, vein_color, (cx_l, body_rect.top + int(size * 0.2)),
                         (cx_l, body_rect.bottom - int(size * 0.2)), 1)

        pygame.draw.ellipse(surf, outline_color, body_rect, 1)
        pygame.draw.polygon(surf, outline_color, tip_points, 1)

        # Stem
        stem_length = max(4, int(size * 0.5))
        stem_start = (cx_l, body_rect.bottom - 1)
        stem_end = (cx_l, body_rect.bottom + stem_length)
        pygame.draw.line(surf, outline_color, stem_start, stem_end, max(1, size // 6))

        base_offset = (stem_end[0] - cx_l, stem_end[1] - cy_l)
        self.leaf_surfaces[size] = (surf, base_offset)
        return surf, base_offset

    @staticmethod
    def _rotate_point(x: float, y: float, angle_deg: float) -> tuple[float, float]:
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        return (x * cos_a - y * sin_a, x * sin_a + y * cos_a)

    def _get_rotated_leaf(self, size: int, angle: float) -> tuple[pygame.Surface, tuple[float, float]]:
        leaf_surf, base_offset = self._get_leaf_surface(size)
        angle = angle % 360
        bucket = int(round(angle / 6)) * 6
        key = (size, bucket)
        cached = self.leaf_rotation_cache.get(key)
        if cached:
            return cached
        rotated = pygame.transform.rotate(leaf_surf, -bucket)
        base_rot = self._rotate_point(base_offset[0], base_offset[1], -bucket)
        self.leaf_rotation_cache[key] = (rotated, base_rot)
        return rotated, base_rot

    @staticmethod
    def _get_branch_point_and_angle(points: List[tuple], t: float) -> tuple[tuple[float, float], float]:
        if len(points) < 2:
            return (points[0] if points else (0.0, 0.0)), -90.0
        t = max(0.0, min(1.0, t))
        lengths = []
        total = 0.0
        for i in range(len(points) - 1):
            seg_len = math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
            lengths.append(seg_len)
            total += seg_len
        if total <= 0:
            p0, p1 = points[-2], points[-1]
            angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
            return p1, angle
        target = total * t
        acc = 0.0
        for i, seg_len in enumerate(lengths):
            if acc + seg_len >= target:
                local_t = (target - acc) / seg_len if seg_len else 0.0
                x0, y0 = points[i]
                x1, y1 = points[i + 1]
                x = x0 + (x1 - x0) * local_t
                y = y0 + (y1 - y0) * local_t
                angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
                return (x, y), angle
            acc += seg_len
        p0, p1 = points[-2], points[-1]
        angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
        return p1, angle

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

    def _get_perch_point(self) -> Optional[Tuple[float, float, float]]:
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

    def _draw_bird(self, surface: pygame.Surface):
        target = self._get_perch_point()
        if not target:
            return
        bird = self.bird
        body_color = (110, 235, 255)
        belly_color = (160, 255, 255)
        wing_color = (70, 200, 255)
        shadow_color = (30, 120, 160)
        beak_color = (255, 210, 80)

        flap = math.sin(bird.wing_phase) if bird.wing_speed > 0 else 0.2
        wing_span = 18 + 8 * abs(flap)
        wing_lift = -6 - 5 * flap
        facing = bird.facing

        body_rect = pygame.Rect(0, 0, 28, 14)
        body_rect.center = (int(bird.x), int(bird.y))
        pygame.gfxdraw.filled_ellipse(surface, body_rect.centerx, body_rect.centery,
                                      body_rect.width // 2, body_rect.height // 2, body_color)
        pygame.gfxdraw.aaellipse(surface, body_rect.centerx, body_rect.centery,
                                 body_rect.width // 2, body_rect.height // 2, belly_color)

        belly_rect = pygame.Rect(0, 0, 20, 10)
        belly_rect.center = (int(bird.x - 2 * facing), int(bird.y + 2))
        pygame.gfxdraw.filled_ellipse(surface, belly_rect.centerx, belly_rect.centery,
                                      belly_rect.width // 2, belly_rect.height // 2, belly_color)

        head_center = (int(bird.x + 12 * facing), int(bird.y - 4))
        pygame.gfxdraw.filled_circle(surface, head_center[0], head_center[1], 6, body_color)
        pygame.gfxdraw.aacircle(surface, head_center[0], head_center[1], 6, belly_color)

        beak = [
            (head_center[0] + 6 * facing, head_center[1]),
            (head_center[0] + 12 * facing, head_center[1] - 2),
            (head_center[0] + 12 * facing, head_center[1] + 2)
        ]
        pygame.gfxdraw.filled_polygon(surface, beak, beak_color)
        pygame.gfxdraw.aapolygon(surface, beak, beak_color)

        tail = [
            (int(bird.x - 14 * facing), int(bird.y)),
            (int(bird.x - 24 * facing), int(bird.y - 4)),
            (int(bird.x - 24 * facing), int(bird.y + 4))
        ]
        pygame.gfxdraw.filled_polygon(surface, tail, shadow_color)
        pygame.gfxdraw.aapolygon(surface, tail, shadow_color)

        wing_base = (int(bird.x - 2 * facing), int(bird.y - 2))
        wing = [
            wing_base,
            (int(bird.x - wing_span * facing), int(bird.y + wing_lift)),
            (int(bird.x - 6 * facing), int(bird.y + 8))
        ]
        pygame.gfxdraw.filled_polygon(surface, wing, wing_color)
        pygame.gfxdraw.aapolygon(surface, wing, wing_color)

    def _gen_branch(self, parent: int, angle: float, length: float, depth: int, max_d: int, z: float):
        if depth >= max_d or length < 10:
            return
        stiff = 1.0 - (depth / max_d) * 0.85
        nz = max(-1, min(1, z + random.uniform(-0.1, 0.1)))
        b = Branch(parent, angle, length, max(2, (max_d - depth) * 2.5),
                   depth, nz, stiff, depth * 0.3 + random.random() * 0.5)
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

    def update(self, dt: float):
        self.time += dt
        self.wind.update(dt)

        # Update branches
        for b in self.branches:
            if b.parent_index < 0:
                b.start_x, b.start_y = self.root_x, self.root_y
            else:
                p = self.branches[b.parent_index]
                b.start_x, b.start_y = p.end_x, p.end_y

            flex = 1.0 - b.stiffness
            wave = math.sin(self.time * 2.5 - b.phase_offset) * 0.3
            scale = 1.0 + b.z_depth * 0.15
            length = b.length * scale
            base_angle = b.base_angle + math.sin(self.time * 1.8 + b.phase_offset * 2) * 2 * flex

            points = [(b.start_x, b.start_y)]
            x, y = b.start_x, b.start_y
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
                x += seg_len * math.cos(rad)
                y += seg_len * math.sin(rad)
                points.append((x, y))

            b.current_angle = current_angle
            b.segment_points = points
            b.end_x, b.end_y = points[-1]

        # Flicker
        self.flicker = 0.88 + 0.12 * math.sin(self.time * 10)
        if random.random() > 0.97:
            self.flicker *= 0.75

        # Spawn leaves
        if len(self.falling_leaves) < self.max_leaves and self.tip_branches and random.random() > 0.988:
            tb = random.choice(self.tip_branches)
            self.falling_leaves.append(FallingLeaf(
                tb.end_x, tb.end_y, tb.z_depth,
                random.uniform(-0.4, 0.4), random.uniform(0.5, 1.5),
                random.uniform(0, 360), random.uniform(-4, 4),
                random.uniform(12, 20), random.uniform(0, 6.28),
                random.uniform(2, 4), 1.0, 0
            ))

        # Update leaves
        ground = self.h - 60
        new_l = []
        for lf in self.falling_leaves:
            lf.lifetime += 1
            lf.vx += self.wind.get_force(lf.y, self.h) * 0.003
            lf.vx *= 0.98
            lf.vy = min(lf.vy + 0.03, 2.5)
            lf.x += lf.vx + math.sin(lf.wobble_phase) * 1.5
            lf.wobble_phase += lf.wobble_speed * 0.016
            lf.y += lf.vy
            lf.rotation += lf.rotation_speed

            if lf.y >= ground:
                lf.y = ground
                lf.vy = 0
                lf.rotation_speed *= 0.92
                lf.alpha -= 0.01

            if lf.alpha > 0 and 0 < lf.x < self.w:
                new_l.append(lf)
        self.falling_leaves = new_l

        self._update_bird(dt)

    def draw(self, screen: pygame.Surface):
        f = self.flicker

        # Background
        screen.fill((4, 12, 28))

        # Grid
        cx, hy = self.w // 2, int(self.h * 0.35)
        grid_color = (0, 50, 70)
        for i in range(-8, 9, 2):
            pygame.gfxdraw.line(screen, cx + i * 100, self.h, cx, hy, grid_color)
        for i in range(8):
            ratio = i / 8
            y = self.h - int((self.h - hy) * (ratio ** 1.3))
            sp = int((1 - ratio ** 1.3) * self.w * 0.6)
            pygame.gfxdraw.line(screen, cx - sp, y, cx + sp, y, grid_color)

        # Platform
        py = self.h - 40
        for i in range(4):
            a = int((180 - i * 40) * f)
            color = (0, a, a)
            ew, eh = 420 + i * 60, 40 + i * 10
            rect = pygame.Rect(cx - ew // 2, py - eh // 2, ew, eh)
            pygame.gfxdraw.filled_ellipse(screen, rect.centerx, rect.centery, rect.width // 2, rect.height // 2,
                                          (0, a // 3, a // 2, 40))
            pygame.gfxdraw.aaellipse(screen, rect.centerx, rect.centery, rect.width // 2, rect.height // 2, color)

        # Tree glow (using thick antialiased lines)
        for b in self.sorted_branches:
            zf = (b.z_depth + 1) / 2
            depth_offset = int((1 - zf) * 6)
            points = b.segment_points or [(b.start_x, b.start_y), (b.end_x, b.end_y)]
            if depth_offset:
                for i in range(len(points) - 1):
                    self.draw_thick_aaline(screen, (0, int(30 * f), int(50 * f)),
                                          points[i][0] + depth_offset, points[i][1] + depth_offset,
                                          points[i + 1][0] + depth_offset, points[i + 1][1] + depth_offset,
                                          int(b.thickness + 12))
            glow_thick = int(b.thickness + 10)
            for i in range(len(points) - 1):
                self.draw_thick_aaline(screen, (0, int(80 * f), int(120 * f)),
                                      points[i][0], points[i][1], points[i + 1][0], points[i + 1][1],
                                      glow_thick)
            for px, py in points:
                pygame.gfxdraw.filled_circle(screen, int(px), int(py), glow_thick // 2,
                                             (0, int(80 * f), int(120 * f)))
                pygame.gfxdraw.aacircle(screen, int(px), int(py), glow_thick // 2,
                                        (0, int(90 * f), int(140 * f)))

        # Tree branches
        for b in self.sorted_branches:
            zf = (b.z_depth + 1) / 2
            pulse = (0.88 + 0.12 * math.sin(self.time * 3 + b.phase_offset)) * f
            r = int((60 + (1 - zf) * 50) * pulse)
            g = int((200 + zf * 55) * pulse)
            blue = int(255 * pulse)
            th = max(1, int(b.thickness * (0.85 + zf * 0.3)))

            # Draw thick antialiased branch
            points = b.segment_points or [(b.start_x, b.start_y), (b.end_x, b.end_y)]
            for i in range(len(points) - 1):
                shadow_offset = 2 + int((1 - zf) * 3)
                back_offset = 3 + int((1 - zf) * 5)
                self.draw_thick_aaline(screen, (0, int(25 * f), int(35 * f)),
                                      points[i][0] + back_offset, points[i][1] + back_offset,
                                      points[i + 1][0] + back_offset, points[i + 1][1] + back_offset, th + 6)
                self.draw_thick_aaline(screen, (0, int(40 * f), int(60 * f)),
                                      points[i][0] + shadow_offset, points[i][1] + shadow_offset,
                                      points[i + 1][0] + shadow_offset, points[i + 1][1] + shadow_offset, th + 2)
                self.draw_thick_aaline(screen, (r, g, blue),
                                      points[i][0], points[i][1], points[i + 1][0], points[i + 1][1], th)
            for px, py in points:
                pygame.gfxdraw.filled_circle(screen, int(px), int(py), max(1, th // 2), (r, g, blue))
                pygame.gfxdraw.aacircle(screen, int(px), int(py), max(1, th // 2), (r, g, blue))

            # Core highlight
            if b.thickness > 4:
                core_th = max(1, th // 3)
                ca = int(220 * pulse)
                for i in range(len(points) - 1):
                    self.draw_thick_aaline(screen, (ca, 255, 255),
                                          points[i][0], points[i][1], points[i + 1][0], points[i + 1][1],
                                          core_th)

        # Attached leaves
        for leaf in self.attached_leaves:
            if leaf.branch_index >= len(self.branches):
                continue
            branch = self.branches[leaf.branch_index]
            points = branch.segment_points or [(branch.start_x, branch.start_y), (branch.end_x, branch.end_y)]
            if not points:
                continue
            (attach_x, attach_y), tangent_angle = self._get_branch_point_and_angle(points, leaf.t)
            angle = tangent_angle + 180 + leaf.angle_offset + leaf.side * 6
            a = 0.7 * f
            size = int(leaf.size * (0.7 + (branch.z_depth + 1) * 0.15))
            size = max(4, size)
            zf = (branch.z_depth + 1) / 2
            shadow_offset = 2 + int((1 - zf) * 3)

            rotated_base, base_rot = self._get_rotated_leaf(size, angle)
            rotated = rotated_base.copy()
            x = attach_x - base_rot[0]
            y = attach_y - base_rot[1]
            rect = rotated.get_rect(center=(int(x), int(y)))
            shadow = rotated.copy()
            shadow.fill((30, 90, 120, 140), special_flags=pygame.BLEND_RGBA_MULT)
            shadow_rect = shadow.get_rect(center=(int(x + shadow_offset), int(y + shadow_offset)))
            shadow.set_alpha(int(200 * a))
            screen.blit(shadow, shadow_rect, special_flags=pygame.BLEND_ALPHA_SDL2)

            rotated.set_alpha(int(255 * a))
            screen.blit(rotated, rect, special_flags=pygame.BLEND_ADD)

            highlight = rotated.copy()
            highlight.fill((220, 255, 255, 200), special_flags=pygame.BLEND_RGBA_MULT)
            highlight_rect = highlight.get_rect(center=(int(x - 1), int(y - 1)))
            highlight.set_alpha(int(120 * a))
            screen.blit(highlight, highlight_rect, special_flags=pygame.BLEND_ADD)

        # Leaves
        for lf in self.falling_leaves:
            sz = int(lf.size * (0.8 + (lf.z + 1) * 0.2))
            a = lf.alpha * f
            zf = (lf.z + 1) / 2
            shadow_offset = 2 + int((1 - zf) * 4)

            rotated_base, _ = self._get_rotated_leaf(sz, lf.rotation)
            rotated = rotated_base.copy()
            rect = rotated.get_rect(center=(int(lf.x), int(lf.y)))

            shadow = rotated.copy()
            shadow.fill((20, 80, 120, 140), special_flags=pygame.BLEND_RGBA_MULT)
            shadow_rect = shadow.get_rect(center=(int(lf.x + shadow_offset), int(lf.y + shadow_offset)))
            shadow.set_alpha(int(200 * a))
            screen.blit(shadow, shadow_rect, special_flags=pygame.BLEND_ALPHA_SDL2)

            rotated.set_alpha(int(255 * a))
            screen.blit(rotated, rect, special_flags=pygame.BLEND_ADD)

            highlight = rotated.copy()
            highlight.fill((220, 255, 255, 200), special_flags=pygame.BLEND_RGBA_MULT)
            highlight_rect = highlight.get_rect(center=(int(lf.x - 1), int(lf.y - 1)))
            highlight.set_alpha(int(110 * a))
            screen.blit(highlight, highlight_rect, special_flags=pygame.BLEND_ADD)

        self._draw_bird(screen)


def main():
    pygame.init()

    # Try to load font, but continue without it if there's a circular import issue (Python 3.14)
    font = None
    use_freetype = False
    try:
        from pygame import freetype
        font = freetype.SysFont("Courier", 16)
        use_freetype = True
    except ImportError:
        print("Warning: Could not load pygame.freetype due to circular import (Python 3.14 issue)")
        pygame.font.init()
        try:
            font = pygame.font.SysFont("Courier", 16)
        except Exception:
            font = pygame.font.Font(None, 16)

    # Get primary display
    info = pygame.display.Info()
    width, height = info.current_w, info.current_h

    # Fullscreen with hardware acceleration
    screen = pygame.display.set_mode((width, height),
                                      pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Holographic Tree")
    pygame.mouse.set_visible(False)

    clock = pygame.time.Clock()
    tree = HolographicTree(width, height)

    running = True
    frame_times = []
    last_time = time.perf_counter()

    while running:
        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_r, pygame.K_SPACE):
                    tree.regenerate_tree()
                elif event.key == pygame.K_f:
                    pygame.display.toggle_fullscreen()

        # FPS calculation
        now = time.perf_counter()
        dt = now - last_time
        last_time = now
        frame_times.append(dt)
        if len(frame_times) > 60:
            frame_times.pop(0)
        fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 60

        # Update
        tree.update(dt)

        # Draw
        tree.draw(screen)

        # UI overlay
        f = tree.flicker
        ui_color = (0, int(255 * f), int(255 * f))

        # Corner brackets (using thick antialiased lines)
        HolographicTree.draw_thick_aaline(screen, ui_color, 20, 20, 100, 20, 2)
        HolographicTree.draw_thick_aaline(screen, ui_color, 20, 20, 20, 100, 2)
        HolographicTree.draw_thick_aaline(screen, ui_color, width - 100, 20, width - 20, 20, 2)
        HolographicTree.draw_thick_aaline(screen, ui_color, width - 20, 20, width - 20, 100, 2)

        # Text
        if font:
            texts = [
                "HOLOGRAPHIC PROJECTION ACTIVE",
                f"BRANCHES: {len(tree.branches)}",
                f"WIND: {tree.wind.gust_strength:.1f}",
                f"FPS: {fps:.1f}"
            ]
            for i, text in enumerate(texts):
                pos = (30, 40 + i * 22)
                if use_freetype:
                    font.render_to(screen, pos, text, ui_color)
                else:
                    text_surface = font.render(text, True, ui_color)
                    screen.blit(text_surface, pos)

            # Bottom status
            bottom_texts = [
                "QUANTUM COHERENCE: STABLE",
                "DIMENSIONAL MATRIX: SYNCHRONIZED",
                "PHOTON FIELD: ACTIVE"
            ]
            for i, text in enumerate(bottom_texts):
                pos = (30, height - 90 + i * 22)
                if use_freetype:
                    font.render_to(screen, pos, text, ui_color)
                else:
                    text_surface = font.render(text, True, ui_color)
                    screen.blit(text_surface, pos)

        pygame.display.flip()
        clock.tick(165)  # Match your 165Hz monitor

    pygame.quit()


if __name__ == "__main__":
    main()
