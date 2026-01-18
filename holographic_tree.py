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
from typing import List


@dataclass
class Particle:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    size: float
    alpha: float
    lifetime: float
    max_lifetime: float


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
        self.particles: List[Particle] = []
        self.falling_leaves: List[FallingLeaf] = []
        self.max_particles = 150
        self.max_leaves = 15

        self.sorted_branches: List[Branch] = []
        self.tip_branches: List[Branch] = []
        self.attached_leaves: List[AttachedLeaf] = []

        self.root_x = width / 2
        self.root_y = height - 50

        # Pre-create glow surfaces for particles (GPU cached)
        self.glow_surfaces = {}
        for size in range(1, 8):
            surf = pygame.Surface((size * 6, size * 6), pygame.SRCALPHA)
            for r in range(size * 3, 0, -1):
                alpha = int(60 * (1 - r / (size * 3)))
                pygame.draw.circle(surf, (80, 220, 255, alpha), (size * 3, size * 3), r)
            self.glow_surfaces[size] = surf

        # Leaf surface cache
        self.leaf_surfaces = {}

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
        self._gen_branch(-1, -90, 180, 0, 9, 0.0)
        self.sorted_branches = sorted(self.branches, key=lambda b: b.z_depth)
        tip_indices = [i for i, branch in enumerate(self.branches) if branch.depth >= 6]
        self.tip_branches = [self.branches[i] for i in tip_indices]
        self.attached_leaves = []
        for idx in tip_indices:
            for _ in range(random.randint(1, 3)):
                self.attached_leaves.append(AttachedLeaf(
                    branch_index=idx,
                    t=random.uniform(0.6, 1.0),
                    side=random.choice([-1, 1]),
                    size=random.uniform(6, 12),
                    angle_offset=random.uniform(-25, 25)
                ))

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

        # Spawn particles
        if len(self.particles) < self.max_particles and self.tip_branches and random.random() > 0.65:
            tb = random.choice(self.tip_branches)
            self.particles.append(Particle(
                tb.end_x + random.uniform(-10, 10), tb.end_y + random.uniform(-10, 10),
                tb.z_depth, random.uniform(-0.6, 0.6), random.uniform(-2.5, -0.8),
                random.uniform(1.5, 3.0), random.uniform(0.6, 1.0), 0, random.uniform(80, 150)
            ))

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

        # Update particles
        new_p = []
        for p in self.particles:
            p.x += p.vx
            p.y += p.vy
            p.vy -= 0.008
            p.lifetime += 1
            if p.lifetime < p.max_lifetime and 0 < p.x < self.w and p.y > -30:
                new_p.append(p)
        self.particles = new_p

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
            pygame.draw.ellipse(screen, color, rect, 2)

        # Tree glow (using thick antialiased lines)
        for b in self.sorted_branches:
            glow_thick = int(b.thickness + 6)
            points = b.segment_points or [(b.start_x, b.start_y), (b.end_x, b.end_y)]
            for i in range(len(points) - 1):
                self.draw_thick_aaline(screen, (0, int(80 * f), int(120 * f)),
                                      points[i][0], points[i][1], points[i + 1][0], points[i + 1][1],
                                      glow_thick)

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
                self.draw_thick_aaline(screen, (r, g, blue),
                                      points[i][0], points[i][1], points[i + 1][0], points[i + 1][1], th)

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
            if len(points) < 2:
                continue
            start_x, start_y = points[-2]
            end_x, end_y = points[-1]
            dx = end_x - start_x
            dy = end_y - start_y
            length = math.hypot(dx, dy)
            if length == 0:
                continue
            nx, ny = -dy / length, dx / length
            offset = leaf.side * leaf.size * 0.4
            x = start_x + dx * leaf.t + nx * offset
            y = start_y + dy * leaf.t + ny * offset
            angle = math.degrees(math.atan2(dy, dx)) + leaf.angle_offset
            a = 0.7 * f
            size = int(leaf.size * (0.7 + (branch.z_depth + 1) * 0.15))
            size = max(4, size)

            leaf_surf = pygame.Surface((size * 3, size * 3), pygame.SRCALPHA)
            cx_l, cy_l = size * 3 // 2, size * 3 // 2
            body_w = int(size * 1.2)
            body_h = int(size * 1.8)
            body_rect = pygame.Rect(0, 0, body_w, body_h)
            body_rect.center = (cx_l, cy_l + int(size * 0.2))
            tip_height = int(size * 0.6)
            tip_top = body_rect.top - tip_height
            tip_left = body_rect.left + int(body_w * 0.15)
            tip_right = body_rect.right - int(body_w * 0.15)
            tip_points = [(cx_l, tip_top), (tip_right, body_rect.top + int(body_h * 0.2)),
                          (tip_left, body_rect.top + int(body_h * 0.2))]
            fill_color = (90, 220, 255, int(170 * a))
            outline_color = (180, 255, 255, int(210 * a))
            pygame.draw.ellipse(leaf_surf, fill_color, body_rect)
            pygame.draw.polygon(leaf_surf, fill_color, tip_points)
            pygame.draw.ellipse(leaf_surf, outline_color, body_rect, 1)
            pygame.draw.polygon(leaf_surf, outline_color, tip_points, 1)
            stem_length = max(3, int(size * 0.4))
            stem_start = (cx_l, body_rect.bottom - 1)
            stem_end = (cx_l, body_rect.bottom + stem_length)
            pygame.draw.line(leaf_surf, outline_color, stem_start, stem_end, 1)

            rotated = pygame.transform.rotate(leaf_surf, -angle)
            rect = rotated.get_rect(center=(int(x), int(y)))
            screen.blit(rotated, rect, special_flags=pygame.BLEND_ADD)

        # Particles with cached glow surfaces
        for p in self.particles:
            fade = 1.0 if p.lifetime < p.max_lifetime * 0.6 else (
                1 - (p.lifetime - p.max_lifetime * 0.6) / (p.max_lifetime * 0.4))
            a = p.alpha * fade * f
            sz = int(p.size * (0.7 + (p.z + 1) * 0.3))
            sz = max(1, min(sz, 7))

            if sz in self.glow_surfaces:
                glow = self.glow_surfaces[sz].copy()
                glow.set_alpha(int(a * 255))
                screen.blit(glow, (int(p.x - sz * 3), int(p.y - sz * 3)), special_flags=pygame.BLEND_ADD)

            # Core
            core_a = int(255 * a)
            pygame.draw.circle(screen, (200, 255, 255, core_a),
                             (int(p.x), int(p.y)), max(1, sz // 3))

        # Leaves
        for lf in self.falling_leaves:
            sz = int(lf.size * (0.8 + (lf.z + 1) * 0.2))
            a = lf.alpha * f

            # Create rotated leaf
            leaf_surf = pygame.Surface((sz * 3, sz * 3), pygame.SRCALPHA)

            # Leaf shape (teardrop/ellipse)
            cx_l, cy_l = sz * 3 // 2, sz * 3 // 2
            body_w = int(sz * 1.2)
            body_h = int(sz * 1.8)
            body_rect = pygame.Rect(0, 0, body_w, body_h)
            body_rect.center = (cx_l, cy_l + int(sz * 0.2))
            tip_height = int(sz * 0.6)
            tip_top = body_rect.top - tip_height
            tip_left = body_rect.left + int(body_w * 0.15)
            tip_right = body_rect.right - int(body_w * 0.15)
            tip_points = [(cx_l, tip_top), (tip_right, body_rect.top + int(body_h * 0.2)),
                          (tip_left, body_rect.top + int(body_h * 0.2))]
            outline_color = (180, 255, 255, int(200 * a))
            pygame.draw.ellipse(leaf_surf, outline_color, body_rect, 1)
            pygame.draw.polygon(leaf_surf, outline_color, tip_points, 1)

            # Rotate and blit
            rotated = pygame.transform.rotate(leaf_surf, -lf.rotation)
            rect = rotated.get_rect(center=(int(lf.x), int(lf.y)))
            screen.blit(rotated, rect, special_flags=pygame.BLEND_ADD)

        # Scanlines
        scanline_surf = pygame.Surface((self.w, 1), pygame.SRCALPHA)
        scanline_surf.fill((0, 255, 255, 8))
        offset = int(self.time * 60) % 6
        for y in range(offset, self.h, 6):
            screen.blit(scanline_surf, (0, y))

        # Interference glitch
        if random.random() > 0.96:
            y = random.randint(0, self.h)
            h_line = random.randint(2, 10)
            glitch = pygame.Surface((self.w, h_line), pygame.SRCALPHA)
            glitch.fill((0, 255, 255, random.randint(20, 50)))
            screen.blit(glitch, (0, y))


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
                f"PARTICLES: {len(tree.particles)}",
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
