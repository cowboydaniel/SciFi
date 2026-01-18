#!/usr/bin/env python3
"""
Holographic Tree - A Sci-Fi visualization using PySide6
An otherworldly holographic tree that appears to emerge from the screen
GPU ACCELERATED - Full visual effects
"""

import sys
import math
import random
from dataclasses import dataclass
from typing import List

from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QRadialGradient,
    QLinearGradient, QPainterPath, QFont, QSurfaceFormat
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget


@dataclass
class Particle:
    """Floating holographic particle with 3D depth"""
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    size: float
    alpha: float
    pulse_phase: float
    lifetime: float
    max_lifetime: float
    color_hue: float


@dataclass
class EnergyNode:
    """Glowing energy node at branch tips"""
    x: float
    y: float
    z: float
    size: float
    pulse_phase: float
    color_shift: float


@dataclass
class FallingLeaf:
    """A holographic leaf that falls from the tree"""
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
    color_hue: float
    lifetime: float


@dataclass
class BranchSegment:
    """Individual branch segment with physics properties"""
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


class WindSystem:
    """Simulates organic wind patterns"""

    def __init__(self):
        self.time = 0.0
        self.base_strength = 15.0
        self.gust_strength = 0.0
        self.gust_target = 0.0
        self.gust_timer = 0.0

    def update(self, dt: float):
        self.time += dt
        self.gust_timer -= dt
        if self.gust_timer <= 0:
            self.gust_target = random.uniform(-25, 35)
            self.gust_timer = random.uniform(2.0, 5.0)
        self.gust_strength += (self.gust_target - self.gust_strength) * dt * 0.5

    def get_wind_force(self, x: float, y: float, height: float) -> float:
        """Get wind force at a position"""
        base = math.sin(self.time * 0.8) * 0.5
        wave1 = math.sin(self.time * 1.3 + x * 0.005) * 0.3
        wave2 = math.sin(self.time * 2.1 + y * 0.003) * 0.2
        wave3 = math.sin(self.time * 0.5) * 0.4
        height_factor = 0.3 + (1.0 - y / height) * 0.7
        total = (base + wave1 + wave2 + wave3) * self.base_strength
        total += self.gust_strength * height_factor
        return total * height_factor


class HolographicTree(QOpenGLWidget):
    """Main holographic tree visualization - GPU ACCELERATED"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Holographic Tree")

        # Enable OpenGL acceleration
        fmt = QSurfaceFormat()
        fmt.setSamples(4)
        fmt.setSwapInterval(1)
        self.setFormat(fmt)

        # Animation state
        self.time = 0.0
        self.flicker_intensity = 1.0
        self.scanline_offset = 0

        # Wind system
        self.wind = WindSystem()

        # Tree structure
        self.branches: List[BranchSegment] = []
        self.energy_nodes: List[EnergyNode] = []
        self.particles: List[Particle] = []
        self.falling_leaves: List[FallingLeaf] = []
        self.max_particles = 250
        self.max_leaves = 20

        # Cached values
        self.sorted_branches: List[BranchSegment] = []

        # 3D effect
        self.perspective_amount = 0.15
        self.root_x = 0.0
        self.root_y = 0.0

        self.regenerate_tree()

        # Animation timer (60 FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16)

        # Particle spawn timer
        self.particle_timer = QTimer(self)
        self.particle_timer.timeout.connect(self.spawn_particles)
        self.particle_timer.start(30)

        # Leaf fall timer
        self.leaf_timer = QTimer(self)
        self.leaf_timer.timeout.connect(self.spawn_falling_leaf)
        self.leaf_timer.start(600)

    def regenerate_tree(self):
        """Generate the fractal tree structure with 3D depth"""
        self.branches.clear()
        self.energy_nodes.clear()

        width = self.width() if self.width() > 0 else 1920
        height = self.height() if self.height() > 0 else 1080

        self.root_x = width / 2
        self.root_y = height - 50

        self._generate_branch(-1, -90, 200, 0, 10, 0.0)
        self.sorted_branches = sorted(self.branches, key=lambda b: b.z_depth)

    def _generate_branch(self, parent_idx: int, angle: float, length: float,
                         depth: int, max_depth: int, z_depth: float):
        if depth >= max_depth or length < 8:
            if len(self.branches) > 0 and depth >= max_depth - 2:
                parent = self.branches[parent_idx] if parent_idx >= 0 else None
                if parent:
                    self.energy_nodes.append(EnergyNode(
                        x=parent.end_x, y=parent.end_y, z=z_depth,
                        size=random.uniform(4, 12),
                        pulse_phase=random.uniform(0, 6.28),
                        color_shift=random.uniform(0, 1)
                    ))
            return

        stiffness = 1.0 - (depth / max_depth) * 0.85
        z_variation = random.uniform(-0.1, 0.1)
        new_z = max(-1, min(1, z_depth + z_variation))

        branch = BranchSegment(
            parent_index=parent_idx,
            base_angle=angle,
            length=length,
            thickness=max(2, (max_depth - depth) * 2.5),
            depth=depth,
            z_depth=new_z,
            stiffness=stiffness,
            phase_offset=depth * 0.3 + random.uniform(0, 0.5)
        )

        current_idx = len(self.branches)
        self.branches.append(branch)

        spread = 22 + depth * 2 + random.uniform(-5, 5)
        length_ratio = 0.72 + random.uniform(-0.08, 0.08)

        self._generate_branch(current_idx, angle - spread, length * length_ratio,
                            depth + 1, max_depth, new_z - 0.05)
        self._generate_branch(current_idx, angle + spread, length * length_ratio,
                            depth + 1, max_depth, new_z + 0.05)

        if depth < 5 and random.random() > 0.5:
            self._generate_branch(current_idx, angle + random.uniform(-12, 12),
                                length * 0.55, depth + 2, max_depth, new_z)

    def update_branch_positions(self):
        """Update all branch positions based on wind"""
        height = self.height() if self.height() > 0 else 1080

        for branch in self.branches:
            if branch.parent_index < 0:
                branch.start_x = self.root_x
                branch.start_y = self.root_y
            else:
                parent = self.branches[branch.parent_index]
                branch.start_x = parent.end_x
                branch.start_y = parent.end_y

            wind_force = self.wind.get_wind_force(branch.start_x, branch.start_y, height)
            flexibility = 1.0 - branch.stiffness
            wave_delay = math.sin(self.time * 2.5 - branch.phase_offset) * 0.3
            wind_offset = wind_force * flexibility * (1 + wave_delay)

            branch.current_angle = branch.base_angle + wind_offset
            branch.current_angle += math.sin(self.time * 1.8 + branch.phase_offset * 2) * (2 * flexibility)

            rad = math.radians(branch.current_angle)
            scale = 1.0 + branch.z_depth * self.perspective_amount
            scaled_length = branch.length * scale

            branch.end_x = branch.start_x + scaled_length * math.cos(rad)
            branch.end_y = branch.start_y + scaled_length * math.sin(rad)

        # Update energy nodes
        node_idx = 0
        for branch in self.branches:
            if branch.depth >= 7:
                if node_idx < len(self.energy_nodes):
                    node = self.energy_nodes[node_idx]
                    node.x = branch.end_x
                    node.y = branch.end_y
                    node.z = branch.z_depth
                    node_idx += 1

    def spawn_particles(self):
        """Spawn particles from energy nodes"""
        if len(self.particles) >= self.max_particles:
            return

        height = self.height() if self.height() > 0 else 1080

        for node in self.energy_nodes:
            if random.random() > 0.88:
                self.particles.append(Particle(
                    x=node.x + random.uniform(-10, 10),
                    y=node.y + random.uniform(-10, 10),
                    z=node.z + random.uniform(-0.2, 0.2),
                    vx=random.uniform(-0.8, 0.8),
                    vy=random.uniform(-2.5, -0.8),
                    vz=random.uniform(-0.01, 0.01),
                    size=random.uniform(2, 5),
                    alpha=random.uniform(0.5, 1.0),
                    pulse_phase=random.uniform(0, 6.28),
                    lifetime=0,
                    max_lifetime=random.uniform(80, 200),
                    color_hue=node.color_shift
                ))

        # Ambient particles
        if random.random() > 0.7:
            self.particles.append(Particle(
                x=self.root_x + random.uniform(-400, 400),
                y=height - random.uniform(50, 500),
                z=random.uniform(-0.8, 0.8),
                vx=random.uniform(-0.3, 0.3),
                vy=random.uniform(-1.0, -0.3),
                vz=random.uniform(-0.005, 0.005),
                size=random.uniform(1, 3),
                alpha=random.uniform(0.2, 0.5),
                pulse_phase=random.uniform(0, 6.28),
                lifetime=0,
                max_lifetime=random.uniform(150, 350),
                color_hue=random.uniform(0, 1)
            ))

    def spawn_falling_leaf(self):
        """Spawn a leaf from a branch tip"""
        if len(self.falling_leaves) >= self.max_leaves or not self.energy_nodes:
            return

        node = random.choice(self.energy_nodes)
        self.falling_leaves.append(FallingLeaf(
            x=node.x, y=node.y, z=node.z,
            vx=random.uniform(-0.5, 0.5),
            vy=random.uniform(0.5, 1.5),
            rotation=random.uniform(0, 360),
            rotation_speed=random.uniform(-4, 4),
            size=random.uniform(8, 18),
            wobble_phase=random.uniform(0, 6.28),
            wobble_speed=random.uniform(2, 4),
            alpha=random.uniform(0.7, 1.0),
            color_hue=node.color_shift,
            lifetime=0
        ))

    def update_animation(self):
        """Update animation state"""
        dt = 0.016
        self.time += dt

        self.wind.update(dt)
        self.update_branch_positions()

        # Flicker
        self.flicker_intensity = 0.88 + 0.12 * math.sin(self.time * 12)
        if random.random() > 0.97:
            self.flicker_intensity *= random.uniform(0.6, 0.95)

        self.scanline_offset = (self.scanline_offset + 2) % 4

        # Update particles
        height = self.height()
        width = self.width()
        self.particles = [p for p in self.particles if self._update_particle(p, height, width)]

        # Update leaves
        ground_y = height - 60
        self.falling_leaves = [l for l in self.falling_leaves
                               if self._update_leaf(l, height, width, ground_y)]

        self.update()

    def _update_particle(self, p: Particle, height: int, width: int) -> bool:
        wind = self.wind.get_wind_force(p.x, p.y, height) * 0.02
        p.vx += wind * 0.1
        p.x += p.vx
        p.y += p.vy
        p.z += p.vz
        p.lifetime += 1
        p.pulse_phase += 0.15
        p.vy -= 0.005
        return (p.lifetime < p.max_lifetime and
                -50 < p.y < height + 50 and -50 < p.x < width + 50)

    def _update_leaf(self, leaf: FallingLeaf, height: int, width: int, ground_y: float) -> bool:
        leaf.lifetime += 1
        wind = self.wind.get_wind_force(leaf.x, leaf.y, height) * 0.08
        leaf.vx += wind * 0.05
        leaf.vx *= 0.98
        leaf.vy = min(leaf.vy + 0.03, 2.5)

        wobble = math.sin(leaf.wobble_phase) * 1.5
        leaf.wobble_phase += leaf.wobble_speed * 0.016

        leaf.x += leaf.vx + wobble
        leaf.y += leaf.vy
        leaf.rotation += leaf.rotation_speed

        if leaf.y >= ground_y:
            leaf.y = ground_y
            leaf.vy = 0
            leaf.rotation_speed *= 0.9
            leaf.alpha -= 0.008
            if leaf.alpha <= 0:
                return False

        return -50 < leaf.x < width + 50

    def paintEvent(self, event):
        """Main paint event - full effects"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        # Background with grid
        self._draw_background(painter, width, height)

        # Back particles
        self._draw_particles(painter, z_min=-1.0, z_max=0.0)

        # Platform
        self._draw_platform(painter, width, height)

        # Tree layers
        self._draw_tree_shadows(painter)
        self._draw_tree_glow(painter)
        self._draw_tree(painter)
        self._draw_energy_nodes(painter)

        # Leaves
        self._draw_falling_leaves(painter)

        # Front particles
        self._draw_particles(painter, z_min=0.0, z_max=1.0)

        # Volumetric rays
        self._draw_volumetric_rays(painter, width, height)

        # Overlays
        self._draw_scanlines(painter, width, height)
        self._draw_interference(painter, width, height)
        self._draw_ui(painter, width, height)

        painter.end()

    def _draw_background(self, painter: QPainter, width: int, height: int):
        gradient = QLinearGradient(0, 0, 0, height)
        gradient.setColorAt(0, QColor(2, 8, 18))
        gradient.setColorAt(0.4, QColor(5, 15, 30))
        gradient.setColorAt(0.7, QColor(8, 20, 35))
        gradient.setColorAt(1, QColor(3, 12, 22))
        painter.fillRect(0, 0, width, height, gradient)

        # Perspective grid
        painter.setPen(QPen(QColor(0, 80, 100, 25), 1))
        horizon_y = height * 0.3
        cx = width / 2

        for i in range(-20, 21):
            painter.drawLine(int(cx + i * 80), height, int(cx), int(horizon_y))

        for i in range(15):
            ratio = i / 15
            y = height - (height - horizon_y) * (ratio ** 1.5)
            spread = (1 - ratio ** 1.5) * width * 0.8
            painter.drawLine(int(cx - spread), int(y), int(cx + spread), int(y))

    def _draw_platform(self, painter: QPainter, width: int, height: int):
        cx = width / 2
        py = height - 40
        flicker = self.flicker_intensity

        for i in range(8):
            alpha = int((60 - i * 7) * flicker)
            painter.setPen(QPen(QColor(0, 255, 255, max(0, alpha)), 3 - i * 0.3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            ew = 450 + i * 40
            eh = 50 + i * 8
            pulse = 1 + 0.05 * math.sin(self.time * 3 + i * 0.5)
            painter.drawEllipse(QRectF(cx - ew*pulse/2, py - eh*pulse/2, ew*pulse, eh*pulse))

        gradient = QRadialGradient(cx, py, 250)
        gradient.setColorAt(0, QColor(100, 255, 255, int(80 * flicker)))
        gradient.setColorAt(0.3, QColor(0, 200, 220, int(50 * flicker)))
        gradient.setColorAt(0.7, QColor(0, 100, 150, int(20 * flicker)))
        gradient.setColorAt(1, QColor(0, 50, 80, 0))
        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(cx - 250, py - 30, 500, 60))

    def _draw_tree_shadows(self, painter: QPainter):
        for branch in self.sorted_branches:
            shadow_offset = branch.z_depth * 8
            shadow_alpha = int(30 * (1 - abs(branch.z_depth)) * self.flicker_intensity)
            if shadow_alpha > 0:
                pen = QPen(QColor(0, 150, 180, shadow_alpha), branch.thickness * 0.8)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawLine(
                    QPointF(branch.start_x + shadow_offset, branch.start_y + abs(shadow_offset) * 0.5),
                    QPointF(branch.end_x + shadow_offset, branch.end_y + abs(shadow_offset) * 0.5)
                )

    def _draw_tree_glow(self, painter: QPainter):
        for branch in self.branches:
            depth_factor = 0.5 + (branch.z_depth + 1) * 0.25
            glow_intensity = (0.4 + 0.2 * math.sin(self.time * 2.5 + branch.phase_offset))
            glow_intensity *= self.flicker_intensity * depth_factor

            for glow_size in [25, 15, 8]:
                alpha = int(25 * glow_intensity * (25 / glow_size))
                hue_shift = branch.z_depth * 30
                color = QColor(
                    int(max(0, min(255, 30 + hue_shift))),
                    int(max(0, min(255, 255 - abs(hue_shift)))),
                    255, alpha
                )
                pen = QPen(color, branch.thickness + glow_size)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawLine(QPointF(branch.start_x, branch.start_y),
                               QPointF(branch.end_x, branch.end_y))

    def _draw_tree(self, painter: QPainter):
        for branch in self.sorted_branches:
            depth_ratio = branch.depth / 10
            z_factor = (branch.z_depth + 1) / 2

            pulse = 0.85 + 0.15 * math.sin(self.time * 3.5 + branch.phase_offset)
            pulse *= self.flicker_intensity

            r = int(50 + depth_ratio * 80 + (1 - z_factor) * 60)
            g = int(200 + z_factor * 55 - depth_ratio * 80)
            alpha = int((200 + z_factor * 55) * pulse)
            thickness = branch.thickness * (0.8 + z_factor * 0.4)

            pen = QPen(QColor(r, g, 255, alpha), thickness)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(branch.start_x, branch.start_y),
                           QPointF(branch.end_x, branch.end_y))

            # Core
            core_alpha = int((220 + z_factor * 35) * pulse)
            core_pen = QPen(QColor(200, 255, 255, core_alpha), max(1, thickness * 0.35))
            core_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(core_pen)
            painter.drawLine(QPointF(branch.start_x, branch.start_y),
                           QPointF(branch.end_x, branch.end_y))

    def _draw_energy_nodes(self, painter: QPainter):
        for node in self.energy_nodes:
            pulse = 0.6 + 0.4 * math.sin(self.time * 4 + node.pulse_phase)
            pulse *= self.flicker_intensity
            size = node.size * (0.7 + (node.z + 1) * 0.3)

            hue = node.color_shift
            r = int(100 + hue * 100)
            g = int(220 - hue * 50)

            gradient = QRadialGradient(node.x, node.y, size * 4)
            gradient.setColorAt(0, QColor(r, g, 255, int(150 * pulse)))
            gradient.setColorAt(0.3, QColor(r//2, g, 255, int(80 * pulse)))
            gradient.setColorAt(0.6, QColor(0, g//2, 128, int(30 * pulse)))
            gradient.setColorAt(1, QColor(0, 50, 80, 0))

            painter.setBrush(gradient)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(node.x, node.y), size * 4, size * 4)

            painter.setBrush(QColor(220, 255, 255, int(255 * pulse)))
            painter.drawEllipse(QPointF(node.x, node.y), size * 0.5, size * 0.5)

    def _draw_falling_leaves(self, painter: QPainter):
        for leaf in self.falling_leaves:
            painter.save()
            painter.translate(leaf.x, leaf.y)
            painter.rotate(leaf.rotation)

            size = leaf.size * (0.7 + (leaf.z + 1) * 0.3)
            alpha = leaf.alpha * self.flicker_intensity

            hue = leaf.color_hue
            r = int(80 + hue * 120)
            g = int(220 - hue * 40)

            # Glow
            gradient = QRadialGradient(0, 0, size * 2)
            gradient.setColorAt(0, QColor(r, g, 255, int(120 * alpha)))
            gradient.setColorAt(0.5, QColor(r//2, g, 255, int(50 * alpha)))
            gradient.setColorAt(1, QColor(0, 80, 120, 0))

            painter.setBrush(gradient)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(0, 0), size * 2, size * 2)

            # Leaf shape
            path = QPainterPath()
            path.moveTo(0, -size)
            path.lineTo(size * 0.5, 0)
            path.lineTo(0, size * 0.8)
            path.lineTo(-size * 0.5, 0)
            path.closeSubpath()

            leaf_gradient = QLinearGradient(0, -size, 0, size)
            leaf_gradient.setColorAt(0, QColor(r, g, 255, int(200 * alpha)))
            leaf_gradient.setColorAt(0.5, QColor(min(255, r + 50), min(255, g + 30), 255, int(255 * alpha)))
            leaf_gradient.setColorAt(1, QColor(r, g, 255, int(180 * alpha)))

            painter.setBrush(leaf_gradient)
            painter.setPen(QPen(QColor(200, 255, 255, int(200 * alpha)), 1))
            painter.drawPath(path)

            # Vein
            painter.setPen(QPen(QColor(220, 255, 255, int(255 * alpha)), 1))
            painter.drawLine(QPointF(0, -size * 0.8), QPointF(0, size * 0.6))

            painter.restore()

    def _draw_particles(self, painter: QPainter, z_min: float, z_max: float):
        for p in self.particles:
            if not (z_min <= p.z <= z_max):
                continue

            pulse = 0.5 + 0.5 * math.sin(p.pulse_phase)
            alpha = p.alpha * pulse * self.flicker_intensity

            if p.lifetime > p.max_lifetime * 0.7:
                alpha *= 1 - (p.lifetime - p.max_lifetime * 0.7) / (p.max_lifetime * 0.3)

            z_factor = (p.z + 1) / 2
            size = p.size * (0.5 + z_factor * 0.8)
            alpha *= (0.4 + z_factor * 0.6)

            hue = p.color_hue
            r = int(50 + hue * 150)
            g = int(200 + hue * 55)

            gradient = QRadialGradient(p.x, p.y, size * 4)
            gradient.setColorAt(0, QColor(r, g, 255, int(180 * alpha)))
            gradient.setColorAt(0.4, QColor(r//2, g, 255, int(60 * alpha)))
            gradient.setColorAt(1, QColor(0, 80, 120, 0))

            painter.setBrush(gradient)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(p.x, p.y), size * 4, size * 4)

            painter.setBrush(QColor(220, 255, 255, int(255 * alpha)))
            painter.drawEllipse(QPointF(p.x, p.y), size * 0.4, size * 0.4)

    def _draw_volumetric_rays(self, painter: QPainter, width: int, height: int):
        cx = width / 2
        base_y = height - 50

        for i in range(12):
            angle = -90 + (i - 6) * 15
            angle += math.sin(self.time * 0.5 + i) * 5

            rad = math.radians(angle)
            ray_length = 600 + math.sin(self.time * 2 + i * 0.7) * 100

            end_x = cx + ray_length * math.cos(rad)
            end_y = base_y + ray_length * math.sin(rad)

            gradient = QLinearGradient(cx, base_y, end_x, end_y)
            alpha = int(20 * self.flicker_intensity)
            gradient.setColorAt(0, QColor(0, 255, 255, alpha))
            gradient.setColorAt(0.3, QColor(0, 200, 220, alpha // 2))
            gradient.setColorAt(1, QColor(0, 100, 150, 0))

            pen = QPen(QBrush(gradient), 30 + i % 3 * 10)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(cx, base_y), QPointF(end_x, end_y))

    def _draw_scanlines(self, painter: QPainter, width: int, height: int):
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 255, 255, 12))
        for y in range(self.scanline_offset, height, 4):
            painter.drawRect(0, y, width, 1)

    def _draw_interference(self, painter: QPainter, width: int, height: int):
        if random.random() > 0.96:
            y = random.randint(0, height)
            h = random.randint(2, 12)
            painter.fillRect(0, y, width, h, QColor(0, 255, 255, random.randint(20, 60)))

        if random.random() > 0.92 and self.branches:
            branch = random.choice(self.branches)
            offset = random.uniform(2, 5)
            painter.setPen(QPen(QColor(255, 50, 100, 25), branch.thickness * 0.6))
            painter.drawLine(QPointF(branch.start_x + offset, branch.start_y),
                           QPointF(branch.end_x + offset, branch.end_y))
            painter.setPen(QPen(QColor(100, 50, 255, 25), branch.thickness * 0.6))
            painter.drawLine(QPointF(branch.start_x - offset, branch.start_y),
                           QPointF(branch.end_x - offset, branch.end_y))

    def _draw_ui(self, painter: QPainter, width: int, height: int):
        alpha = int(150 * self.flicker_intensity)
        color = QColor(0, 255, 255, alpha)
        painter.setPen(QPen(color, 2))

        # Corners
        painter.drawLine(20, 20, 100, 20)
        painter.drawLine(20, 20, 20, 100)
        painter.drawLine(width - 100, 20, width - 20, 20)
        painter.drawLine(width - 20, 20, width - 20, 100)
        painter.drawLine(20, height - 100, 20, height - 20)
        painter.drawLine(20, height - 20, 100, height - 20)
        painter.drawLine(width - 20, height - 100, width - 20, height - 20)
        painter.drawLine(width - 100, height - 20, width - 20, height - 20)

        painter.setFont(QFont("Courier", 10))
        painter.setPen(QColor(0, 255, 255, int(200 * self.flicker_intensity)))

        wind_str = f"{self.wind.base_strength + self.wind.gust_strength:.1f}"
        painter.drawText(30, 45, "HOLOGRAPHIC PROJECTION ACTIVE")
        painter.drawText(30, 65, f"PARTICLES: {len(self.particles)}")
        painter.drawText(30, 85, f"BRANCHES: {len(self.branches)}")
        painter.drawText(30, 105, f"WIND FORCE: {wind_str}")

        for i, text in enumerate(["QUANTUM COHERENCE: STABLE",
                                   "DIMENSIONAL MATRIX: SYNCHRONIZED",
                                   "PHOTON FIELD: ACTIVE"]):
            painter.drawText(30, height - 80 + i * 20, text)

        # Indicator
        pulse = 0.5 + 0.5 * math.sin(self.time * 4)
        painter.setBrush(QColor(0, 255, 255, int(200 * pulse * self.flicker_intensity)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(width - 40, 40), 8, 8)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(color, 1))
        painter.drawEllipse(QPointF(width - 40, 40), 15, 15)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.regenerate_tree()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        elif event.key() in (Qt.Key.Key_R, Qt.Key.Key_Space):
            self.regenerate_tree()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Holographic Tree")
        self.tree_widget = HolographicTree()
        self.setCentralWidget(self.tree_widget)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        # Open on primary screen
        screen = QApplication.primaryScreen()
        if screen:
            self.setGeometry(screen.geometry())
        self.showFullScreen()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        elif event.key() == Qt.Key.Key_F:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        else:
            self.tree_widget.keyPressEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
