#!/usr/bin/env python3
"""
Holographic Tree - A Sci-Fi visualization using PySide6
An otherworldly holographic tree that appears to emerge from the screen
OPTIMIZED VERSION - 60 FPS target
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
    """Floating holographic particle"""
    x: float
    y: float
    z: float
    vx: float
    vy: float
    size: float
    alpha: float
    pulse_phase: float
    lifetime: float
    max_lifetime: float


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
    """Simulates organic wind patterns - optimized"""

    def __init__(self):
        self.time = 0.0
        self.base_strength = 15.0
        self.gust_strength = 0.0
        self.gust_target = 0.0
        self.gust_timer = 0.0
        # Cache sin values
        self._cached_base = 0.0
        self._cached_wave3 = 0.0

    def update(self, dt: float):
        self.time += dt
        self.gust_timer -= dt
        if self.gust_timer <= 0:
            self.gust_target = random.uniform(-25, 35)
            self.gust_timer = random.uniform(2.0, 5.0)
        self.gust_strength += (self.gust_target - self.gust_strength) * dt * 0.5
        # Pre-calculate common values
        self._cached_base = math.sin(self.time * 0.8) * 0.5
        self._cached_wave3 = math.sin(self.time * 0.5) * 0.4

    def get_wind_force(self, y: float, height: float) -> float:
        """Simplified wind force - less spatial variation for speed"""
        height_factor = 0.3 + (1.0 - y / height) * 0.7
        total = (self._cached_base + self._cached_wave3) * self.base_strength
        total += self.gust_strength * height_factor
        return total * height_factor


class HolographicTree(QOpenGLWidget):
    """Main holographic tree visualization - GPU ACCELERATED"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Holographic Tree")

        # Enable OpenGL acceleration
        fmt = QSurfaceFormat()
        fmt.setSamples(4)  # Anti-aliasing
        fmt.setSwapInterval(1)  # VSync
        self.setFormat(fmt)

        # Animation state
        self.time = 0.0
        self.flicker_intensity = 1.0
        self.frame_count = 0

        # Wind system
        self.wind = WindSystem()

        # Tree structure
        self.branches: List[BranchSegment] = []
        self.particles: List[Particle] = []
        self.falling_leaves: List[FallingLeaf] = []
        self.max_particles = 80  # Reduced from 300
        self.max_leaves = 15

        # Cached values
        self.sorted_branches: List[BranchSegment] = []
        self.tip_branches: List[BranchSegment] = []

        # 3D effect
        self.perspective_amount = 0.15
        self.root_x = 0.0
        self.root_y = 0.0

        # Pre-computed colors
        self.glow_color = QColor(0, 200, 255, 40)
        self.core_color = QColor(180, 255, 255)
        self.particle_color = QColor(100, 255, 255)

        self.regenerate_tree()

        # Single timer at 60 FPS
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16)

    def regenerate_tree(self):
        """Generate the fractal tree structure"""
        self.branches.clear()

        width = self.width() if self.width() > 0 else 1920
        height = self.height() if self.height() > 0 else 1080

        self.root_x = width / 2
        self.root_y = height - 50

        self._generate_branch(-1, -90, 180, 0, 9, 0.0)  # Reduced depth from 10 to 9

        # Cache sorted branches and tips
        self.sorted_branches = sorted(self.branches, key=lambda b: b.z_depth)
        self.tip_branches = [b for b in self.branches if b.depth >= 6]

    def _generate_branch(self, parent_idx: int, angle: float, length: float,
                         depth: int, max_depth: int, z_depth: float):
        if depth >= max_depth or length < 10:
            return

        stiffness = 1.0 - (depth / max_depth) * 0.85
        z_variation = random.uniform(-0.1, 0.1)
        new_z = max(-1, min(1, z_depth + z_variation))

        branch = BranchSegment(
            parent_index=parent_idx,
            base_angle=angle,
            length=length,
            thickness=max(2, (max_depth - depth) * 2.2),
            depth=depth,
            z_depth=new_z,
            stiffness=stiffness,
            phase_offset=depth * 0.3 + random.uniform(0, 0.5)
        )

        current_idx = len(self.branches)
        self.branches.append(branch)

        spread = 24 + depth * 2 + random.uniform(-5, 5)
        length_ratio = 0.7 + random.uniform(-0.05, 0.05)

        self._generate_branch(current_idx, angle - spread, length * length_ratio,
                            depth + 1, max_depth, new_z - 0.05)
        self._generate_branch(current_idx, angle + spread, length * length_ratio,
                            depth + 1, max_depth, new_z + 0.05)

        if depth < 4 and random.random() > 0.6:
            self._generate_branch(current_idx, angle + random.uniform(-12, 12),
                                length * 0.5, depth + 2, max_depth, new_z)

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

            # Simplified wind calculation
            wind_force = self.wind.get_wind_force(branch.start_y, height)
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

    def spawn_particles_and_leaves(self):
        """Spawn particles and leaves - combined for efficiency"""
        # Spawn leaf occasionally
        if len(self.falling_leaves) < self.max_leaves and random.random() > 0.97:
            if self.tip_branches:
                branch = random.choice(self.tip_branches)
                self.falling_leaves.append(FallingLeaf(
                    x=branch.end_x, y=branch.end_y,
                    z=branch.z_depth,
                    vx=random.uniform(-0.5, 0.5),
                    vy=random.uniform(0.5, 1.5),
                    rotation=random.uniform(0, 360),
                    rotation_speed=random.uniform(-4, 4),
                    size=random.uniform(10, 16),
                    wobble_phase=random.uniform(0, 6.28),
                    wobble_speed=random.uniform(2, 4),
                    alpha=1.0,
                    lifetime=0
                ))

        # Spawn particles
        if len(self.particles) < self.max_particles and random.random() > 0.7:
            if self.tip_branches:
                branch = random.choice(self.tip_branches)
                self.particles.append(Particle(
                    x=branch.end_x + random.uniform(-15, 15),
                    y=branch.end_y + random.uniform(-15, 15),
                    z=branch.z_depth,
                    vx=random.uniform(-0.5, 0.5),
                    vy=random.uniform(-2, -0.5),
                    size=random.uniform(2, 5),
                    alpha=random.uniform(0.5, 1.0),
                    pulse_phase=random.uniform(0, 6.28),
                    lifetime=0,
                    max_lifetime=random.uniform(60, 150)
                ))

    def update_particles_and_leaves(self):
        """Update particles and leaves physics"""
        height = self.height() if self.height() > 0 else 1080
        ground_y = height - 60
        width = self.width()

        # Update particles
        self.particles = [p for p in self.particles if self._update_particle(p, height, width)]

        # Update leaves
        self.falling_leaves = [l for l in self.falling_leaves
                               if self._update_leaf(l, height, width, ground_y)]

    def _update_particle(self, p: Particle, height: int, width: int) -> bool:
        p.x += p.vx
        p.y += p.vy
        p.lifetime += 1
        p.pulse_phase += 0.15
        p.vy -= 0.008
        return (p.lifetime < p.max_lifetime and
                0 < p.y < height and 0 < p.x < width)

    def _update_leaf(self, leaf: FallingLeaf, height: int, width: int, ground_y: float) -> bool:
        leaf.lifetime += 1
        wind = self.wind.get_wind_force(leaf.y, height) * 0.003
        leaf.vx += wind
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
            leaf.rotation_speed *= 0.95
            leaf.alpha -= 0.01
            if leaf.alpha <= 0:
                return False

        return 0 < leaf.x < width

    def update_animation(self):
        """Main animation update"""
        self.time += 0.016
        self.frame_count += 1

        self.wind.update(0.016)
        self.update_branch_positions()

        # Spawn less frequently
        if self.frame_count % 2 == 0:
            self.spawn_particles_and_leaves()

        self.update_particles_and_leaves()

        # Flicker
        self.flicker_intensity = 0.9 + 0.1 * math.sin(self.time * 10)
        if random.random() > 0.98:
            self.flicker_intensity *= 0.8

        self.update()

    def paintEvent(self, event):
        """Optimized paint event"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        # Background - simple gradient
        self._draw_background(painter, width, height)

        # Platform - simplified
        self._draw_platform(painter, width, height)

        # Tree - single pass with glow built in
        self._draw_tree_optimized(painter)

        # Particles - simple circles
        self._draw_particles_simple(painter)

        # Leaves
        self._draw_leaves(painter)

        # Minimal overlay
        self._draw_scanlines_fast(painter, width, height)
        self._draw_ui(painter, width, height)

        painter.end()

    def _draw_background(self, painter: QPainter, width: int, height: int):
        gradient = QLinearGradient(0, 0, 0, height)
        gradient.setColorAt(0, QColor(2, 8, 18))
        gradient.setColorAt(0.5, QColor(5, 15, 30))
        gradient.setColorAt(1, QColor(3, 12, 22))
        painter.fillRect(0, 0, width, height, gradient)

        # Simple grid - fewer lines
        painter.setPen(QPen(QColor(0, 60, 80, 20), 1))
        horizon_y = int(height * 0.35)
        cx = width // 2

        for i in range(-8, 9, 2):
            painter.drawLine(cx + i * 100, height, cx, horizon_y)

        for i in range(0, 8):
            y = height - int((height - horizon_y) * (i / 8) ** 1.3)
            spread = int((1 - (i / 8) ** 1.3) * width * 0.6)
            painter.drawLine(cx - spread, y, cx + spread, y)

    def _draw_platform(self, painter: QPainter, width: int, height: int):
        cx = width / 2
        py = height - 40
        flicker = self.flicker_intensity

        # Just 3 rings instead of 8
        for i in range(3):
            alpha = int((50 - i * 15) * flicker)
            painter.setPen(QPen(QColor(0, 255, 255, alpha), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            ew = 400 + i * 60
            eh = 40 + i * 10
            painter.drawEllipse(QRectF(cx - ew/2, py - eh/2, ew, eh))

        # Core glow - single gradient
        gradient = QRadialGradient(cx, py, 200)
        gradient.setColorAt(0, QColor(50, 255, 255, int(50 * flicker)))
        gradient.setColorAt(1, QColor(0, 100, 150, 0))
        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(cx - 200, py - 25, 400, 50))

    def _draw_tree_optimized(self, painter: QPainter):
        """Draw tree with single glow pass"""
        flicker = self.flicker_intensity

        # Glow layer - one pass with thick pen
        glow_pen = QPen(self.glow_color, 1)
        glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        for branch in self.sorted_branches:
            # Glow - just one layer
            glow_alpha = int(35 * flicker * (0.5 + (branch.z_depth + 1) * 0.25))
            glow_pen.setColor(QColor(0, 200, 255, glow_alpha))
            glow_pen.setWidthF(branch.thickness + 12)
            painter.setPen(glow_pen)
            painter.drawLine(QPointF(branch.start_x, branch.start_y),
                           QPointF(branch.end_x, branch.end_y))

        # Main branches
        for branch in self.sorted_branches:
            z_factor = (branch.z_depth + 1) / 2
            pulse = 0.85 + 0.15 * math.sin(self.time * 3 + branch.phase_offset)
            pulse *= flicker

            # Main color
            r = int(60 + (1 - z_factor) * 50)
            g = int(200 + z_factor * 55)
            alpha = int((200 + z_factor * 55) * pulse)
            thickness = branch.thickness * (0.85 + z_factor * 0.3)

            painter.setPen(QPen(QColor(r, g, 255, alpha), thickness,
                               cap=Qt.PenCapStyle.RoundCap))
            painter.drawLine(QPointF(branch.start_x, branch.start_y),
                           QPointF(branch.end_x, branch.end_y))

            # Core - only for thicker branches
            if branch.thickness > 4:
                core_alpha = int(220 * pulse)
                painter.setPen(QPen(QColor(200, 255, 255, core_alpha),
                                   max(1, thickness * 0.3),
                                   cap=Qt.PenCapStyle.RoundCap))
                painter.drawLine(QPointF(branch.start_x, branch.start_y),
                               QPointF(branch.end_x, branch.end_y))

    def _draw_particles_simple(self, painter: QPainter):
        """Draw particles as simple glowing circles - no gradients"""
        painter.setPen(Qt.PenStyle.NoPen)

        for p in self.particles:
            pulse = 0.5 + 0.5 * math.sin(p.pulse_phase)
            alpha = p.alpha * pulse * self.flicker_intensity

            if p.lifetime > p.max_lifetime * 0.7:
                alpha *= 1 - (p.lifetime - p.max_lifetime * 0.7) / (p.max_lifetime * 0.3)

            z_factor = (p.z + 1) / 2
            size = p.size * (0.6 + z_factor * 0.6)

            # Outer glow - simple circle
            painter.setBrush(QColor(50, 200, 255, int(50 * alpha)))
            painter.drawEllipse(QPointF(p.x, p.y), size * 2.5, size * 2.5)

            # Core
            painter.setBrush(QColor(180, 255, 255, int(255 * alpha)))
            painter.drawEllipse(QPointF(p.x, p.y), size * 0.5, size * 0.5)

    def _draw_leaves(self, painter: QPainter):
        """Draw falling leaves"""
        for leaf in self.falling_leaves:
            painter.save()
            painter.translate(leaf.x, leaf.y)
            painter.rotate(leaf.rotation)

            size = leaf.size * (0.8 + (leaf.z + 1) * 0.2)
            alpha = leaf.alpha * self.flicker_intensity

            # Glow
            painter.setBrush(QColor(80, 220, 255, int(60 * alpha)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(0, 0), size * 1.5, size * 1.5)

            # Leaf shape
            path = QPainterPath()
            path.moveTo(0, -size)
            path.lineTo(size * 0.5, 0)
            path.lineTo(0, size * 0.8)
            path.lineTo(-size * 0.5, 0)
            path.closeSubpath()

            painter.setBrush(QColor(100, 230, 255, int(200 * alpha)))
            painter.setPen(QPen(QColor(180, 255, 255, int(220 * alpha)), 1))
            painter.drawPath(path)

            painter.restore()

    def _draw_scanlines_fast(self, painter: QPainter, width: int, height: int):
        """Minimal scanline effect - just a few lines"""
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 255, 255, 8))

        # Draw every 8th line instead of every 4th
        offset = int(self.time * 50) % 8
        for y in range(offset, height, 8):
            painter.drawRect(0, y, width, 1)

    def _draw_ui(self, painter: QPainter, width: int, height: int):
        """Minimal UI overlay"""
        alpha = int(150 * self.flicker_intensity)
        color = QColor(0, 255, 255, alpha)
        painter.setPen(QPen(color, 2))

        # Corners
        painter.drawLine(20, 20, 80, 20)
        painter.drawLine(20, 20, 20, 80)
        painter.drawLine(width - 80, 20, width - 20, 20)
        painter.drawLine(width - 20, 20, width - 20, 80)

        # Text
        painter.setFont(QFont("Courier", 10))
        painter.setPen(QColor(0, 255, 255, int(200 * self.flicker_intensity)))
        painter.drawText(30, 45, "HOLOGRAPHIC PROJECTION")
        painter.drawText(30, 65, f"PARTICLES: {len(self.particles)}")

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
        self.showFullScreen()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

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
