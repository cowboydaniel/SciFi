#!/usr/bin/env python3
"""
Holographic Tree - A Sci-Fi visualization using PySide6
An otherworldly holographic tree that appears to emerge from the screen
"""

import sys
import math
import random
from dataclasses import dataclass
from typing import List

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QRadialGradient,
    QLinearGradient, QPainterPath, QFont
)


@dataclass
class Particle:
    """Floating holographic particle"""
    x: float
    y: float
    vx: float
    vy: float
    size: float
    alpha: float
    pulse_phase: float
    lifetime: float
    max_lifetime: float


@dataclass
class Branch:
    """Tree branch segment"""
    start: QPointF
    end: QPointF
    thickness: float
    depth: int
    angle: float
    glow_phase: float


class HolographicTree(QWidget):
    """Main holographic tree visualization widget"""

    # Holographic color palette
    HOLO_CYAN = QColor(0, 255, 255)
    HOLO_TEAL = QColor(0, 200, 200)
    HOLO_BLUE = QColor(50, 150, 255)
    HOLO_PURPLE = QColor(150, 100, 255)
    HOLO_GREEN = QColor(100, 255, 150)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Holographic Tree - Sci-Fi Visualization")
        self.setStyleSheet("background-color: black;")

        # Animation state
        self.time = 0.0
        self.flicker_intensity = 1.0
        self.scanline_offset = 0

        # Tree parameters
        self.branches: List[Branch] = []
        self.particles: List[Particle] = []
        self.max_particles = 200

        # Generate initial tree
        self.regenerate_tree()

        # Animation timer (60 FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16)

        # Particle spawn timer
        self.particle_timer = QTimer(self)
        self.particle_timer.timeout.connect(self.spawn_particles)
        self.particle_timer.start(50)

    def regenerate_tree(self):
        """Generate the fractal tree structure"""
        self.branches.clear()

        # Tree starts from bottom center
        width = self.width() if self.width() > 0 else 1920
        height = self.height() if self.height() > 0 else 1080

        start = QPointF(width / 2, height - 50)
        self._generate_branch(start, -90, 180, 0, 9)

    def _generate_branch(self, start: QPointF, angle: float, length: float,
                         depth: int, max_depth: int):
        """Recursively generate tree branches"""
        if depth >= max_depth or length < 4:
            return

        # Calculate end point
        rad = math.radians(angle)
        end = QPointF(
            start.x() + length * math.cos(rad),
            start.y() + length * math.sin(rad)
        )

        # Create branch
        thickness = max(1, (max_depth - depth) * 1.5)
        branch = Branch(
            start=start,
            end=end,
            thickness=thickness,
            depth=depth,
            angle=angle,
            glow_phase=random.uniform(0, 2 * math.pi)
        )
        self.branches.append(branch)

        # Branching parameters - more organic feel
        branch_angle = 25 + random.uniform(-5, 5)
        length_ratio = 0.72 + random.uniform(-0.05, 0.05)

        # Main branches
        self._generate_branch(end, angle - branch_angle, length * length_ratio,
                            depth + 1, max_depth)
        self._generate_branch(end, angle + branch_angle, length * length_ratio,
                            depth + 1, max_depth)

        # Occasional extra branch for complexity
        if depth < 4 and random.random() > 0.6:
            extra_angle = angle + random.uniform(-15, 15)
            self._generate_branch(end, extra_angle, length * 0.5,
                                depth + 2, max_depth)

    def spawn_particles(self):
        """Spawn floating holographic particles"""
        if len(self.particles) >= self.max_particles:
            return

        width = self.width() if self.width() > 0 else 1920
        height = self.height() if self.height() > 0 else 1080

        # Spawn particles near the tree
        center_x = width / 2

        for _ in range(3):
            particle = Particle(
                x=center_x + random.uniform(-300, 300),
                y=height - random.uniform(100, 600),
                vx=random.uniform(-0.5, 0.5),
                vy=random.uniform(-1.5, -0.5),
                size=random.uniform(2, 6),
                alpha=random.uniform(0.3, 0.8),
                pulse_phase=random.uniform(0, 2 * math.pi),
                lifetime=0,
                max_lifetime=random.uniform(100, 300)
            )
            self.particles.append(particle)

    def update_animation(self):
        """Update animation state"""
        self.time += 0.016

        # Holographic flicker effect
        self.flicker_intensity = 0.85 + 0.15 * math.sin(self.time * 10)
        if random.random() > 0.98:
            self.flicker_intensity *= random.uniform(0.5, 0.9)

        # Scanline animation
        self.scanline_offset = (self.scanline_offset + 2) % 4

        # Update particles
        particles_to_remove = []
        for particle in self.particles:
            particle.x += particle.vx
            particle.y += particle.vy
            particle.lifetime += 1
            particle.pulse_phase += 0.1

            # Add slight drift
            particle.vx += random.uniform(-0.02, 0.02)

            if (particle.lifetime >= particle.max_lifetime or
                particle.y < 0 or particle.x < 0 or
                particle.x > self.width()):
                particles_to_remove.append(particle)

        for particle in particles_to_remove:
            self.particles.remove(particle)

        self.update()

    def paintEvent(self, event):
        """Main paint event - render the holographic tree"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        # Draw background with subtle grid
        self._draw_background(painter, width, height)

        # Draw holographic base platform
        self._draw_platform(painter, width, height)

        # Draw the tree with glow
        self._draw_tree_glow(painter)
        self._draw_tree(painter)

        # Draw particles
        self._draw_particles(painter)

        # Draw scanlines overlay
        self._draw_scanlines(painter, width, height)

        # Draw holographic interference
        self._draw_interference(painter, width, height)

        # Draw UI elements
        self._draw_ui_elements(painter, width, height)

        painter.end()

    def _draw_background(self, painter: QPainter, width: int, height: int):
        """Draw dark background with grid"""
        # Deep space background gradient
        gradient = QLinearGradient(0, 0, 0, height)
        gradient.setColorAt(0, QColor(5, 10, 20))
        gradient.setColorAt(0.5, QColor(10, 20, 35))
        gradient.setColorAt(1, QColor(5, 15, 25))
        painter.fillRect(0, 0, width, height, gradient)

        # Draw subtle grid
        grid_color = QColor(0, 100, 100, 20)
        painter.setPen(QPen(grid_color, 1))

        grid_size = 50
        for x in range(0, width, grid_size):
            painter.drawLine(x, 0, x, height)
        for y in range(0, height, grid_size):
            painter.drawLine(0, y, width, y)

    def _draw_platform(self, painter: QPainter, width: int, height: int):
        """Draw holographic projection platform"""
        center_x = width / 2
        platform_y = height - 40

        # Elliptical platform glow
        for i in range(5):
            alpha = int(50 - i * 10) * self.flicker_intensity
            color = QColor(0, 255, 255, int(alpha))
            painter.setPen(QPen(color, 2 - i * 0.3))
            painter.setBrush(Qt.BrushStyle.NoBrush)

            ellipse_width = 400 + i * 30
            ellipse_height = 40 + i * 5
            painter.drawEllipse(
                QRectF(center_x - ellipse_width/2, platform_y - ellipse_height/2,
                       ellipse_width, ellipse_height)
            )

        # Inner glow
        gradient = QRadialGradient(center_x, platform_y, 200)
        gradient.setColorAt(0, QColor(0, 255, 255, int(60 * self.flicker_intensity)))
        gradient.setColorAt(0.5, QColor(0, 150, 200, int(30 * self.flicker_intensity)))
        gradient.setColorAt(1, QColor(0, 100, 150, 0))
        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(center_x - 200, platform_y - 20, 400, 40))

    def _draw_tree_glow(self, painter: QPainter):
        """Draw outer glow effect for the tree"""
        for branch in self.branches:
            glow_intensity = 0.3 + 0.2 * math.sin(self.time * 2 + branch.glow_phase)
            glow_intensity *= self.flicker_intensity

            # Multiple glow layers
            for glow_size in [20, 12, 6]:
                alpha = int(20 * glow_intensity * (20 / glow_size))
                color = QColor(0, 255, 255, alpha)
                pen = QPen(color, branch.thickness + glow_size)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawLine(branch.start, branch.end)

    def _draw_tree(self, painter: QPainter):
        """Draw the main tree structure"""
        for branch in self.branches:
            # Calculate color based on depth
            depth_ratio = branch.depth / 9

            # Pulse effect
            pulse = 0.8 + 0.2 * math.sin(self.time * 3 + branch.glow_phase)
            pulse *= self.flicker_intensity

            # Color gradient from cyan to purple based on depth
            r = int(50 + depth_ratio * 100)
            g = int(255 - depth_ratio * 100)
            b = 255
            alpha = int(255 * pulse)

            color = QColor(r, g, b, alpha)
            pen = QPen(color, branch.thickness)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(branch.start, branch.end)

            # Inner bright core
            core_color = QColor(200, 255, 255, int(200 * pulse))
            core_pen = QPen(core_color, max(1, branch.thickness * 0.3))
            core_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(core_pen)
            painter.drawLine(branch.start, branch.end)

    def _draw_particles(self, painter: QPainter):
        """Draw floating holographic particles"""
        for particle in self.particles:
            # Pulsing alpha
            pulse = 0.5 + 0.5 * math.sin(particle.pulse_phase)
            alpha = particle.alpha * pulse * self.flicker_intensity

            # Fade out near end of lifetime
            if particle.lifetime > particle.max_lifetime * 0.7:
                fade = 1 - (particle.lifetime - particle.max_lifetime * 0.7) / (particle.max_lifetime * 0.3)
                alpha *= fade

            # Outer glow
            gradient = QRadialGradient(particle.x, particle.y, particle.size * 3)
            gradient.setColorAt(0, QColor(0, 255, 255, int(150 * alpha)))
            gradient.setColorAt(0.5, QColor(100, 200, 255, int(50 * alpha)))
            gradient.setColorAt(1, QColor(0, 100, 150, 0))

            painter.setBrush(gradient)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(
                QPointF(particle.x, particle.y),
                particle.size * 3, particle.size * 3
            )

            # Bright core
            core_color = QColor(200, 255, 255, int(255 * alpha))
            painter.setBrush(core_color)
            painter.drawEllipse(
                QPointF(particle.x, particle.y),
                particle.size * 0.5, particle.size * 0.5
            )

    def _draw_scanlines(self, painter: QPainter, width: int, height: int):
        """Draw holographic scanline effect"""
        scanline_color = QColor(0, 255, 255, 15)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(scanline_color)

        for y in range(self.scanline_offset, height, 4):
            painter.drawRect(0, y, width, 1)

    def _draw_interference(self, painter: QPainter, width: int, height: int):
        """Draw holographic interference patterns"""
        # Occasional horizontal interference lines
        if random.random() > 0.97:
            y = random.randint(0, height)
            interference_height = random.randint(2, 8)
            color = QColor(0, 255, 255, random.randint(30, 80))
            painter.fillRect(0, y, width, interference_height, color)

        # Chromatic aberration simulation at edges
        center_x = width / 2
        center_y = height / 2

        # Draw subtle color shift at tree boundaries
        for branch in self.branches[:20]:  # Only process some branches for performance
            if random.random() > 0.95:
                offset = random.uniform(1, 3)
                alpha = int(30 * self.flicker_intensity)

                # Red shift
                painter.setPen(QPen(QColor(255, 0, 100, alpha), branch.thickness * 0.5))
                shifted_start = QPointF(branch.start.x() + offset, branch.start.y())
                shifted_end = QPointF(branch.end.x() + offset, branch.end.y())
                painter.drawLine(shifted_start, shifted_end)

    def _draw_ui_elements(self, painter: QPainter, width: int, height: int):
        """Draw sci-fi UI overlay elements"""
        font = QFont("Courier", 10)
        painter.setFont(font)

        # Top left corner decoration
        corner_color = QColor(0, 255, 255, int(150 * self.flicker_intensity))
        painter.setPen(QPen(corner_color, 2))

        # Top left
        painter.drawLine(20, 20, 80, 20)
        painter.drawLine(20, 20, 20, 80)

        # Top right
        painter.drawLine(width - 80, 20, width - 20, 20)
        painter.drawLine(width - 20, 20, width - 20, 80)

        # Bottom left
        painter.drawLine(20, height - 80, 20, height - 20)
        painter.drawLine(20, height - 20, 80, height - 20)

        # Bottom right
        painter.drawLine(width - 20, height - 80, width - 20, height - 20)
        painter.drawLine(width - 80, height - 20, width - 20, height - 20)

        # Status text
        text_color = QColor(0, 255, 255, int(200 * self.flicker_intensity))
        painter.setPen(text_color)

        painter.drawText(30, 45, "HOLOGRAPHIC PROJECTION ACTIVE")
        painter.drawText(30, 65, f"PARTICLES: {len(self.particles)}")
        painter.drawText(30, 85, f"BRANCHES: {len(self.branches)}")

        # Bottom status
        status_texts = [
            "QUANTUM COHERENCE: STABLE",
            "PHOTON MATRIX: ALIGNED",
            "DIMENSIONAL ANCHOR: LOCKED"
        ]
        for i, text in enumerate(status_texts):
            painter.drawText(30, height - 70 + i * 20, text)

        # Pulsing circle indicator
        pulse = 0.5 + 0.5 * math.sin(self.time * 4)
        indicator_color = QColor(0, 255, 255, int(200 * pulse * self.flicker_intensity))
        painter.setBrush(indicator_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(width - 40, 40), 8, 8)

        # Ring around indicator
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(corner_color, 1))
        painter.drawEllipse(QPointF(width - 40, 40), 15, 15)

    def resizeEvent(self, event):
        """Handle window resize"""
        super().resizeEvent(event)
        self.regenerate_tree()

    def keyPressEvent(self, event):
        """Handle key presses"""
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        elif event.key() == Qt.Key.Key_R:
            self.regenerate_tree()
        elif event.key() == Qt.Key.Key_Space:
            # Toggle tree regeneration
            self.regenerate_tree()


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Holographic Tree - Sci-Fi Visualization")

        # Create central widget
        self.tree_widget = HolographicTree()
        self.setCentralWidget(self.tree_widget)

        # Set window to fullscreen
        self.showFullScreen()

        # Remove window frame for immersive experience
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

    def keyPressEvent(self, event):
        """Handle key presses at window level"""
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        elif event.key() == Qt.Key.Key_F:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        else:
            self.tree_widget.keyPressEvent(event)


def main():
    """Main entry point"""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
