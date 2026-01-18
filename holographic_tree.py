#!/usr/bin/env python3
"""
Holographic Tree - A Sci-Fi visualization using PySide6
An otherworldly holographic tree that appears to emerge from the screen
With wind physics, 3D depth, and volumetric effects
"""

import sys
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QRadialGradient,
    QLinearGradient, QPainterPath, QFont, QTransform
)


@dataclass
class Particle:
    """Floating holographic particle with 3D depth"""
    x: float
    y: float
    z: float  # Depth layer (0 = back, 1 = front)
    vx: float
    vy: float
    vz: float
    size: float
    alpha: float
    pulse_phase: float
    lifetime: float
    max_lifetime: float
    color_hue: float  # For color variation


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
    parent_index: int  # -1 for root
    base_angle: float  # Static base angle
    length: float
    thickness: float
    depth: int  # Tree depth level
    z_depth: float  # 3D depth for parallax (-1 to 1)
    stiffness: float  # How much it resists wind (0-1)
    phase_offset: float  # For wave propagation

    # Computed values (updated each frame)
    current_angle: float = 0.0
    start_x: float = 0.0
    start_y: float = 0.0
    end_x: float = 0.0
    end_y: float = 0.0
    wind_offset: float = 0.0


class WindSystem:
    """Simulates organic wind patterns"""

    def __init__(self):
        self.time = 0.0
        # Multiple wind layers for complex motion
        self.base_strength = 15.0
        self.gust_strength = 0.0
        self.gust_target = 0.0
        self.gust_timer = 0.0

    def update(self, dt: float):
        self.time += dt

        # Smooth gust transitions
        self.gust_timer -= dt
        if self.gust_timer <= 0:
            self.gust_target = random.uniform(-25, 35)
            self.gust_timer = random.uniform(2.0, 5.0)

        # Lerp toward gust target
        self.gust_strength += (self.gust_target - self.gust_strength) * dt * 0.5

    def get_wind_force(self, x: float, y: float, height: float) -> float:
        """Get wind force at a position - varies spatially and temporally"""
        # Base oscillation
        base = math.sin(self.time * 0.8) * 0.5

        # Secondary waves
        wave1 = math.sin(self.time * 1.3 + x * 0.005) * 0.3
        wave2 = math.sin(self.time * 2.1 + y * 0.003) * 0.2
        wave3 = math.sin(self.time * 0.5) * 0.4  # Slow sway

        # Height factor - more wind effect higher up
        height_factor = 0.3 + (1.0 - y / height) * 0.7

        # Combine all forces
        total = (base + wave1 + wave2 + wave3) * self.base_strength
        total += self.gust_strength * height_factor

        return total * height_factor


class HolographicTree(QWidget):
    """Main holographic tree visualization with wind and 3D effects"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Holographic Tree - Sci-Fi Visualization")
        self.setStyleSheet("background-color: black;")
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

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
        self.max_particles = 300
        self.max_leaves = 25

        # 3D effect parameters
        self.perspective_amount = 0.15
        self.depth_blur_amount = 0.3

        # Tree root position
        self.root_x = 0.0
        self.root_y = 0.0

        # Generate initial tree
        self.regenerate_tree()

        # Animation timer (60 FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16)

        # Particle spawn timer
        self.particle_timer = QTimer(self)
        self.particle_timer.timeout.connect(self.spawn_particles)
        self.particle_timer.start(30)

        # Leaf fall timer - occasional leaves
        self.leaf_timer = QTimer(self)
        self.leaf_timer.timeout.connect(self.spawn_falling_leaf)
        self.leaf_timer.start(800)  # Every ~0.8 seconds

    def regenerate_tree(self):
        """Generate the fractal tree structure with 3D depth"""
        self.branches.clear()
        self.energy_nodes.clear()

        width = self.width() if self.width() > 0 else 1920
        height = self.height() if self.height() > 0 else 1080

        self.root_x = width / 2
        self.root_y = height - 50

        # Create root branch
        self._generate_branch(-1, -90, 200, 0, 10, 0.0)

    def _generate_branch(self, parent_idx: int, angle: float, length: float,
                         depth: int, max_depth: int, z_depth: float):
        """Recursively generate tree branches with depth info"""
        if depth >= max_depth or length < 8:
            # Add energy node at tips
            if len(self.branches) > 0 and depth >= max_depth - 2:
                parent = self.branches[parent_idx] if parent_idx >= 0 else None
                if parent:
                    node = EnergyNode(
                        x=parent.end_x,
                        y=parent.end_y,
                        z=z_depth,
                        size=random.uniform(4, 12),
                        pulse_phase=random.uniform(0, math.pi * 2),
                        color_shift=random.uniform(0, 1)
                    )
                    self.energy_nodes.append(node)
            return

        # Calculate stiffness - trunk is stiff, tips are flexible
        stiffness = 1.0 - (depth / max_depth) * 0.85

        # Add slight z-depth variation for 3D effect
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

        # Branching parameters
        spread = 22 + depth * 2 + random.uniform(-5, 5)
        length_ratio = 0.72 + random.uniform(-0.08, 0.08)

        # Main branches - split into slightly different z-depths for 3D
        self._generate_branch(current_idx, angle - spread, length * length_ratio,
                            depth + 1, max_depth, new_z - 0.05)
        self._generate_branch(current_idx, angle + spread, length * length_ratio,
                            depth + 1, max_depth, new_z + 0.05)

        # Extra branches for fullness
        if depth < 5 and random.random() > 0.5:
            extra_angle = angle + random.uniform(-12, 12)
            extra_z = new_z + random.uniform(-0.15, 0.15)
            self._generate_branch(current_idx, extra_angle, length * 0.55,
                                depth + 2, max_depth, extra_z)

    def update_branch_positions(self):
        """Update all branch positions based on wind"""
        height = self.height() if self.height() > 0 else 1080

        for i, branch in enumerate(self.branches):
            # Get parent end position or root
            if branch.parent_index < 0:
                branch.start_x = self.root_x
                branch.start_y = self.root_y
                parent_angle = -90
            else:
                parent = self.branches[branch.parent_index]
                branch.start_x = parent.end_x
                branch.start_y = parent.end_y
                parent_angle = parent.current_angle

            # Calculate wind effect with wave propagation
            wind_force = self.wind.get_wind_force(branch.start_x, branch.start_y, height)

            # Wave propagation delay based on depth
            wave_delay = math.sin(self.time * 2.5 - branch.phase_offset) * 0.3

            # Apply wind with stiffness resistance
            flexibility = 1.0 - branch.stiffness
            branch.wind_offset = wind_force * flexibility * (1 + wave_delay)

            # Calculate current angle
            branch.current_angle = branch.base_angle + branch.wind_offset

            # Add subtle organic movement
            organic_sway = math.sin(self.time * 1.8 + branch.phase_offset * 2) * (2 * flexibility)
            branch.current_angle += organic_sway

            # Calculate end position
            rad = math.radians(branch.current_angle)

            # Apply 3D perspective scaling based on z-depth
            perspective_scale = 1.0 + branch.z_depth * self.perspective_amount
            scaled_length = branch.length * perspective_scale

            branch.end_x = branch.start_x + scaled_length * math.cos(rad)
            branch.end_y = branch.start_y + scaled_length * math.sin(rad)

        # Update energy nodes to follow branch tips
        node_idx = 0
        for branch in self.branches:
            if branch.depth >= 7:  # Tips
                if node_idx < len(self.energy_nodes):
                    node = self.energy_nodes[node_idx]
                    node.x = branch.end_x
                    node.y = branch.end_y
                    node.z = branch.z_depth
                    node_idx += 1

    def spawn_particles(self):
        """Spawn particles along branches and at nodes"""
        if len(self.particles) >= self.max_particles:
            return

        width = self.width() if self.width() > 0 else 1920
        height = self.height() if self.height() > 0 else 1080

        # Spawn from energy nodes
        for node in self.energy_nodes:
            if random.random() > 0.85:
                particle = Particle(
                    x=node.x + random.uniform(-10, 10),
                    y=node.y + random.uniform(-10, 10),
                    z=node.z + random.uniform(-0.2, 0.2),
                    vx=random.uniform(-0.8, 0.8),
                    vy=random.uniform(-2.5, -0.8),
                    vz=random.uniform(-0.01, 0.01),
                    size=random.uniform(2, 5),
                    alpha=random.uniform(0.5, 1.0),
                    pulse_phase=random.uniform(0, math.pi * 2),
                    lifetime=0,
                    max_lifetime=random.uniform(80, 200),
                    color_hue=node.color_shift
                )
                self.particles.append(particle)

        # Spawn ambient particles
        if random.random() > 0.7:
            particle = Particle(
                x=self.root_x + random.uniform(-400, 400),
                y=height - random.uniform(50, 500),
                z=random.uniform(-0.8, 0.8),
                vx=random.uniform(-0.3, 0.3),
                vy=random.uniform(-1.0, -0.3),
                vz=random.uniform(-0.005, 0.005),
                size=random.uniform(1, 3),
                alpha=random.uniform(0.2, 0.5),
                pulse_phase=random.uniform(0, math.pi * 2),
                lifetime=0,
                max_lifetime=random.uniform(150, 350),
                color_hue=random.uniform(0, 1)
            )
            self.particles.append(particle)

    def spawn_falling_leaf(self):
        """Spawn a leaf that falls from a branch tip"""
        if len(self.falling_leaves) >= self.max_leaves:
            return

        # Pick a random energy node (branch tip) to drop from
        if not self.energy_nodes:
            return

        node = random.choice(self.energy_nodes)

        leaf = FallingLeaf(
            x=node.x,
            y=node.y,
            z=node.z + random.uniform(-0.1, 0.1),
            vx=random.uniform(-0.5, 0.5),
            vy=random.uniform(0.5, 1.5),  # Falls down
            rotation=random.uniform(0, 360),
            rotation_speed=random.uniform(-4, 4),
            size=random.uniform(8, 18),
            wobble_phase=random.uniform(0, math.pi * 2),
            wobble_speed=random.uniform(2, 4),
            alpha=random.uniform(0.7, 1.0),
            color_hue=node.color_shift,
            lifetime=0
        )
        self.falling_leaves.append(leaf)

    def update_falling_leaves(self):
        """Update falling leaf physics"""
        height = self.height() if self.height() > 0 else 1080
        ground_y = height - 60  # Just above the platform

        leaves_to_remove = []
        for leaf in self.falling_leaves:
            leaf.lifetime += 1

            # Wind effect - leaves are very light
            wind_force = self.wind.get_wind_force(leaf.x, leaf.y, height) * 0.08
            leaf.vx += wind_force * 0.05

            # Wobble side to side as it falls
            wobble = math.sin(leaf.wobble_phase) * 1.5
            leaf.wobble_phase += leaf.wobble_speed * 0.016

            # Air resistance
            leaf.vx *= 0.98
            leaf.vy *= 0.99

            # Gravity (gentle)
            leaf.vy += 0.03

            # Terminal velocity
            leaf.vy = min(leaf.vy, 2.5)

            # Apply movement
            leaf.x += leaf.vx + wobble
            leaf.y += leaf.vy
            leaf.rotation += leaf.rotation_speed

            # Check if landed
            if leaf.y >= ground_y:
                leaf.y = ground_y
                leaf.vy = 0
                leaf.vx *= 0.8
                leaf.rotation_speed *= 0.9

                # Fade out after landing
                leaf.alpha -= 0.008

                if leaf.alpha <= 0:
                    leaves_to_remove.append(leaf)

            # Remove if off screen
            if leaf.x < -50 or leaf.x > self.width() + 50:
                leaves_to_remove.append(leaf)

        for leaf in leaves_to_remove:
            if leaf in self.falling_leaves:
                self.falling_leaves.remove(leaf)

    def update_animation(self):
        """Update animation state"""
        dt = 0.016
        self.time += dt

        # Update wind
        self.wind.update(dt)

        # Update branch positions
        self.update_branch_positions()

        # Update falling leaves
        self.update_falling_leaves()

        # Holographic flicker
        self.flicker_intensity = 0.88 + 0.12 * math.sin(self.time * 12)
        if random.random() > 0.97:
            self.flicker_intensity *= random.uniform(0.6, 0.95)

        # Scanlines
        self.scanline_offset = (self.scanline_offset + 2) % 4

        # Update particles with wind influence
        particles_to_remove = []
        for particle in self.particles:
            # Wind affects particles too
            wind_effect = self.wind.get_wind_force(particle.x, particle.y, self.height()) * 0.02
            particle.vx += wind_effect * 0.1

            particle.x += particle.vx
            particle.y += particle.vy
            particle.z += particle.vz
            particle.lifetime += 1
            particle.pulse_phase += 0.15

            # Slight upward acceleration (heat rising)
            particle.vy -= 0.005

            if (particle.lifetime >= particle.max_lifetime or
                particle.y < -50 or particle.x < -50 or
                particle.x > self.width() + 50):
                particles_to_remove.append(particle)

        for particle in particles_to_remove:
            self.particles.remove(particle)

        self.update()

    def paintEvent(self, event):
        """Main paint event"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        # Background
        self._draw_background(painter, width, height)

        # Draw back-layer particles (z < 0)
        self._draw_particles(painter, z_min=-1.0, z_max=0.0)

        # Platform
        self._draw_platform(painter, width, height)

        # Tree layers - back to front for proper depth
        self._draw_tree_shadows(painter)
        self._draw_tree_glow(painter)
        self._draw_tree(painter)
        self._draw_energy_nodes(painter)

        # Falling leaves
        self._draw_falling_leaves(painter)

        # Front-layer particles (z >= 0)
        self._draw_particles(painter, z_min=0.0, z_max=1.0)

        # Volumetric light rays
        self._draw_volumetric_rays(painter, width, height)

        # Overlays
        self._draw_scanlines(painter, width, height)
        self._draw_interference(painter, width, height)
        self._draw_ui_elements(painter, width, height)

        painter.end()

    def _draw_background(self, painter: QPainter, width: int, height: int):
        """Draw background with depth grid"""
        # Deep gradient
        gradient = QLinearGradient(0, 0, 0, height)
        gradient.setColorAt(0, QColor(2, 8, 18))
        gradient.setColorAt(0.4, QColor(5, 15, 30))
        gradient.setColorAt(0.7, QColor(8, 20, 35))
        gradient.setColorAt(1, QColor(3, 12, 22))
        painter.fillRect(0, 0, width, height, gradient)

        # Perspective grid for depth
        grid_color = QColor(0, 80, 100, 25)
        painter.setPen(QPen(grid_color, 1))

        # Vertical lines converging at horizon
        horizon_y = height * 0.3
        vanishing_x = width / 2

        for i in range(-20, 21):
            bottom_x = width / 2 + i * 80
            painter.drawLine(int(bottom_x), height, int(vanishing_x), int(horizon_y))

        # Horizontal lines with perspective spacing
        for i in range(15):
            ratio = i / 15
            y = height - (height - horizon_y) * (ratio ** 1.5)
            spread = (1 - ratio ** 1.5) * width * 0.8
            painter.drawLine(int(width/2 - spread), int(y),
                           int(width/2 + spread), int(y))

    def _draw_platform(self, painter: QPainter, width: int, height: int):
        """Draw holographic projection platform with 3D effect"""
        center_x = width / 2
        platform_y = height - 40

        # Outer glow rings
        for i in range(8):
            alpha = int((60 - i * 7) * self.flicker_intensity)
            color = QColor(0, 255, 255, max(0, alpha))
            painter.setPen(QPen(color, 3 - i * 0.3))
            painter.setBrush(Qt.BrushStyle.NoBrush)

            ellipse_width = 450 + i * 40
            ellipse_height = 50 + i * 8

            # Pulsing effect
            pulse = 1 + 0.05 * math.sin(self.time * 3 + i * 0.5)
            painter.drawEllipse(
                QRectF(center_x - ellipse_width*pulse/2,
                       platform_y - ellipse_height*pulse/2,
                       ellipse_width * pulse, ellipse_height * pulse)
            )

        # Core glow
        gradient = QRadialGradient(center_x, platform_y, 250)
        gradient.setColorAt(0, QColor(100, 255, 255, int(80 * self.flicker_intensity)))
        gradient.setColorAt(0.3, QColor(0, 200, 220, int(50 * self.flicker_intensity)))
        gradient.setColorAt(0.7, QColor(0, 100, 150, int(20 * self.flicker_intensity)))
        gradient.setColorAt(1, QColor(0, 50, 80, 0))
        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QRectF(center_x - 250, platform_y - 30, 500, 60))

    def _draw_tree_shadows(self, painter: QPainter):
        """Draw shadow/depth layers for 3D effect"""
        # Sort branches by z-depth for proper layering
        sorted_branches = sorted(self.branches, key=lambda b: b.z_depth)

        for branch in sorted_branches:
            # Shadow offset based on z-depth
            shadow_offset = branch.z_depth * 8
            shadow_alpha = int(30 * (1 - abs(branch.z_depth)) * self.flicker_intensity)

            if shadow_alpha > 0:
                shadow_color = QColor(0, 150, 180, shadow_alpha)
                pen = QPen(shadow_color, branch.thickness * 0.8)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)

                painter.drawLine(
                    QPointF(branch.start_x + shadow_offset, branch.start_y + abs(shadow_offset) * 0.5),
                    QPointF(branch.end_x + shadow_offset, branch.end_y + abs(shadow_offset) * 0.5)
                )

    def _draw_tree_glow(self, painter: QPainter):
        """Draw outer glow with depth-based intensity"""
        for branch in self.branches:
            # Depth affects glow - closer = brighter
            depth_factor = 0.5 + (branch.z_depth + 1) * 0.25
            glow_intensity = (0.4 + 0.2 * math.sin(self.time * 2.5 + branch.phase_offset))
            glow_intensity *= self.flicker_intensity * depth_factor

            # Multiple glow layers
            for glow_size in [25, 15, 8]:
                alpha = int(25 * glow_intensity * (25 / glow_size))

                # Color shifts with depth
                hue_shift = branch.z_depth * 30
                color = QColor(
                    int(max(0, min(255, 30 + hue_shift))),
                    int(max(0, min(255, 255 - abs(hue_shift)))),
                    255,
                    alpha
                )

                pen = QPen(color, branch.thickness + glow_size)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawLine(
                    QPointF(branch.start_x, branch.start_y),
                    QPointF(branch.end_x, branch.end_y)
                )

    def _draw_tree(self, painter: QPainter):
        """Draw main tree with 3D depth coloring"""
        # Sort by z-depth for proper layering
        sorted_branches = sorted(self.branches, key=lambda b: b.z_depth)

        for branch in sorted_branches:
            depth_ratio = branch.depth / 10

            # Pulse with wave propagation
            pulse = 0.85 + 0.15 * math.sin(self.time * 3.5 + branch.phase_offset)
            pulse *= self.flicker_intensity

            # 3D depth coloring - back is more purple, front is more cyan/white
            z_color_factor = (branch.z_depth + 1) / 2  # 0 to 1

            r = int(50 + depth_ratio * 80 + (1 - z_color_factor) * 60)
            g = int(200 + z_color_factor * 55 - depth_ratio * 80)
            b = 255
            alpha = int((200 + z_color_factor * 55) * pulse)

            # Thickness varies with depth perception
            perceived_thickness = branch.thickness * (0.8 + z_color_factor * 0.4)

            color = QColor(r, g, b, alpha)
            pen = QPen(color, perceived_thickness)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(
                QPointF(branch.start_x, branch.start_y),
                QPointF(branch.end_x, branch.end_y)
            )

            # Bright inner core - more prominent on front branches
            core_brightness = int((220 + z_color_factor * 35) * pulse)
            core_color = QColor(200, 255, 255, core_brightness)
            core_thickness = max(1, perceived_thickness * 0.35)
            core_pen = QPen(core_color, core_thickness)
            core_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(core_pen)
            painter.drawLine(
                QPointF(branch.start_x, branch.start_y),
                QPointF(branch.end_x, branch.end_y)
            )

    def _draw_energy_nodes(self, painter: QPainter):
        """Draw glowing energy nodes at branch tips"""
        for node in self.energy_nodes:
            # Pulsing
            pulse = 0.6 + 0.4 * math.sin(self.time * 4 + node.pulse_phase)
            pulse *= self.flicker_intensity

            # Size based on z-depth
            perceived_size = node.size * (0.7 + (node.z + 1) * 0.3)

            # Color variation
            hue = node.color_shift
            r = int(100 + hue * 100)
            g = int(220 - hue * 50)
            b = 255

            # Outer glow
            gradient = QRadialGradient(node.x, node.y, perceived_size * 4)
            gradient.setColorAt(0, QColor(r, g, b, int(150 * pulse)))
            gradient.setColorAt(0.3, QColor(r//2, g, b, int(80 * pulse)))
            gradient.setColorAt(0.6, QColor(0, g//2, b//2, int(30 * pulse)))
            gradient.setColorAt(1, QColor(0, 50, 80, 0))

            painter.setBrush(gradient)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(node.x, node.y),
                              perceived_size * 4, perceived_size * 4)

            # Bright core
            core_color = QColor(220, 255, 255, int(255 * pulse))
            painter.setBrush(core_color)
            painter.drawEllipse(QPointF(node.x, node.y),
                              perceived_size * 0.5, perceived_size * 0.5)

    def _draw_falling_leaves(self, painter: QPainter):
        """Draw falling holographic leaves"""
        for leaf in self.falling_leaves:
            painter.save()

            # Translate to leaf position
            painter.translate(leaf.x, leaf.y)
            painter.rotate(leaf.rotation)

            # Size based on z-depth
            z_factor = (leaf.z + 1) / 2
            size = leaf.size * (0.7 + z_factor * 0.5)

            alpha = leaf.alpha * self.flicker_intensity

            # Color based on hue
            hue = leaf.color_hue
            r = int(80 + hue * 120)
            g = int(220 - hue * 40)
            b = 255

            # Outer glow
            gradient = QRadialGradient(0, 0, size * 2)
            gradient.setColorAt(0, QColor(r, g, b, int(120 * alpha)))
            gradient.setColorAt(0.5, QColor(r//2, g, b, int(50 * alpha)))
            gradient.setColorAt(1, QColor(0, 80, 120, 0))

            painter.setBrush(gradient)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(0, 0), size * 2, size * 2)

            # Draw leaf shape (diamond/rhombus for holographic leaf)
            leaf_path = QPainterPath()
            leaf_path.moveTo(0, -size)
            leaf_path.lineTo(size * 0.5, 0)
            leaf_path.lineTo(0, size * 0.8)
            leaf_path.lineTo(-size * 0.5, 0)
            leaf_path.closeSubpath()

            # Leaf fill with gradient
            leaf_gradient = QLinearGradient(0, -size, 0, size)
            leaf_gradient.setColorAt(0, QColor(r, g, b, int(200 * alpha)))
            leaf_gradient.setColorAt(0.5, QColor(r + 50, g + 30, b, int(255 * alpha)))
            leaf_gradient.setColorAt(1, QColor(r, g, b, int(180 * alpha)))

            painter.setBrush(leaf_gradient)
            painter.setPen(QPen(QColor(200, 255, 255, int(200 * alpha)), 1))
            painter.drawPath(leaf_path)

            # Center vein
            painter.setPen(QPen(QColor(220, 255, 255, int(255 * alpha)), 1))
            painter.drawLine(QPointF(0, -size * 0.8), QPointF(0, size * 0.6))

            painter.restore()

    def _draw_particles(self, painter: QPainter, z_min: float, z_max: float):
        """Draw particles within a z-depth range"""
        for particle in self.particles:
            if not (z_min <= particle.z <= z_max):
                continue

            # Pulsing
            pulse = 0.5 + 0.5 * math.sin(particle.pulse_phase)
            alpha = particle.alpha * pulse * self.flicker_intensity

            # Fade at end of life
            if particle.lifetime > particle.max_lifetime * 0.7:
                fade = 1 - (particle.lifetime - particle.max_lifetime * 0.7) / (particle.max_lifetime * 0.3)
                alpha *= fade

            # Size/alpha based on z-depth
            z_factor = (particle.z + 1) / 2
            perceived_size = particle.size * (0.5 + z_factor * 0.8)
            alpha *= (0.4 + z_factor * 0.6)

            # Color based on hue
            hue = particle.color_hue
            r = int(50 + hue * 150)
            g = int(200 + hue * 55)
            b = 255

            # Glow
            gradient = QRadialGradient(particle.x, particle.y, perceived_size * 4)
            gradient.setColorAt(0, QColor(r, g, b, int(180 * alpha)))
            gradient.setColorAt(0.4, QColor(r//2, g, b, int(60 * alpha)))
            gradient.setColorAt(1, QColor(0, 80, 120, 0))

            painter.setBrush(gradient)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(particle.x, particle.y),
                              perceived_size * 4, perceived_size * 4)

            # Core
            core_color = QColor(220, 255, 255, int(255 * alpha))
            painter.setBrush(core_color)
            painter.drawEllipse(QPointF(particle.x, particle.y),
                              perceived_size * 0.4, perceived_size * 0.4)

    def _draw_volumetric_rays(self, painter: QPainter, width: int, height: int):
        """Draw volumetric light rays emanating from tree"""
        center_x = width / 2
        base_y = height - 50

        num_rays = 12
        for i in range(num_rays):
            angle = -90 + (i - num_rays/2) * 15
            angle += math.sin(self.time * 0.5 + i) * 5  # Subtle sway

            rad = math.radians(angle)
            ray_length = 600 + math.sin(self.time * 2 + i * 0.7) * 100

            end_x = center_x + ray_length * math.cos(rad)
            end_y = base_y + ray_length * math.sin(rad)

            # Create gradient along ray
            gradient = QLinearGradient(center_x, base_y, end_x, end_y)
            alpha = int(20 * self.flicker_intensity)
            gradient.setColorAt(0, QColor(0, 255, 255, alpha))
            gradient.setColorAt(0.3, QColor(0, 200, 220, int(alpha * 0.5)))
            gradient.setColorAt(1, QColor(0, 100, 150, 0))

            pen = QPen(QBrush(gradient), 30 + i % 3 * 10)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(center_x, base_y), QPointF(end_x, end_y))

    def _draw_scanlines(self, painter: QPainter, width: int, height: int):
        """Draw scanline overlay"""
        scanline_color = QColor(0, 255, 255, 12)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(scanline_color)

        for y in range(self.scanline_offset, height, 4):
            painter.drawRect(0, y, width, 1)

    def _draw_interference(self, painter: QPainter, width: int, height: int):
        """Draw holographic interference"""
        # Random glitch lines
        if random.random() > 0.96:
            y = random.randint(0, height)
            h = random.randint(2, 12)
            alpha = random.randint(20, 60)
            painter.fillRect(0, y, width, h, QColor(0, 255, 255, alpha))

        # Chromatic aberration on random branches
        if random.random() > 0.92:
            branch = random.choice(self.branches) if self.branches else None
            if branch:
                offset = random.uniform(2, 5)
                # Red ghost
                painter.setPen(QPen(QColor(255, 50, 100, 25), branch.thickness * 0.6))
                painter.drawLine(
                    QPointF(branch.start_x + offset, branch.start_y),
                    QPointF(branch.end_x + offset, branch.end_y)
                )
                # Blue ghost
                painter.setPen(QPen(QColor(100, 50, 255, 25), branch.thickness * 0.6))
                painter.drawLine(
                    QPointF(branch.start_x - offset, branch.start_y),
                    QPointF(branch.end_x - offset, branch.end_y)
                )

    def _draw_ui_elements(self, painter: QPainter, width: int, height: int):
        """Draw sci-fi UI overlay"""
        font = QFont("Courier", 10)
        painter.setFont(font)

        corner_color = QColor(0, 255, 255, int(150 * self.flicker_intensity))
        painter.setPen(QPen(corner_color, 2))

        # Corners
        painter.drawLine(20, 20, 100, 20)
        painter.drawLine(20, 20, 20, 100)
        painter.drawLine(width - 100, 20, width - 20, 20)
        painter.drawLine(width - 20, 20, width - 20, 100)
        painter.drawLine(20, height - 100, 20, height - 20)
        painter.drawLine(20, height - 20, 100, height - 20)
        painter.drawLine(width - 20, height - 100, width - 20, height - 20)
        painter.drawLine(width - 100, height - 20, width - 20, height - 20)

        text_color = QColor(0, 255, 255, int(200 * self.flicker_intensity))
        painter.setPen(text_color)

        # Status
        wind_str = f"{self.wind.base_strength + self.wind.gust_strength:.1f}"
        painter.drawText(30, 45, "HOLOGRAPHIC PROJECTION ACTIVE")
        painter.drawText(30, 65, f"PARTICLES: {len(self.particles)}")
        painter.drawText(30, 85, f"BRANCHES: {len(self.branches)}")
        painter.drawText(30, 105, f"WIND FORCE: {wind_str}")

        status_texts = [
            "QUANTUM COHERENCE: STABLE",
            "DIMENSIONAL MATRIX: SYNCHRONIZED",
            "PHOTON FIELD: ACTIVE",
        ]
        for i, text in enumerate(status_texts):
            painter.drawText(30, height - 80 + i * 20, text)

        # Pulsing indicator
        pulse = 0.5 + 0.5 * math.sin(self.time * 4)
        painter.setBrush(QColor(0, 255, 255, int(200 * pulse * self.flicker_intensity)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(width - 40, 40), 8, 8)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(corner_color, 1))
        painter.drawEllipse(QPointF(width - 40, 40), 15, 15)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.regenerate_tree()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        elif event.key() == Qt.Key.Key_R:
            self.regenerate_tree()
        elif event.key() == Qt.Key.Key_Space:
            self.regenerate_tree()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Holographic Tree - Sci-Fi Visualization")
        self.tree_widget = HolographicTree()
        self.setCentralWidget(self.tree_widget)
        self.showFullScreen()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

    def keyPressEvent(self, event):
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
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
