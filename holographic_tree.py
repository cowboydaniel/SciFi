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
        self.falling_leaves: List[FallingLeaf] = []
        self.max_leaves = 18

        self.sorted_branches: List[Branch] = []
        self.tip_branches: List[Branch] = []
        self.tip_indices: List[int] = []
        self.attached_leaves: List[AttachedLeaf] = []

        self.root_x = width / 2
        self.root_y = height - 80

        self.regenerate_tree()

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

    def update(self, dt: float):
        self.time += dt
        self.wind.update(dt)

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

        self.flicker = 0.88 + 0.12 * math.sin(self.time * 10)
        if random.random() > 0.97:
            self.flicker *= 0.75

        if len(self.falling_leaves) < self.max_leaves and self.tip_branches and random.random() > 0.988:
            tb = random.choice(self.tip_branches)
            self.falling_leaves.append(FallingLeaf(
                tb.end_x, tb.end_y, tb.z_depth,
                random.uniform(-0.4, 0.4), random.uniform(0.5, 1.5),
                random.uniform(0, 360), random.uniform(-4, 4),
                random.uniform(12, 20), random.uniform(0, 6.28),
                random.uniform(2, 4), 1.0, 0
            ))

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
            points = b.segment_points or [(b.start_x, b.start_y), (b.end_x, b.end_y)]
            for i in range(len(points) - 1):
                x0, y0 = points[i]
                x1, y1 = points[i + 1]
                instances.extend([x0, y0, x1, y1, th, *color])
        return instances

    def build_leaf_instances(self) -> List[float]:
        instances: List[float] = []
        f = self.flicker
        for leaf in self.attached_leaves:
            if leaf.branch_index >= len(self.branches):
                continue
            branch = self.branches[leaf.branch_index]
            points = branch.segment_points or [(branch.start_x, branch.start_y), (branch.end_x, branch.end_y)]
            if not points:
                continue
            (attach_x, attach_y), tangent_angle = self._get_branch_point_and_angle(points, leaf.t)
            angle = tangent_angle + 180 + leaf.angle_offset + leaf.side * 6
            size = leaf.size * (0.7 + (branch.z_depth + 1) * 0.15)
            zf = (branch.z_depth + 1) / 2
            color = (0.6 + zf * 0.4, 0.95, 1.0, 0.6 * f)
            instances.extend([attach_x, attach_y, size, math.radians(angle), *color])

        for lf in self.falling_leaves:
            size = lf.size * (0.8 + (lf.z + 1) * 0.2)
            color = (0.6, 0.9, 1.0, lf.alpha * f)
            instances.extend([lf.x, lf.y, size, math.radians(lf.rotation), *color])

        return instances


class OpenGLRenderer:
    def __init__(self, window: pyglet.window.Window, tree: HolographicTree):
        self.window = window
        self.tree = tree
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE

        self.branch_program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                in vec2 in_start;
                in vec2 in_end;
                in float in_thickness;
                in vec4 in_color;

                uniform vec2 u_resolution;
                uniform float u_thickness_scale;

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
                    float half_thick = in_thickness * 0.5 * u_thickness_scale;
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

                uniform float u_glow;

                out vec4 f_color;

                void main() {
                    float edge = abs(v_local.y);
                    float core = smoothstep(1.0, 0.0, edge);
                    float glow = exp(-edge * 4.0) * u_glow;
                    float alpha = (v_color.a * core) + glow;
                    f_color = vec4(v_color.rgb, alpha);
                }
            """,
        )

        self.leaf_program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                in vec2 in_center;
                in float in_size;
                in float in_rotation;
                in vec4 in_color;

                uniform vec2 u_resolution;

                out vec2 v_uv;
                out vec4 v_color;

                void main() {
                    float c = cos(in_rotation);
                    float s = sin(in_rotation);
                    vec2 rotated = vec2(in_pos.x * c - in_pos.y * s,
                                        in_pos.x * s + in_pos.y * c);
                    vec2 world = in_center + rotated * in_size;
                    vec2 ndc = vec2((world.x / u_resolution.x) * 2.0 - 1.0,
                                    (world.y / u_resolution.y) * 2.0 - 1.0);
                    gl_Position = vec4(ndc, 0.0, 1.0);
                    v_uv = in_pos;
                    v_color = in_color;
                }
            """,
            fragment_shader="""
                #version 330
                in vec2 v_uv;
                in vec4 v_color;

                out vec4 f_color;

                void main() {
                    float dist = length(v_uv);
                    float alpha = exp(-dist * 2.5);
                    f_color = vec4(v_color.rgb, v_color.a * alpha);
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
        self.quad_vbo = self.ctx.buffer(data=array('f', quad))

        self.branch_instance_vbo = self.ctx.buffer(reserve=4 * 1024 * 32)
        self.leaf_instance_vbo = self.ctx.buffer(reserve=4 * 1024 * 16)

        self.branch_vao = self.ctx.vertex_array(
            self.branch_program,
            [
                (self.quad_vbo, "2f", "in_pos"),
                (self.branch_instance_vbo, "2f 2f f 4f /i",
                 "in_start", "in_end", "in_thickness", "in_color"),
            ],
        )

        self.leaf_vao = self.ctx.vertex_array(
            self.leaf_program,
            [
                (self.quad_vbo, "2f", "in_pos"),
                (self.leaf_instance_vbo, "2f f f 4f /i",
                 "in_center", "in_size", "in_rotation", "in_color"),
            ],
        )

    def render(self):
        width, height = self.window.get_framebuffer_size()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.clear(0.015, 0.05, 0.11)

        branch_instances = self.tree.build_branch_instances()
        if branch_instances:
            data = array('f', branch_instances)
            self.branch_instance_vbo.orphan(len(data) * 4)
            self.branch_instance_vbo.write(data)

            self.branch_program["u_resolution"].value = (width, height)

            self.branch_program["u_thickness_scale"].value = 3.0
            self.branch_program["u_glow"].value = 0.6
            self.branch_vao.render(instances=len(branch_instances) // 9)

            self.branch_program["u_thickness_scale"].value = 1.0
            self.branch_program["u_glow"].value = 0.0
            self.branch_vao.render(instances=len(branch_instances) // 9)

        leaf_instances = self.tree.build_leaf_instances()
        if leaf_instances:
            data = array('f', leaf_instances)
            self.leaf_instance_vbo.orphan(len(data) * 4)
            self.leaf_instance_vbo.write(data)
            self.leaf_program["u_resolution"].value = (width, height)
            self.leaf_vao.render(instances=len(leaf_instances) // 8)


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

        self.last_time = time.perf_counter()
        self.frame_times: List[float] = []
        pyglet.clock.schedule_interval(self.update, 1 / 120.0)

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
        self.tree.update(dt)

    def on_draw(self):
        self.clear()
        self.renderer.render()


def main():
    window = HolographicWindow()
    pyglet.app.run()


if __name__ == "__main__":
    main()
