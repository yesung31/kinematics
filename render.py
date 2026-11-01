import math

import numpy as np
import torch
from PIL import Image, ImageDraw

PALETTES = [
    {
        "bg": (245, 245, 248),
        "grid": (225, 225, 230),
        "floor": (230, 230, 235),
        "primary": (70, 75, 85),
        "mount": (100, 105, 115),
        "accent1": (230, 50, 50),
        "accent2": (50, 100, 230),
        "highlight": (60, 65, 75),
        "inner": (255, 255, 255),
        "shadow": (215, 215, 222),
        "tips": [(230, 50, 50), (50, 200, 50), (50, 100, 230), (230, 200, 50)],
    },
    {
        "bg": (248, 245, 238),
        "grid": (230, 225, 215),
        "floor": (235, 228, 218),
        "primary": (160, 95, 45),
        "mount": (130, 100, 75),
        "accent1": (220, 90, 40),
        "accent2": (40, 150, 120),
        "highlight": (90, 75, 60),
        "inner": (255, 250, 240),
        "shadow": (220, 214, 202),
        "tips": [(220, 60, 40), (40, 160, 130), (50, 110, 220), (230, 170, 30)],
    },
    {
        "bg": (28, 30, 36),
        "grid": (45, 48, 56),
        "floor": (20, 22, 27),
        "primary": (180, 190, 205),
        "mount": (140, 150, 165),
        "accent1": (255, 85, 85),
        "accent2": (70, 200, 240),
        "highlight": (110, 120, 140),
        "inner": (20, 20, 25),
        "shadow": (16, 18, 22),
        "tips": [(255, 85, 85), (80, 230, 120), (70, 180, 255), (255, 220, 60)],
    },
    {
        "bg": (238, 244, 248),
        "grid": (215, 228, 238),
        "floor": (220, 232, 242),
        "primary": (60, 75, 95),
        "mount": (90, 110, 135),
        "accent1": (210, 40, 110),
        "accent2": (30, 120, 210),
        "highlight": (55, 70, 90),
        "inner": (250, 250, 255),
        "shadow": (205, 218, 228),
        "tips": [(220, 50, 110), (40, 180, 160), (30, 110, 220), (240, 180, 40)],
    },
    {
        "bg": (244, 248, 244),
        "grid": (220, 232, 220),
        "floor": (228, 236, 228),
        "primary": (70, 85, 70),
        "mount": (110, 125, 110),
        "accent1": (210, 120, 30),
        "accent2": (40, 160, 60),
        "highlight": (65, 85, 65),
        "inner": (255, 255, 255),
        "shadow": (214, 224, 214),
        "tips": [(210, 50, 50), (45, 170, 80), (60, 100, 230), (220, 190, 40)],
    },
    {
        "bg": (32, 30, 38),
        "grid": (48, 45, 56),
        "floor": (24, 24, 29),
        "primary": (210, 180, 130),
        "mount": (150, 130, 100),
        "accent1": (250, 180, 50),
        "accent2": (190, 70, 220),
        "highlight": (140, 120, 100),
        "inner": (20, 22, 26),
        "shadow": (18, 17, 22),
        "tips": [(250, 90, 70), (70, 220, 160), (100, 160, 255), (255, 205, 50)],
    },
]


def draw_background(
    draw: ImageDraw.ImageDraw,
    size: int,
    style: str,
    colors: dict,
    spacing: int = 10,
    ruler_side: str = "left",
):
    if style == "grid":
        grid_col = colors["grid"]
        for x in range(0, size, spacing):
            draw.line([(x, 0), (x, size)], fill=grid_col, width=1)
        for y in range(0, size, spacing):
            draw.line([(0, y), (size, y)], fill=grid_col, width=1)
    elif style == "split":
        split_y = int(size * 0.72)
        draw.rectangle([0, split_y, size, size], fill=colors["floor"])
        draw.line([(0, split_y), (size, split_y)], fill=colors["grid"], width=1)
    elif style == "ruler":
        grid_col = colors["grid"]
        rx = int(size * 0.08) if ruler_side == "left" else int(size * 0.92)
        tdir = 1 if ruler_side == "left" else -1
        draw.line([(rx, int(size * 0.12)), (rx, int(size * 0.88))], fill=grid_col, width=1)
        tlen = int(size * 0.035)
        for ty in range(int(size * 0.12), int(size * 0.88) + 1, spacing):
            draw.line([(rx, ty), (rx + tdir * tlen, ty)], fill=grid_col, width=1)


def draw_shape(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    r: float,
    shape: str,
    fill: tuple,
    inner_fill: tuple | None = None,
):
    if shape == "circle":
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)
        if inner_fill:
            ri = r * 0.45
            draw.ellipse([x - ri, y - ri, x + ri, y + ri], fill=inner_fill)
    elif shape == "square":
        draw.rectangle([x - r, y - r, x + r, y + r], fill=fill)
        if inner_fill:
            ri = r * 0.45
            draw.rectangle([x - ri, y - ri, x + ri, y + ri], fill=inner_fill)
    elif shape == "diamond":
        poly = [(x, y - r * 1.25), (x + r * 1.25, y), (x, y + r * 1.25), (x - r * 1.25, y)]
        draw.polygon(poly, fill=fill)
        if inner_fill:
            ri = r * 0.55
            draw.polygon([(x, y - ri), (x + ri, y), (x, y + ri), (x - ri, y)], fill=inner_fill)
    elif shape == "ring":
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)
        if inner_fill:
            ri = r * 0.5
            draw.ellipse([x - ri, y - ri, x + ri, y + ri], fill=inner_fill)


def draw_rod(
    draw: ImageDraw.ImageDraw,
    p1: tuple[float, float],
    p2: tuple[float, float],
    color: tuple,
    style: str,
    width: int,
    gap: float = 2.0,
):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if length < 1e-4 or style == "line":
        draw.line([p1, p2], fill=color, width=width)
        return

    nx, ny = -dy / length, dx / length
    if style == "capsule":
        hw = width / 2.0
        poly = [
            (p1[0] - nx * hw, p1[1] - ny * hw),
            (p2[0] - nx * hw, p2[1] - ny * hw),
            (p2[0] + nx * hw, p2[1] + ny * hw),
            (p1[0] + nx * hw, p1[1] + ny * hw),
        ]
        draw.polygon(poly, fill=color)
        draw.ellipse([p1[0] - hw, p1[1] - hw, p1[0] + hw, p1[1] + hw], fill=color)
        draw.ellipse([p2[0] - hw, p2[1] - hw, p2[0] + hw, p2[1] + hw], fill=color)
    elif style == "double":
        g = gap / 2.0
        w = max(1, width // 2)
        draw.line(
            [(p1[0] - nx * g, p1[1] - ny * g), (p2[0] - nx * g, p2[1] - ny * g)],
            fill=color,
            width=w,
        )
        draw.line(
            [(p1[0] + nx * g, p1[1] + ny * g), (p2[0] + nx * g, p2[1] + ny * g)],
            fill=color,
            width=w,
        )


def draw_mount(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    size: int,
    style: str,
    color: tuple,
    inner_color: tuple | None = None,
    r: float = 3.0,
    y_top: float = 5.0,
    width: float = 10.0,
):
    if style == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif style == "bracket":
        poly = [
            (cx - width, 0),
            (cx + width, 0),
            (cx + width * 0.4, y_top),
            (cx - width * 0.4, y_top),
        ]
        draw.polygon(poly, fill=color)
        if inner_color:
            draw.ellipse([cx - r * 0.4, cy - r * 0.4, cx + r * 0.4, cy + r * 0.4], fill=inner_color)
    elif style == "plate":
        draw.rectangle([cx - width, 0, cx + width, y_top], fill=color)
        if inner_color:
            draw.ellipse(
                [cx - r * 0.45, cy - r * 0.45, cx + r * 0.45, cy + r * 0.45], fill=inner_color
            )
    elif style == "gantry":
        draw.rectangle([0, 0, size, y_top], fill=color)
        w_col = max(2, int(size * 0.03))
        draw.rectangle([int(size * 0.03), 0, int(size * 0.03) + w_col, size], fill=color)
        draw.rectangle(
            [size - int(size * 0.03) - w_col, 0, size - int(size * 0.03), size], fill=color
        )


class BaseKinematicsRenderer:
    def __init__(self, img_size: int = 64, seed: int = 42):
        self.img_size = img_size
        self.seed = seed

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def render_sequence(self, s1_seq: torch.Tensor, s2_seq: torch.Tensor) -> torch.Tensor:
        frames = [self.render_frame(float(s1), float(s2)) for s1, s2 in zip(s1_seq, s2_seq)]
        return torch.stack(frames, dim=0)


class DoublePendulumRenderer(BaseKinematicsRenderer):
    def __init__(self, img_size: int = 64, l1: float = 1.0, l2: float = 1.0, seed: int = 42):
        super().__init__(img_size=img_size, seed=seed)
        self.l1 = l1
        self.l2 = l2

        rng = np.random.RandomState(seed)
        scale_ratio = rng.uniform(0.34, 0.40)
        self.scale = (img_size * scale_ratio) / (l1 + l2)
        self.cx = img_size * rng.uniform(0.48, 0.52)
        self.cy = img_size * rng.uniform(0.48, 0.52)
        self.camera_tilt = rng.uniform(-0.35, 0.35)

        self.rod_style = rng.choice(["line", "capsule", "double"])
        self.rod_width = max(2, int(img_size * rng.uniform(0.025, 0.05)))
        self.rod_gap = max(2, int(img_size * 0.03))

        self.bob1_shape = rng.choice(["circle", "square", "diamond"])
        self.bob2_shape = rng.choice(["circle", "square", "diamond"])
        self.r_bob1 = max(3.0, img_size * rng.uniform(0.045, 0.07))
        self.r_bob2 = max(3.0, img_size * rng.uniform(0.045, 0.07))
        self.bob1_has_inner = bool(rng.choice([True, False]))
        self.bob2_has_inner = bool(rng.choice([True, False]))

        self.mount_style = rng.choice(["circle", "bracket", "plate"])
        self.r_pivot = max(2.0, img_size * rng.uniform(0.035, 0.055))
        self.has_joint_pin = bool(rng.choice([True, False]))

        self.has_shadow = bool(rng.choice([True, False]))
        self.shadow_offset = (
            max(1, int(img_size * rng.uniform(0.02, 0.035))),
            max(1, int(img_size * rng.uniform(0.02, 0.035))),
        )
        self.bg_style = rng.choice(["solid", "grid", "split"])
        self.grid_spacing = max(6, int(img_size * rng.uniform(0.12, 0.2)))
        self.colors = PALETTES[rng.randint(0, len(PALETTES))]

    def _draw_pendulum(
        self,
        draw: ImageDraw.ImageDraw,
        cx: float,
        cy: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        rod_color: tuple,
        bob1_color: tuple,
        bob2_color: tuple,
        mount_color: tuple,
        is_shadow: bool = False,
    ):
        draw_rod(draw, (cx, cy), (x1, y1), rod_color, self.rod_style, self.rod_width, self.rod_gap)
        draw_rod(draw, (x1, y1), (x2, y2), rod_color, self.rod_style, self.rod_width, self.rod_gap)
        draw_mount(
            draw,
            cx,
            cy,
            self.img_size,
            self.mount_style,
            mount_color,
            inner_color=None if is_shadow else self.colors["inner"],
            r=self.r_pivot,
            y_top=cy,
            width=self.r_pivot * 2.0,
        )
        if self.has_joint_pin and not is_shadow:
            r_pin = max(1.5, self.r_bob1 * 0.35)
            draw.ellipse([x1 - r_pin, y1 - r_pin, x1 + r_pin, y1 + r_pin], fill=mount_color)

        draw_shape(
            draw,
            x1,
            y1,
            self.r_bob1,
            self.bob1_shape,
            bob1_color,
            self.colors["inner"] if (self.bob1_has_inner and not is_shadow) else None,
        )
        draw_shape(
            draw,
            x2,
            y2,
            self.r_bob2,
            self.bob2_shape,
            bob2_color,
            self.colors["inner"] if (self.bob2_has_inner and not is_shadow) else None,
        )

    def render_frame(self, theta1: float, theta2: float) -> torch.Tensor:
        img = Image.new("RGB", (self.img_size, self.img_size), color=self.colors["bg"])
        draw = ImageDraw.Draw(img)
        draw_background(draw, self.img_size, self.bg_style, self.colors, spacing=self.grid_spacing)

        th1 = theta1 + self.camera_tilt
        th2 = theta2 + self.camera_tilt
        x1 = self.cx + self.scale * self.l1 * math.sin(th1)
        y1 = self.cy + self.scale * self.l1 * math.cos(th1)
        x2 = x1 + self.scale * self.l2 * math.sin(th2)
        y2 = y1 + self.scale * self.l2 * math.cos(th2)

        if self.has_shadow:
            sx, sy = self.shadow_offset
            scol = self.colors["shadow"]
            self._draw_pendulum(
                draw,
                self.cx + sx,
                self.cy + sy,
                x1 + sx,
                y1 + sy,
                x2 + sx,
                y2 + sy,
                scol,
                scol,
                scol,
                scol,
                is_shadow=True,
            )

        self._draw_pendulum(
            draw,
            self.cx,
            self.cy,
            x1,
            y1,
            x2,
            y2,
            self.colors["primary"],
            self.colors["accent1"],
            self.colors["accent2"],
            self.colors["mount"],
            is_shadow=False,
        )
        return self._to_tensor(img)


class WilberforceRenderer(BaseKinematicsRenderer):
    def __init__(self, img_size: int = 64, max_amplitude: float = 0.5, seed: int = 42):
        super().__init__(img_size=img_size, seed=seed)
        self.max_amplitude = max_amplitude

        rng = np.random.RandomState(seed)
        self.cx = img_size * rng.uniform(0.48, 0.52)
        self.y_top = img_size * rng.uniform(0.06, 0.10)
        self.y_eq = img_size * rng.uniform(0.49, 0.53)
        self.max_pix_offset = img_size * rng.uniform(0.18, 0.22)
        self.pitch = rng.uniform(0.25, 0.60)
        self.arm_len = img_size * rng.uniform(0.20, 0.26)
        self.arm_angle_offset = rng.uniform(0, 2 * math.pi)

        self.n_coils = int(rng.choice([6, 7, 8, 9, 10]))
        self.coil_width = img_size * rng.uniform(0.05, 0.08)
        self.spring_style = rng.choice(["zigzag", "helical", "double"])
        self.spring_width = max(1, int(img_size * rng.uniform(0.02, 0.035)))

        self.bob_shape = rng.choice(["cylinder", "circle", "block", "disc"])
        self.r_bob = img_size * rng.uniform(0.065, 0.09)
        self.h_bob = img_size * rng.uniform(0.06, 0.10)

        self.arm_width = max(1, int(img_size * rng.uniform(0.018, 0.03)))
        self.tip_shape = rng.choice(["circle", "square", "diamond", "ring"])
        self.tip_radius = max(2.0, img_size * rng.uniform(0.03, 0.045))

        self.mount_style = rng.choice(["plate", "bracket", "gantry"])
        self.mount_width = img_size * rng.uniform(0.12, 0.20)

        self.bg_style = rng.choice(["solid", "ruler", "grid"])
        self.ruler_side = rng.choice(["left", "right"])
        self.tick_spacing = max(4, int(img_size * 0.06))

        self.has_shadow = bool(rng.choice([True, False]))
        self.shadow_offset = (
            max(1, int(img_size * rng.uniform(0.02, 0.03))),
            max(1, int(img_size * rng.uniform(0.02, 0.03))),
        )
        self.colors = PALETTES[rng.randint(0, len(PALETTES))]
        perm = rng.permutation(4)
        self.tip_colors = [self.colors["tips"][p] for p in perm]

    def _draw_spring(
        self,
        draw: ImageDraw.ImageDraw,
        cx: float,
        y_top: float,
        spring_bottom: float,
        color: tuple,
        width: int,
    ):
        if self.spring_style == "zigzag":
            n_points = self.n_coils * 2
            pts = [(cx, y_top)]
            for i in range(1, n_points):
                frac = i / n_points
                py = y_top + frac * (spring_bottom - y_top)
                direction = -1 if i % 2 == 1 else 1
                pts.append((cx + direction * self.coil_width, py))
            pts.append((cx, spring_bottom))
            draw.line(pts, fill=color, width=width)
        elif self.spring_style == "helical":
            steps = self.n_coils * 12
            pts = [
                (
                    cx + self.coil_width * math.sin(2 * math.pi * self.n_coils * (i / steps)),
                    y_top + (i / steps) * (spring_bottom - y_top),
                )
                for i in range(steps + 1)
            ]
            draw.line(pts, fill=color, width=width)
        elif self.spring_style == "double":
            for offset in [-1.5, 1.5]:
                n_points = self.n_coils * 2
                pts = [(cx + offset, y_top)]
                for i in range(1, n_points):
                    frac = i / n_points
                    py = y_top + frac * (spring_bottom - y_top)
                    direction = -1 if i % 2 == 1 else 1
                    pts.append((cx + offset + direction * self.coil_width, py))
                pts.append((cx + offset, spring_bottom))
                draw.line(pts, fill=color, width=max(1, width - 1))

    def _draw_bob(
        self,
        draw: ImageDraw.ImageDraw,
        cx: float,
        y_bob: float,
        color: tuple,
        highlight_color: tuple,
    ):
        r, h = self.r_bob, self.h_bob
        if self.bob_shape == "cylinder":
            draw.rectangle([cx - r, y_bob - h * 0.5, cx + r, y_bob + h * 0.5], fill=color)
            draw.ellipse(
                [cx - r, y_bob - h * 0.5 - h * 0.2, cx + r, y_bob - h * 0.5 + h * 0.2],
                fill=highlight_color,
            )
            draw.ellipse(
                [cx - r, y_bob + h * 0.5 - h * 0.2, cx + r, y_bob + h * 0.5 + h * 0.2],
                fill=color,
            )
        elif self.bob_shape == "circle":
            draw.ellipse([cx - r, y_bob - r, cx + r, y_bob + r], fill=color)
        elif self.bob_shape == "block":
            draw.rectangle([cx - r, y_bob - h * 0.5, cx + r, y_bob + h * 0.5], fill=color)
        elif self.bob_shape == "disc":
            draw.ellipse(
                [cx - r * 1.3, y_bob - h * 0.35, cx + r * 1.3, y_bob + h * 0.35], fill=color
            )

    def render_frame(self, z: float, theta: float) -> torch.Tensor:
        img = Image.new("RGB", (self.img_size, self.img_size), color=self.colors["bg"])
        draw = ImageDraw.Draw(img)
        draw_background(
            draw,
            self.img_size,
            self.bg_style,
            self.colors,
            spacing=self.tick_spacing,
            ruler_side=self.ruler_side,
        )

        clamped_z = max(-self.max_amplitude, min(self.max_amplitude, z))
        y_bob = self.y_eq + (clamped_z / self.max_amplitude) * self.max_pix_offset
        spring_bottom = y_bob - self.r_bob

        arm_angles = [theta + self.arm_angle_offset + i * (math.pi / 2.0) for i in range(4)]
        arm_tips = [
            (
                i,
                self.cx + self.arm_len * math.cos(ang),
                y_bob + self.arm_len * self.pitch * math.sin(ang),
                math.sin(ang),
            )
            for i, ang in enumerate(arm_angles)
        ]

        if self.has_shadow:
            sx, sy = self.shadow_offset
            scol = self.colors["shadow"]
            self._draw_spring(
                draw,
                self.cx + sx,
                self.y_top + sy,
                spring_bottom + sy,
                scol,
                self.spring_width,
            )
            self._draw_bob(draw, self.cx + sx, y_bob + sy, scol, scol)
            for idx, tx, ty, _ in arm_tips:
                draw.line(
                    [(self.cx + sx, y_bob + sy), (tx + sx, ty + sy)],
                    fill=scol,
                    width=self.arm_width,
                )
                draw_shape(draw, tx + sx, ty + sy, self.tip_radius, self.tip_shape, scol)

        draw_mount(
            draw,
            self.cx,
            self.y_top,
            self.img_size,
            self.mount_style,
            self.colors["mount"],
            r=self.r_bob * 0.5,
            y_top=self.y_top,
            width=self.mount_width,
        )
        self._draw_spring(
            draw,
            self.cx,
            self.y_top,
            spring_bottom,
            self.colors["primary"],
            self.spring_width,
        )

        for idx, tx, ty, d in arm_tips:
            if d < 0:
                draw.line(
                    [(self.cx, y_bob), (tx, ty)],
                    fill=self.colors["primary"],
                    width=self.arm_width,
                )
                draw_shape(
                    draw,
                    tx,
                    ty,
                    self.tip_radius,
                    self.tip_shape,
                    self.tip_colors[idx],
                    inner_fill=self.colors["bg"] if self.tip_shape == "ring" else None,
                )

        self._draw_bob(draw, self.cx, y_bob, self.colors["primary"], self.colors["highlight"])

        for idx, tx, ty, d in arm_tips:
            if d >= 0:
                draw.line(
                    [(self.cx, y_bob), (tx, ty)],
                    fill=self.colors["primary"],
                    width=self.arm_width,
                )
                draw_shape(
                    draw,
                    tx,
                    ty,
                    self.tip_radius,
                    self.tip_shape,
                    self.tip_colors[idx],
                    inner_fill=self.colors["bg"] if self.tip_shape == "ring" else None,
                )

        return self._to_tensor(img)
