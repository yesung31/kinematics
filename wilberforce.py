import math
from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageDraw

from .base import (
    BaseKinematicsDataModule,
    BaseKinematicsDataset,
    RealNVP,
)

WilberforceRealNVP = RealNVP


class WilberforceSimulation:
    def __init__(
        self,
        omega_0: float = 2.0 * math.pi * 1.0,
        beat_freq: float = 0.1,
        num_energy_levels: int = 8,
        energy_levels: list[float] | None = None,
        trajectory_length: int = 65536,
        dt: float = 0.05,
    ):
        self.omega_0 = omega_0
        self.beat_freq = beat_freq
        self.num_energy_levels = num_energy_levels
        self.trajectory_length = trajectory_length
        self.dt = dt

        gamma = 2.0 * omega_0 * math.pi * beat_freq
        self.omega_1 = math.sqrt(max(1e-6, omega_0**2 - gamma))
        self.omega_2 = math.sqrt(omega_0**2 + gamma)

        if energy_levels is None:
            self.energy_levels = [0.5 * (k + 1) for k in range(self.num_energy_levels)]
        else:
            self.energy_levels = list(energy_levels)
            self.num_energy_levels = len(self.energy_levels)

        self.trajectories = self._compute_trajectories()

    def _compute_trajectories(self) -> torch.Tensor:
        t = torch.arange(self.trajectory_length, dtype=torch.float32) * self.dt
        cos_w1 = torch.cos(self.omega_1 * t)
        cos_w2 = torch.cos(self.omega_2 * t)

        trajectories = []
        for energy in self.energy_levels:
            z_0 = math.sqrt(2.0 * energy / (self.omega_0**2))
            z = 0.5 * z_0 * (cos_w1 + cos_w2)
            theta = 0.5 * z_0 * (-cos_w1 + cos_w2)
            trajectories.append(torch.stack([z, theta], dim=-1))

        return torch.stack(trajectories, dim=0)


class WilberforceRenderer:
    def __init__(
        self,
        img_size: int = 64,
        max_amplitude: float = 0.5,
    ):
        self.img_size = img_size
        self.max_amplitude = max_amplitude
        self.colors = [
            (230, 50, 50),
            (50, 200, 50),
            (50, 100, 230),
            (230, 200, 50),
        ]

    def render_frame(self, z: float, theta: float) -> torch.Tensor:
        size = self.img_size
        img = Image.new("RGB", (size, size), color=(245, 245, 248))
        draw = ImageDraw.Draw(img)

        cx = size / 2.0
        y_top = size * 0.08
        y_eq = size * 0.52
        max_pix_offset = size * 0.22
        clamped_z = max(-self.max_amplitude, min(self.max_amplitude, z))
        y_bob = y_eq + (clamped_z / self.max_amplitude) * max_pix_offset

        draw.rectangle([cx - size * 0.15, 0, cx + size * 0.15, y_top], fill=(120, 120, 125))

        r_bob = size * 0.08
        spring_bottom = y_bob - r_bob
        n_coils = 8
        coil_width = size * 0.07
        points = [(cx, y_top)]
        for i in range(1, n_coils * 2):
            frac = i / (n_coils * 2.0)
            py = y_top + frac * (spring_bottom - y_top)
            direction = -1 if i % 2 == 1 else 1
            px = cx + direction * coil_width
            points.append((px, py))
        points.append((cx, spring_bottom))
        draw.line(points, fill=(80, 80, 85), width=max(1, int(size * 0.025)))

        draw.ellipse([cx - r_bob, y_bob - r_bob, cx + r_bob, y_bob + r_bob], fill=(40, 45, 55))

        arm_len = size * 0.26
        pitch = 0.45
        tip_radius = max(1.5, size * 0.035)

        for i in range(4):
            angle = theta + i * (math.pi / 2.0)
            dx = arm_len * math.cos(angle)
            dy = arm_len * pitch * math.sin(angle)
            tip_x = cx + dx
            tip_y = y_bob + dy

            draw.line(
                [(cx, y_bob), (tip_x, tip_y)],
                fill=(70, 75, 85),
                width=max(1, int(size * 0.02)),
            )
            draw.ellipse(
                [
                    tip_x - tip_radius,
                    tip_y - tip_radius,
                    tip_x + tip_radius,
                    tip_y + tip_radius,
                ],
                fill=self.colors[i],
            )

        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def render_sequence(self, z_seq: torch.Tensor, theta_seq: torch.Tensor) -> torch.Tensor:
        frames = []
        for z_val, th_val in zip(z_seq, theta_seq):
            frames.append(self.render_frame(float(z_val), float(th_val)))
        return torch.stack(frames, dim=0)


class WilberforceDataset(BaseKinematicsDataset):
    def __init__(
        self,
        size: int = 65536,
        trajectory_length: int = 65536,
        num_energy_levels: int = 8,
        energy_levels: list[float] | None = None,
        seq_len: int = 32,
        same_energy: bool = True,
        min_time_dist: int = 0,
        max_time_dist: int | None = None,
        mixing_type: Literal["realnvp", "image", "none"] = "realnvp",
        flow_layers: int = 4,
        flow_hidden: int = 64,
        mlp_layers: int = 3,
        mlp_hidden: int = 64,
        nonlinearity_scale: float = 4.0,
        scale_max: float = 0.5,
        shift_scale: float = 0.4,
        img_size: int = 64,
        sample_seed: int = 42,
        model_seed: int = 42,
        return_z_full: bool = False,
        squeeze_single: bool = True,
        target_frame: str = "last",
        dt: float = 0.05,
        simulation: WilberforceSimulation | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            size=size,
            trajectory_length=trajectory_length,
            num_energy_levels=num_energy_levels,
            seq_len=seq_len,
            same_energy=same_energy,
            min_time_dist=min_time_dist,
            max_time_dist=max_time_dist,
            mixing_type=mixing_type,
            img_size=img_size,
            sample_seed=sample_seed,
            model_seed=model_seed,
            return_z_full=return_z_full,
            squeeze_single=squeeze_single,
            target_frame=target_frame,
            **kwargs,
        )
        self.z_dim = 2

        if simulation is not None:
            self.simulation = simulation
        else:
            self.simulation = WilberforceSimulation(
                num_energy_levels=num_energy_levels,
                energy_levels=energy_levels,
                trajectory_length=trajectory_length,
                dt=dt,
            )
        self.trajectories = self.simulation.trajectories
        self.num_energy_levels = self.simulation.num_energy_levels

        max_amp = float(self.trajectories[..., 0].abs().max())
        self.renderer = WilberforceRenderer(img_size=img_size, max_amplitude=max_amp)

        self.x_dim = kwargs.get("x_dim", 2)
        if self.mixing_type == "realnvp":
            out_features = self.x_dim if isinstance(self.x_dim, int) else self.z_dim
            self.flow = RealNVP(
                in_features=self.z_dim,
                out_features=out_features,
                num_layers=flow_layers,
                hidden_features=flow_hidden,
                seed=model_seed,
                input_scale=nonlinearity_scale,
                scale_max=scale_max,
                shift_scale=shift_scale,
                mlp_layers=mlp_layers,
                mlp_hidden=mlp_hidden,
            )
            for p in self.flow.parameters():
                p.requires_grad = False
        else:
            self.flow = None

        self.pairs = self._sample_pair_indices()


class WilberforceRealNVPDataset(WilberforceDataset):
    def __init__(self, *args, **kwargs):
        kwargs["mixing_type"] = "realnvp"
        super().__init__(*args, **kwargs)


class WilberforceImageDataset(WilberforceDataset):
    def __init__(self, *args, **kwargs):
        kwargs["mixing_type"] = "image"
        super().__init__(*args, **kwargs)


class Wilberforce(BaseKinematicsDataModule):
    dataset_cls = WilberforceDataset

    def __init__(
        self,
        size: int = 65536,
        val_size: int = 8192,
        test_size: int = 16384,
        trajectory_length: int = 65536,
        num_energy_levels: int = 8,
        seq_len: int = 32,
        same_energy: bool = True,
        min_time_dist: int = 0,
        max_time_dist: int | None = None,
        mixing_type: Literal["realnvp", "image", "none"] = "realnvp",
        flow_layers: int = 4,
        flow_hidden: int = 64,
        mlp_layers: int = 3,
        mlp_hidden: int = 64,
        nonlinearity_scale: float = 4.0,
        scale_max: float = 0.5,
        shift_scale: float = 0.4,
        img_size: int = 64,
        batch_size: int = 64,
        num_workers: int = 0,
        seed: int = 42,
        dt: float = 0.05,
        squeeze_single: bool = True,
        target_frame: str = "last",
        **kwargs,
    ):
        super().__init__(batch_size=batch_size, num_workers=num_workers)
        self.save_hyperparameters()
        self.z_dim = kwargs.get("z_dim", 3 if same_energy else 2)
        multiplier = seq_len if (seq_len > 1 or not squeeze_single) else 1
        if mixing_type == "image":
            base_x_dim = kwargs.get("x_dim", [3, img_size, img_size])
            self.x_dim = [base_x_dim[0] * multiplier, base_x_dim[1], base_x_dim[2]]
        else:
            base_x_dim = kwargs.get("x_dim", 2)
            self.x_dim = base_x_dim * multiplier
        self.dataset_kwargs = {
            "trajectory_length": trajectory_length,
            "num_energy_levels": num_energy_levels,
            "seq_len": seq_len,
            "same_energy": same_energy,
            "min_time_dist": min_time_dist,
            "max_time_dist": max_time_dist,
            "mixing_type": mixing_type,
            "flow_layers": flow_layers,
            "flow_hidden": flow_hidden,
            "mlp_layers": mlp_layers,
            "mlp_hidden": mlp_hidden,
            "nonlinearity_scale": nonlinearity_scale,
            "scale_max": scale_max,
            "shift_scale": shift_scale,
            "img_size": img_size,
            "x_dim": base_x_dim,
            "squeeze_single": squeeze_single,
            "target_frame": target_frame,
            "dt": dt,
            "model_seed": seed,
            **kwargs,
        }

    @property
    def factor_types(self) -> list[str]:
        if self.dataset_kwargs.get("same_energy", False):
            return ["continuous", "continuous", "continuous"]
        return ["continuous", "continuous"]


class WilberforceImage(Wilberforce):
    def __init__(self, *args, **kwargs):
        kwargs["mixing_type"] = "image"
        super().__init__(*args, **kwargs)
