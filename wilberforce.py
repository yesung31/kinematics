import math
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class RealNVPCoupling(nn.Module):
    def __init__(
        self,
        features: int = 2,
        mask_even: bool = True,
        hidden: int = 64,
        input_scale: float = 4.0,
        scale_max: float = 0.5,
        shift_scale: float = 0.4,
        gain: float = 1.4,
    ):
        super().__init__()
        self.features = features
        self.mask_even = mask_even
        self.input_scale = input_scale
        self.scale_max = scale_max
        self.shift_scale = shift_scale

        d = features // 2
        in_dim = d if mask_even else (features - d)
        out_dim = (features - d) if mask_even else d

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim * 2),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.uniform_(m.bias, -0.15, 0.15)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.features // 2
        if self.mask_even:
            x_id, x_tr = x[..., :d], x[..., d:]
        else:
            x_tr, x_id = x[..., :d], x[..., d:]

        st = self.net(x_id * self.input_scale)
        s, t = st.chunk(2, dim=-1)
        scale = torch.exp(self.scale_max * torch.tanh(s))
        shift = self.shift_scale * torch.tanh(t)
        y_tr = x_tr * scale + shift

        if self.mask_even:
            return torch.cat([x_id, y_tr], dim=-1)
        return torch.cat([y_tr, x_id], dim=-1)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        d = self.features // 2
        if self.mask_even:
            y_id, y_tr = y[..., :d], y[..., d:]
        else:
            y_tr, y_id = y[..., :d], y[..., d:]

        st = self.net(y_id * self.input_scale)
        s, t = st.chunk(2, dim=-1)
        scale = torch.exp(self.scale_max * torch.tanh(s))
        shift = self.shift_scale * torch.tanh(t)
        x_tr = (y_tr - shift) / scale

        if self.mask_even:
            return torch.cat([y_id, x_tr], dim=-1)
        return torch.cat([x_tr, y_id], dim=-1)


class SmoothLeakyReLU(nn.Module):
    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x + (1.0 - self.alpha) * torch.nn.functional.softplus(x)


class WilberforceRealNVP(nn.Module):
    def __init__(
        self,
        features: int | None = None,
        in_features: int = 2,
        out_features: int | None = None,
        num_layers: int = 4,
        hidden_features: int = 64,
        seed: int = 42,
        input_scale: float = 4.0,
        scale_max: float = 0.5,
        shift_scale: float = 0.4,
        gain: float = 1.4,
        mlp_layers: int = 3,
        mlp_hidden: int = 64,
    ):
        super().__init__()
        if features is not None:
            in_features = features
        self.in_features = in_features
        self.out_features = out_features if out_features is not None else in_features
        if self.out_features < self.in_features:
            raise ValueError(
                f"out_features ({self.out_features}) must be >= in_features ({self.in_features})"
            )

        torch.manual_seed(seed)
        self.layers = nn.ModuleList(
            [
                RealNVPCoupling(
                    features=in_features,
                    mask_even=(i % 2 == 0),
                    hidden=hidden_features,
                    input_scale=input_scale,
                    scale_max=scale_max,
                    shift_scale=shift_scale,
                    gain=gain,
                )
                for i in range(num_layers)
            ]
        )

        extra_dims = self.out_features - self.in_features
        if extra_dims > 0:
            mlp_modules = []
            curr_dim = in_features
            for _ in range(mlp_layers - 1):
                lin = nn.Linear(curr_dim, mlp_hidden)
                nn.init.xavier_normal_(lin.weight, gain=1.0)
                nn.init.zeros_(lin.bias)
                mlp_modules.append(lin)
                mlp_modules.append(SmoothLeakyReLU(alpha=0.2))
                curr_dim = mlp_hidden

            final_lin = nn.Linear(curr_dim, extra_dims)
            nn.init.xavier_normal_(final_lin.weight, gain=1.0)
            nn.init.zeros_(final_lin.bias)
            mlp_modules.append(final_lin)
            self.mlp = nn.Sequential(*mlp_modules)

            gen = torch.Generator().manual_seed(seed)
            rand_mat = torch.randn(self.out_features, self.out_features, generator=gen)
            q, r = torch.linalg.qr(rand_mat)
            d_diag = torch.diag(r).sign()
            d_diag[d_diag == 0] = 1.0
            q = q * d_diag.unsqueeze(0)
            self.register_buffer("Q", q)
        else:
            self.mlp = None
            self.register_buffer("Q", torch.eye(self.in_features))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        orig_shape = list(z.shape)
        orig_dtype = z.dtype
        x = z.view(-1, orig_shape[-1]).to(dtype=self.Q.dtype)
        with torch.autocast(device_type=z.device.type, enabled=False):
            u = x
            for layer in self.layers:
                u = layer.forward(u)

            if self.out_features > self.in_features:
                v = self.mlp(x)
                w = torch.cat([u, v], dim=-1)
                res = w @ self.Q.T.to(device=w.device, dtype=w.dtype)
            else:
                res = u
        orig_shape[-1] = self.out_features
        return res.view(*orig_shape).to(orig_dtype)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = list(x.shape)
        orig_dtype = x.dtype
        x_flat = x.view(-1, orig_shape[-1]).to(dtype=self.Q.dtype)
        with torch.autocast(device_type=x.device.type, enabled=False):
            if self.out_features > self.in_features:
                w = x_flat @ self.Q.to(device=x_flat.device, dtype=x_flat.dtype)
                u = w[..., : self.in_features]
            else:
                u = x_flat

            z = u
            for layer in reversed(self.layers):
                z = layer.inverse(z)
        orig_shape[-1] = self.in_features
        return z.view(*orig_shape).to(orig_dtype)


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


class WilberforceDataset(Dataset):
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
        *args,
        **kwargs,
    ):
        super().__init__()
        self.size = size
        self.trajectory_length = trajectory_length
        self.num_energy_levels = num_energy_levels
        self.seq_len = seq_len
        self.same_energy = same_energy
        self.min_time_dist = min_time_dist
        self.max_time_dist = max_time_dist
        self.mixing_type = mixing_type
        self.img_size = img_size
        self.sample_seed = sample_seed
        self.model_seed = model_seed
        self.return_z_full = return_z_full
        self.squeeze_single = squeeze_single
        self.target_frame = target_frame
        self.z_dim = 2

        self.simulation = WilberforceSimulation(
            num_energy_levels=num_energy_levels,
            energy_levels=energy_levels,
            trajectory_length=trajectory_length,
            dt=dt,
        )
        self.trajectories = self.simulation.trajectories

        max_amp = float(self.trajectories[..., 0].abs().max())
        self.renderer = WilberforceRenderer(img_size=img_size, max_amplitude=max_amp)

        self.x_dim = kwargs.get("x_dim", 2)
        if self.mixing_type == "realnvp":
            out_features = self.x_dim if isinstance(self.x_dim, int) else self.z_dim
            self.flow = WilberforceRealNVP(
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

    def _sample_t2(self, t1: int, max_start: int, gen: torch.Generator) -> int:
        if self.min_time_dist == 0 and self.max_time_dist is None:
            return torch.randint(0, max_start + 1, (1,), generator=gen).item()

        max_d = max_start if self.max_time_dist is None else self.max_time_dist
        min_d = self.min_time_dist

        left_start = max(0, t1 - max_d)
        left_end = t1 - min_d
        len_left = max(0, left_end - left_start + 1)

        right_start = t1 + min_d
        right_end = min(max_start, t1 + max_d)
        len_right = max(0, right_end - right_start + 1)

        total_valid = len_left + len_right
        if total_valid <= 0:
            return torch.randint(0, max_start + 1, (1,), generator=gen).item()

        k = torch.randint(0, total_valid, (1,), generator=gen).item()
        if k < len_left:
            return left_start + k
        return right_start + (k - len_left)

    def _sample_pair_indices(self) -> torch.Tensor:
        gen = torch.Generator()
        gen.manual_seed(self.sample_seed)

        max_start = self.trajectory_length - self.seq_len
        pairs = torch.empty((self.size, 4), dtype=torch.long)

        for i in range(self.size):
            if self.same_energy:
                e1 = torch.randint(0, self.num_energy_levels, (1,), generator=gen).item()
                e2 = e1
            else:
                e1 = torch.randint(0, self.num_energy_levels, (1,), generator=gen).item()
                e2 = torch.randint(0, self.num_energy_levels, (1,), generator=gen).item()

            t1 = torch.randint(0, max_start + 1, (1,), generator=gen).item()
            t2 = self._sample_t2(t1, max_start, gen)
            pairs[i] = torch.tensor([e1, e2, t1, t2])

        return pairs

    def __len__(self) -> int:
        return self.size

    def _apply_mixing(self, z: torch.Tensor) -> torch.Tensor:
        if self.mixing_type == "realnvp":
            with torch.no_grad():
                return self.flow(z)
        elif self.mixing_type == "image":
            if z.ndim == 1:
                return self.renderer.render_frame(float(z[0]), float(z[1]))
            else:
                return self.renderer.render_sequence(z[:, 0], z[:, 1])
        return z

    def _extract_target_z(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 1:
            return z
        if self.target_frame == "last":
            return z[-1]
        elif self.target_frame == "center":
            return z[len(z) // 2]
        elif self.target_frame == "first":
            return z[0]
        return z

    def __getitem__(self, idx: int):
        e1, e2, t1, t2 = self.pairs[idx].tolist()

        z1 = self.trajectories[e1, t1 : t1 + self.seq_len]
        z2 = self.trajectories[e2, t2 : t2 + self.seq_len]

        if self.seq_len == 1 and self.squeeze_single:
            z1 = z1.squeeze(0)
            z2 = z2.squeeze(0)

        x1 = self._apply_mixing(z1)
        x2 = self._apply_mixing(z2)

        if self.seq_len > 1 or not self.squeeze_single:
            if self.mixing_type == "image":
                x1 = x1.reshape(-1, x1.shape[-2], x1.shape[-1])
                x2 = x2.reshape(-1, x2.shape[-2], x2.shape[-1])
            else:
                x1 = x1.reshape(-1)
                x2 = x2.reshape(-1)

        target_z1 = self._extract_target_z(z1)
        target_z2 = self._extract_target_z(z2)

        if self.same_energy:
            energy_val1 = torch.tensor(
                [self.simulation.energy_levels[e1]],
                dtype=target_z1.dtype,
                device=target_z1.device,
            )
            energy_val2 = torch.tensor(
                [self.simulation.energy_levels[e2]],
                dtype=target_z2.dtype,
                device=target_z2.device,
            )
            if target_z1.ndim == 1:
                target_z1 = torch.cat([target_z1, energy_val1], dim=-1)
                target_z2 = torch.cat([target_z2, energy_val2], dim=-1)
            else:
                target_z1 = torch.cat(
                    [target_z1, energy_val1.expand(target_z1.shape[0], 1)], dim=-1
                )
                target_z2 = torch.cat(
                    [target_z2, energy_val2.expand(target_z2.shape[0], 1)], dim=-1
                )

        mask = torch.ones_like(target_z1)

        if self.return_z_full:
            return [x1, x2], [target_z1, target_z2], mask, target_z1
        return [x1, x2], [target_z1, target_z2], mask


class WilberforceRealNVPDataset(WilberforceDataset):
    def __init__(self, *args, **kwargs):
        kwargs["mixing_type"] = "realnvp"
        super().__init__(*args, **kwargs)


class WilberforceImageDataset(WilberforceDataset):
    def __init__(self, *args, **kwargs):
        kwargs["mixing_type"] = "image"
        super().__init__(*args, **kwargs)


class Wilberforce(LightningDataModule):
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
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_workers = num_workers
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

    def setup(self, stage=None):
        if stage in ("fit", "validate", None):
            train_kwargs = dict(self.dataset_kwargs)
            train_kwargs["size"] = self.hparams.size
            train_kwargs["sample_seed"] = self.hparams.seed
            self.train_dataset = WilberforceDataset(**train_kwargs)

            val_kwargs = dict(self.dataset_kwargs)
            val_kwargs["size"] = self.hparams.val_size
            val_kwargs["sample_seed"] = self.hparams.seed + 1000
            val_kwargs["return_z_full"] = True
            self.val_dataset = WilberforceDataset(**val_kwargs)

        if stage in ("test", None):
            test_kwargs = dict(self.dataset_kwargs)
            test_kwargs["size"] = self.hparams.test_size
            test_kwargs["sample_seed"] = self.hparams.seed + 2000
            test_kwargs["return_z_full"] = True
            self.test_dataset = WilberforceDataset(**test_kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class WilberforceImage(Wilberforce):
    def __init__(self, *args, **kwargs):
        kwargs["mixing_type"] = "image"
        super().__init__(*args, **kwargs)
