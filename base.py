from typing import Literal

import torch
import torch.nn as nn
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


class RealNVP(nn.Module):
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


class BaseKinematicsDataset(Dataset):
    def __init__(
        self,
        size: int = 65536,
        trajectory_length: int = 65536,
        num_energy_levels: int = 8,
        seq_len: int = 32,
        same_energy: bool = True,
        min_time_dist: int = 0,
        max_time_dist: int | None = None,
        mixing_type: Literal["realnvp", "image", "none"] = "realnvp",
        img_size: int = 64,
        sample_seed: int = 42,
        model_seed: int = 42,
        return_z_full: bool = False,
        squeeze_single: bool = True,
        target_frame: str = "last",
        **kwargs,
    ):
        super().__init__()
        if trajectory_length < seq_len:
            raise ValueError(
                f"trajectory_length ({trajectory_length}) must be >= seq_len ({seq_len})"
            )
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
        self.simulation = None
        self.trajectories = None
        self.renderer = None
        self.flow = None

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


class BaseKinematicsDataModule(LightningDataModule):
    dataset_cls: type[BaseKinematicsDataset]

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = {}

    def setup(self, stage=None):
        if stage in ("fit", "validate", None):
            train_kwargs = dict(self.dataset_kwargs)
            train_kwargs["size"] = self.hparams.size
            train_kwargs["sample_seed"] = self.hparams.seed
            self.train_dataset = self.dataset_cls(**train_kwargs)

            val_kwargs = dict(self.dataset_kwargs)
            val_kwargs["size"] = self.hparams.val_size
            val_kwargs["sample_seed"] = self.hparams.seed + 1000
            val_kwargs["return_z_full"] = True
            if hasattr(self.train_dataset, "simulation"):
                val_kwargs["simulation"] = self.train_dataset.simulation
            self.val_dataset = self.dataset_cls(**val_kwargs)

        if stage in ("test", None):
            test_kwargs = dict(self.dataset_kwargs)
            test_kwargs["size"] = self.hparams.test_size
            test_kwargs["sample_seed"] = self.hparams.seed + 2000
            test_kwargs["return_z_full"] = True
            if hasattr(self, "train_dataset") and hasattr(self.train_dataset, "simulation"):
                test_kwargs["simulation"] = self.train_dataset.simulation
            self.test_dataset = self.dataset_cls(**test_kwargs)

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
