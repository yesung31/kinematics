import math
from typing import Literal

import numpy as np
import torch

from .base import (
    BaseKinematicsDataModule,
    BaseKinematicsDataset,
    RealNVP,
)
from .render import DoublePendulumRenderer

DoublePendulumRealNVP = RealNVP


class DoublePendulumSimulation:
    def __init__(
        self,
        l1: float = 1.0,
        l2: float = 1.0,
        m1: float = 1.0,
        m2: float = 1.0,
        g: float = 9.81,
        damping: float = 0.0,
        num_energy_levels: int = 8,
        energy_levels: list[float] | None = None,
        initial_states: list[tuple[float, float, float, float] | list[float]] | None = None,
        trajectory_length: int = 65536,
        dt: float = 0.05,
        substeps: int = 2,
        include_velocities: bool = False,
    ):
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.g = g
        self.damping = damping
        self.trajectory_length = trajectory_length
        self.dt = dt
        self.substeps = substeps
        self.include_velocities = include_velocities

        if initial_states is not None:
            self.initial_states = [list(s) for s in initial_states]
            self.energy_levels = [self.compute_energy(s) for s in self.initial_states]
            self.num_energy_levels = len(self.initial_states)
        else:
            self.num_energy_levels = num_energy_levels
            if energy_levels is None:
                self.energy_levels = [0.5 * (k + 1) for k in range(self.num_energy_levels)]
            else:
                self.energy_levels = list(energy_levels)
                self.num_energy_levels = len(self.energy_levels)
            self.initial_states = [self._initial_state_for_energy(e) for e in self.energy_levels]

        self.trajectories = self._compute_trajectories()

    def compute_energy(
        self, state: tuple[float, float, float, float] | list[float] | np.ndarray
    ) -> float:
        th1, th2, w1, w2 = float(state[0]), float(state[1]), float(state[2]), float(state[3])
        dth = th1 - th2
        t = (
            0.5 * (self.m1 + self.m2) * (self.l1**2) * (w1**2)
            + 0.5 * self.m2 * (self.l2**2) * (w2**2)
            + self.m2 * self.l1 * self.l2 * w1 * w2 * math.cos(dth)
        )
        v = (self.m1 + self.m2) * self.g * self.l1 * (
            1.0 - math.cos(th1)
        ) + self.m2 * self.g * self.l2 * (1.0 - math.cos(th2))
        return t + v

    def _initial_state_for_energy(self, energy: float) -> list[float]:
        low, high = 0.0, math.pi
        c1 = (self.m1 + self.m2) * self.g * self.l1
        c2 = self.m2 * self.g * self.l2
        max_e = c1 * 2.0 + c2 * (1.0 - math.cos(0.5 * math.pi))
        if energy >= max_e:
            return [high, 0.5 * high, 0.0, 0.0]

        for _ in range(60):
            mid = (low + high) * 0.5
            val = c1 * (1.0 - math.cos(mid)) + c2 * (1.0 - math.cos(0.5 * mid))
            if val < energy:
                low = mid
            else:
                high = mid
        th0 = (low + high) * 0.5
        return [th0, 0.5 * th0, 0.0, 0.0]

    def _compute_trajectories(self) -> torch.Tensor:
        states = np.array(self.initial_states, dtype=np.float64)
        n_trajs = len(states)
        dt_sub = self.dt / self.substeps

        trajectories = np.empty((n_trajs, self.trajectory_length, 4), dtype=np.float64)
        trajectories[:, 0] = states
        curr = states.copy()

        m1, m2 = self.m1, self.m2
        l1, l2 = self.l1, self.l2
        g = self.g
        damping = self.damping

        def derivs(st: np.ndarray) -> np.ndarray:
            th1, th2, w1, w2 = st[:, 0], st[:, 1], st[:, 2], st[:, 3]
            dth = th1 - th2
            sin_dth = np.sin(dth)
            cos_dth = np.cos(dth)
            den = l1 * (m1 + m2 * sin_dth * sin_dth)
            f1 = -m2 * l2 * w2 * w2 * sin_dth - (m1 + m2) * g * np.sin(th1) - damping * w1
            f2 = l1 * w1 * w1 * sin_dth - g * np.sin(th2) - damping * w2
            dw1 = (f1 - m2 * cos_dth * f2) / den
            dw2 = ((m1 + m2) * f2 - cos_dth * f1) / (l2 * (m1 + m2 * sin_dth * sin_dth))
            return np.column_stack([w1, w2, dw1, dw2])

        for i in range(1, self.trajectory_length):
            for _ in range(self.substeps):
                k1 = derivs(curr)
                k2 = derivs(curr + 0.5 * dt_sub * k1)
                k3 = derivs(curr + 0.5 * dt_sub * k2)
                k4 = derivs(curr + dt_sub * k3)
                curr += (dt_sub / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            trajectories[:, i] = curr

        if not self.include_velocities:
            res = trajectories[:, :, :2]
        else:
            res = trajectories

        return torch.from_numpy(res).float()


class DoublePendulumDataset(BaseKinematicsDataset):
    def __init__(
        self,
        size: int = 65536,
        trajectory_length: int = 65536,
        num_energy_levels: int = 8,
        energy_levels: list[float] | None = None,
        initial_states: list[tuple[float, float, float, float] | list[float]] | None = None,
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
        mixing_seed: int | None = None,
        return_z_full: bool = False,
        squeeze_single: bool = True,
        target_frame: str = "last",
        dt: float = 0.05,
        l1: float = 1.0,
        l2: float = 1.0,
        m1: float = 1.0,
        m2: float = 1.0,
        g: float = 9.81,
        damping: float = 0.0,
        include_velocities: bool = False,
        simulation: DoublePendulumSimulation | None = None,
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
            mixing_seed=mixing_seed,
            return_z_full=return_z_full,
            squeeze_single=squeeze_single,
            target_frame=target_frame,
            **kwargs,
        )
        self.include_velocities = include_velocities
        self.z_dim = 4 if include_velocities else 2

        if simulation is not None:
            self.simulation = simulation
        else:
            self.simulation = DoublePendulumSimulation(
                l1=l1,
                l2=l2,
                m1=m1,
                m2=m2,
                g=g,
                damping=damping,
                num_energy_levels=num_energy_levels,
                energy_levels=energy_levels,
                initial_states=initial_states,
                trajectory_length=trajectory_length,
                dt=dt,
                include_velocities=include_velocities,
            )
        self.trajectories = self.simulation.trajectories
        self.num_energy_levels = self.simulation.num_energy_levels

        self.renderer = DoublePendulumRenderer(
            img_size=img_size, l1=l1, l2=l2, seed=self.mixing_seed
        )

        self.x_dim = kwargs.get("x_dim", self.z_dim)
        if self.mixing_type == "realnvp":
            out_features = self.x_dim if isinstance(self.x_dim, int) else self.z_dim
            self.flow = RealNVP(
                in_features=self.z_dim,
                out_features=out_features,
                num_layers=flow_layers,
                hidden_features=flow_hidden,
                seed=self.mixing_seed,
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


class DoublePendulumRealNVPDataset(DoublePendulumDataset):
    def __init__(self, *args, **kwargs):
        kwargs["mixing_type"] = "realnvp"
        super().__init__(*args, **kwargs)


class DoublePendulumImageDataset(DoublePendulumDataset):
    def __init__(self, *args, **kwargs):
        kwargs["mixing_type"] = "image"
        super().__init__(*args, **kwargs)


class DoublePendulum(BaseKinematicsDataModule):
    dataset_cls = DoublePendulumDataset

    def __init__(
        self,
        size: int = 65536,
        val_size: int = 8192,
        test_size: int = 16384,
        trajectory_length: int = 65536,
        num_energy_levels: int = 8,
        seq_len: int = 16,
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
        mixing_seed: int | None = None,
        dt: float = 0.05,
        squeeze_single: bool = True,
        target_frame: str = "last",
        l1: float = 1.0,
        l2: float = 1.0,
        m1: float = 1.0,
        m2: float = 1.0,
        g: float = 9.81,
        damping: float = 0.0,
        include_velocities: bool = False,
        as_dict: bool = False,
        **kwargs,
    ):
        eff_mixing_seed = mixing_seed if mixing_seed is not None else kwargs.get("model_seed", seed)
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            as_dict=as_dict,
            mixing_seed=eff_mixing_seed,
            seed=seed,
            size=size,
            val_size=val_size,
            test_size=test_size,
            **kwargs,
        )
        self.save_hyperparameters()
        base_z_dim = 4 if include_velocities else 2
        self.z_dim = kwargs.get("z_dim", base_z_dim + (1 if same_energy else 0))
        multiplier = seq_len if (seq_len > 1 or not squeeze_single) and not as_dict else 1
        if mixing_type == "image":
            base_x_dim = kwargs.get("x_dim", [3, img_size, img_size])
            self.x_dim = [base_x_dim[0] * multiplier, base_x_dim[1], base_x_dim[2]]
        else:
            base_x_dim = kwargs.get("x_dim", base_z_dim)
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
            "as_dict": as_dict,
            "dt": dt,
            "l1": l1,
            "l2": l2,
            "m1": m1,
            "m2": m2,
            "g": g,
            "damping": damping,
            "include_velocities": include_velocities,
            "model_seed": eff_mixing_seed,
            "mixing_seed": eff_mixing_seed,
            **kwargs,
        }

    @property
    def factor_types(self) -> list[str]:
        base_factors = (
            ["continuous", "continuous", "continuous", "continuous"]
            if self.dataset_kwargs.get("include_velocities", False)
            else ["continuous", "continuous"]
        )
        if self.dataset_kwargs.get("same_energy", False):
            return base_factors + ["continuous"]
        return base_factors


class DoublePendulumImage(DoublePendulum):
    def __init__(self, *args, **kwargs):
        kwargs["mixing_type"] = "image"
        super().__init__(*args, **kwargs)
