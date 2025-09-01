from __future__ import annotations

from copy import deepcopy
from typing import Dict, Tuple

import torch
from torch import Tensor, nn


def _erase_rectangles(x: Tensor, erase_frac: float) -> Tensor:
    if x.dim() != 5:
        raise ValueError(f"Expected image tensor (T,B,C,H,W), got shape: {x.shape}")
    T, B, C, H, W = x.shape
    h = int(H * (1 - erase_frac))
    w = int(W * (1 - erase_frac))
    h = max(1, min(H, h))
    w = max(1, min(W, w))
    top = (H - h) // 2
    left = (W - w) // 2
    mask = torch.zeros(1, B, 1, H, W, dtype=x.dtype, device=x.device)
    mask[..., top : top + h, left : left + w] = 1
    return x * mask


@torch.no_grad()
def make_two_views(
    obs: Dict[str, Tensor], erase_frac: float = 0.6, vec_dropout: float = 0.2
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    obs_q: Dict[str, Tensor] = {}
    obs_k: Dict[str, Tensor] = {}
    for k, v in obs.items():
        if v.dim() == 5:  # (T,B,C,H,W)
            obs_q[k] = _erase_rectangles(v, erase_frac)
            obs_k[k] = _erase_rectangles(v, erase_frac)
        else:  # assume (T,B,D,...)
            noise_q = torch.randn_like(v) * vec_dropout
            noise_k = torch.randn_like(v) * vec_dropout
            obs_q[k] = v + noise_q
            obs_k[k] = v + noise_k
    return obs_q, obs_k


class JEPAProjector(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 1024, hidden: int = 1024) -> None:
        super().__init__()
        self.pool_time = True
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=True),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim, bias=True),
        )

    def forward(self, z: Tensor) -> Tensor:
        # z is (T,B,D) or (B,D). Pool over time (dim 0) if 3D
        if z.dim() == 3:
            z = z.mean(dim=0)
        return self.net(z)


class JEPAPredictor(nn.Module):
    def __init__(self, proj_dim: int = 1024, hidden: int = 1024) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proj_dim, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim, bias=True),
        )

    def forward(self, p: Tensor) -> Tensor:
        return self.net(p)


class JEPAHead(nn.Module):
    def __init__(
        self,
        online_encoder: nn.Module,
        enc_out_dim: int,
        proj_dim: int = 1024,
        pred_hidden: int = 1024,
        ema_m: float = 0.996,
    ) -> None:
        super().__init__()
        self.encoder = online_encoder  # shared encoder instance
        self.projector = JEPAProjector(enc_out_dim, proj_dim=proj_dim, hidden=pred_hidden)
        self.predictor = JEPAPredictor(proj_dim=proj_dim, hidden=pred_hidden)

        # target branch (deep-copied, frozen, EMA)
        self.target_encoder = deepcopy(online_encoder).eval()
        self.target_projector = JEPAProjector(enc_out_dim, proj_dim=proj_dim, hidden=pred_hidden).eval()
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

        self.register_buffer("ema_m", torch.tensor(ema_m, dtype=torch.float32))

    @staticmethod
    def _l2_normalize(x: Tensor, eps: float = 1e-6) -> Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def forward(self, obs_q: Dict[str, Tensor], obs_k: Dict[str, Tensor]) -> Tensor:
        zq = self.encoder(obs_q)
        zk = self.target_encoder(obs_k)
        pq = self.predictor(self.projector(zq))
        zk = self.target_projector(zk).detach()
        pq = self._l2_normalize(pq)
        zk = self._l2_normalize(zk)
        loss = 2 - 2 * (pq * zk).sum(dim=-1).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("JEPA loss is not finite")
        return loss

    @torch.no_grad()
    def update_momentum_from(self, online: "JEPAHead") -> None:
        m = float(self.ema_m.item())
        # encoder
        for p_t, p_o in zip(self.target_encoder.parameters(), online.encoder.parameters()):
            p_t.data.mul_(m).add_(p_o.data, alpha=1 - m)
        # projector
        for p_t, p_o in zip(self.target_projector.parameters(), online.projector.parameters()):
            p_t.data.mul_(m).add_(p_o.data, alpha=1 - m)


