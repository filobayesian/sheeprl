import torch
from torch import nn

from sheeprl.models.jepa import JEPAHead, make_two_views


class DummyEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.output_dim = out_dim
        self.net = nn.Linear(out_dim, out_dim)

    def forward(self, obs):
        # obs is dict with one vector key of shape (T,B,D)
        x = next(iter(obs.values()))
        if x.dim() == 3:
            x = x.mean(0)
        return self.net(x)


def test_make_two_views_shapes():
    T, B = 4, 2
    obs = {
        "rgb": torch.randn(T, B, 3, 64, 64),
        "vec": torch.randn(T, B, 32),
    }
    q, k = make_two_views(obs)
    for kx in obs.keys():
        assert q[kx].shape == obs[kx].shape
        assert k[kx].shape == obs[kx].shape
        assert q[kx].dtype == obs[kx].dtype
        assert q[kx].device == obs[kx].device


def test_jepa_forward_and_ema():
    T, B, D = 3, 2, 16
    obs = {"vec": torch.randn(T, B, D)}
    q, k = make_two_views(obs)
    enc = DummyEncoder(D)
    head = JEPAHead(enc, enc_out_dim=D, proj_dim=8, pred_hidden=8)
    loss = head(q, k)
    assert torch.isfinite(loss)
    (loss).backward()
    # EMA should run without error
    head.update_momentum_from(head)


