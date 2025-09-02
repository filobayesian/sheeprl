from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor, nn
from torch.distributions import (
    Distribution,
    Independent,
    Normal,
    OneHotCategoricalStraightThrough,
    TanhTransform,
    TransformedDistribution,
)
from torch.distributions.utils import probs_to_logits

from sheeprl.algos.dreamer_v2.agent import WorldModel
from sheeprl.algos.dreamer_v2.utils import compute_stochastic_state
from sheeprl.algos.dreamer_v3.utils import init_weights, uniform_init_weights
from sheeprl.algos.dreamer_v3.agent import Actor as PlayerDV3Actor
from sheeprl.algos.dreamer_v3.agent import MinedojoActor as PlayerDV3MinedojoActor
from sheeprl.algos.dreamer_v3.agent import PlayerDV3, WorldModel
from sheeprl.algos.dreamer_v3.agent import build_agent as _build_dv3_agent
from sheeprl.models.models import (
    CNN,
    MLP,
    DeCNN,
    LayerNorm,
    LayerNormChannelLast,
    LayerNormGRUCell,
    MultiDecoder,
    MultiEncoder,
)
from sheeprl.utils.fabric import get_single_device_fabric
from sheeprl.utils.model import ModuleType, cnn_forward
from sheeprl.utils.utils import symlog

from sheeprl.models.jepa import JEPAHead



def build_agent(
    fabric: Fabric,
    actions_dim: Sequence[int],
    is_continuous: bool,
    cfg: Dict[str, Any],
    obs_space: gym.spaces.Dict,
    world_model_state: Optional[Dict[str, Tensor]] = None,
    actor_state: Optional[Dict[str, Tensor]] = None,
    critic_state: Optional[Dict[str, Tensor]] = None,
    target_critic_state: Optional[Dict[str, Tensor]] = None,
) -> Tuple[WorldModel, _FabricModule, _FabricModule, _FabricModule, PlayerDV3]:
    """Build agent as Dreamer-V3 and attach a JEPA head to the world model.

    Returns the same tuple as Dreamer-V3 build_agent.
    """
    world_model, actor, critic, target_critic, player = _build_dv3_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        obs_space,
        world_model_state,
        actor_state,
        critic_state,
        target_critic_state,
    )

    # Attach JEPA head: use the shared encoder and its output dimension
    # world_model.encoder is a Fabric-wrapped module; access output_dim on the underlying module
    enc_module = getattr(world_model.encoder, "module", world_model.encoder)
    enc_out_dim = getattr(enc_module, "output_dim")
    jepa_head = JEPAHead(
        online_encoder=world_model.encoder,
        enc_out_dim=enc_out_dim,
        proj_dim=cfg.algo.jepa_proj_dim,
        pred_hidden=cfg.algo.jepa_hidden,
        ema_m=cfg.algo.jepa_ema,
    ).to(fabric.device)
    # Ensure target branch stays EMA-only
    jepa_head.target_encoder.eval()
    jepa_head.target_projector.eval()
    for p in jepa_head.target_encoder.parameters():
        p.requires_grad = False
    for p in jepa_head.target_projector.parameters():
        p.requires_grad = False

    # Register on the world model so its trainable params are optimized
    world_model.jepa = jepa_head

    return world_model, actor, critic, target_critic, player