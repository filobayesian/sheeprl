import gymnasium as gym
import torch
from lightning import Fabric

from sheeprl.algos.dreamer_v3_jepa.dreamer_v3_jepa import build_agent


def test_build_agent_without_decoder():
    fabric = Fabric(devices=1, accelerator="cpu")
    fabric.launch(lambda f: None)
    obs_space = gym.spaces.Dict({
        "rgb": gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=int),
    })
    class Cfg: pass
    cfg = Cfg()
    cfg.env = Cfg(); cfg.env.screen_size = 64; cfg.env.num_envs = 1
    cfg.algo = Cfg(); cfg.algo.world_model = Cfg(); cfg.algo.world_model.recurrent_model = Cfg()
    cfg.algo.world_model.recurrent_model.recurrent_state_size = 16
    cfg.algo.world_model.stochastic_size = 4; cfg.algo.world_model.discrete_size = 4
    cfg.algo.cnn_keys = Cfg(); cfg.algo.cnn_keys.encoder = ["rgb"]; cfg.algo.cnn_keys.decoder = []
    cfg.algo.mlp_keys = Cfg(); cfg.algo.mlp_keys.encoder = []; cfg.algo.mlp_keys.decoder = []
    cfg.algo.actor = Cfg(); cfg.algo.actor.cls = "sheeprl.algos.dreamer_v3.agent.Actor"
    cfg.algo.critic = Cfg(); cfg.algo.critic.layer_norm = Cfg(); cfg.algo.critic.layer_norm.cls = "torch.nn.LayerNorm"; cfg.algo.critic.layer_norm.kw = {}
    cfg.algo.world_model.encoder = Cfg(); cfg.algo.world_model.encoder.cnn_channels_multiplier = 8
    cfg.algo.world_model.encoder.cnn_layer_norm = Cfg(); cfg.algo.world_model.encoder.cnn_layer_norm.cls = "sheeprl.models.models.LayerNormChannelLast"; cfg.algo.world_model.encoder.cnn_layer_norm.kw = {"eps":1e-3}
    cfg.algo.world_model.encoder.dense_act = "torch.nn.SiLU"
    cfg.algo.world_model.representation_model = Cfg(); cfg.algo.world_model.representation_model.hidden_size = 8; cfg.algo.world_model.representation_model.dense_act = "torch.nn.SiLU"; cfg.algo.world_model.representation_model.layer_norm = Cfg(); cfg.algo.world_model.representation_model.layer_norm.cls = "sheeprl.models.models.LayerNorm"; cfg.algo.world_model.representation_model.layer_norm.kw = {"eps":1e-3}
    cfg.algo.world_model.transition_model = Cfg(); cfg.algo.world_model.transition_model.hidden_size = 8; cfg.algo.world_model.transition_model.dense_act = "torch.nn.SiLU"; cfg.algo.world_model.transition_model.layer_norm = Cfg(); cfg.algo.world_model.transition_model.layer_norm.cls = "sheeprl.models.models.LayerNorm"; cfg.algo.world_model.transition_model.layer_norm.kw = {"eps":1e-3}
    cfg.algo.world_model.observation_model = Cfg(); cfg.algo.world_model.observation_model.cnn_channels_multiplier = 8; cfg.algo.world_model.observation_model.cnn_act = "torch.nn.SiLU"; cfg.algo.world_model.observation_model.cnn_layer_norm = cfg.algo.world_model.encoder.cnn_layer_norm; cfg.algo.world_model.observation_model.mlp_layer_norm = Cfg(); cfg.algo.world_model.observation_model.mlp_layer_norm.cls = "sheeprl.models.models.LayerNorm"; cfg.algo.world_model.observation_model.mlp_layer_norm.kw = {"eps":1e-3}; cfg.algo.world_model.observation_model.mlp_layers = 2; cfg.algo.world_model.observation_model.dense_units = 16
    cfg.algo.actor.dense_act = "torch.nn.SiLU"; cfg.algo.actor.mlp_layers = 2; cfg.algo.actor.layer_norm = Cfg(); cfg.algo.actor.layer_norm.cls = "sheeprl.models.models.LayerNorm"; cfg.algo.actor.layer_norm.kw = {"eps":1e-3}; cfg.algo.dense_units = 16; cfg.distribution = Cfg(); cfg.distribution.validate_args = False
    # JEPA params
    cfg.algo.jepa_proj_dim = 16; cfg.algo.jepa_hidden = 16; cfg.algo.jepa_ema = 0.996

    world_model, actor, critic, target_critic, player = build_agent(
        fabric,
        actions_dim=(1,),
        is_continuous=True,
        cfg=cfg,
        obs_space=obs_space,
    )
    assert getattr(world_model, "jepa", None) is not None


