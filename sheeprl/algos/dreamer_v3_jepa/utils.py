from __future__ import annotations

AGGREGATOR_KEYS = {
    "Rewards/rew_avg",
    "Game/ep_len_avg",
    "Loss/world_model_total",
    "Loss/value_loss",
    "Loss/policy_loss",
    "Loss/jepa",
    "Loss/reward_loss",
    "Loss/state_loss",
    "Loss/continue_loss",
    "State/kl",
    "State/post_entropy",
    "State/prior_entropy",
    "Grads/world_model",
    "Grads/actor",
    "Grads/critic",
}

MODELS_TO_REGISTER = {"world_model", "actor", "critic", "target_critic", "moments"}


