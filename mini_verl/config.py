"""Configuration dataclasses whose dotted names mirror verl's Hydra config.

verl: ``verl/trainer/config/ppo_trainer.yaml`` (+ the generated ``_generated_ppo_trainer.yaml``),
``verl/workers/config/{actor,critic,rollout,optimizer}.py``, ``verl/trainer/config/algorithm.py``.
Only the keys that matter for the algorithm are kept, but each keeps verl's name and meaning, e.g.
``actor_rollout_ref.actor.ppo_mini_batch_size`` is counted in PROMPTS and multiplied by
``rollout.n`` inside the trainer, exactly as in verl.

    cfg = PPOConfig()
    cfg.apply_overrides({"algorithm.adv_estimator": "grpo", "actor_rollout_ref.rollout.n": 8})
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


def set_reproducible(num_threads: int = 1):
    """Pin torch's intra-op thread count. Float reductions are split differently for different
    thread counts, so every number in RESULTS.md was produced with num_threads=1."""
    torch.set_num_threads(num_threads)


class _Cfg:
    def get(self, key, default=None):  # verl code often does config.get("x", default)
        return getattr(self, key, default)


@dataclass
class ModelConfig(_Cfg):
    vocab_size: int = 18          # <pad> <eos> 0-9 + = <c> </c> <r> </r>  (see data.py)
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 256
    max_position: int = 32
    seed: int = 0                 # initial weights are a deterministic function of this seed


@dataclass
class DataConfig(_Cfg):
    train_batch_size: int = 64    # prompts per RL step (before rollout.n)
    max_prompt_length: int = 6    # "99+99=" is 6 tokens; shorter prompts are LEFT-padded
    max_response_length: int = 4  # "198" + <eos>; responses are RIGHT-padded
    operand_max: int = 99         # operands a, b are drawn from 0..operand_max
    val_size: int = 200           # held-out (a, b) pairs, never used for training
    data_source: str = "arithmetic/addition"
    seed: int = 0


@dataclass
class OptimConfig(_Cfg):
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    clip_grad: float = 1.0        # verl: optim.clip_grad / actor.grad_clip


@dataclass
class ActorConfig(_Cfg):
    ppo_mini_batch_size: int = 32          # in prompts; x rollout.n samples per optimizer step
    ppo_micro_batch_size_per_gpu: int = 32  # samples per forward/backward (gradient accumulation)
    ppo_epochs: int = 1
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    clip_ratio_c: float = 3.0              # dual-clip lower bound for A < 0
    loss_agg_mode: str = "token-mean"
    loss_scale_factor: Optional[int] = None
    entropy_coeff: float = 0.0
    use_kl_loss: bool = False              # GRPO style: KL to the ref policy inside the loss
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    shuffle: bool = False
    data_loader_seed: int = 42
    optim: OptimConfig = field(default_factory=lambda: OptimConfig(lr=1e-4))


@dataclass
class RefConfig(_Cfg):
    log_prob_micro_batch_size_per_gpu: int = 64


@dataclass
class MultiTurnConfig(_Cfg):
    enable: bool = False
    max_assistant_turns: int = 2
    max_user_turns: int = 1


@dataclass
class RolloutConfig(_Cfg):
    n: int = 8
    temperature: float = 1.0
    top_k: int = -1
    do_sample: bool = True
    response_length: int = 4               # = data.max_response_length (verl interpolates it)
    seed: int = 42
    log_prob_micro_batch_size_per_gpu: int = 64
    multi_turn: MultiTurnConfig = field(default_factory=MultiTurnConfig)


@dataclass
class ActorRolloutRefConfig(_Cfg):
    model: ModelConfig = field(default_factory=ModelConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    ref: RefConfig = field(default_factory=RefConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)


@dataclass
class CriticConfig(_Cfg):
    ppo_mini_batch_size: int = 32
    ppo_micro_batch_size_per_gpu: int = 32
    forward_micro_batch_size_per_gpu: int = 64
    ppo_epochs: int = 1
    cliprange_value: float = 0.5
    loss_agg_mode: str = "token-mean"
    shuffle: bool = False
    data_loader_seed: int = 42
    optim: OptimConfig = field(default_factory=OptimConfig)


@dataclass
class KLControlConfig(_Cfg):
    type: str = "fixed"
    kl_coef: float = 0.001
    horizon: int = 10000
    target_kl: float = 0.1


@dataclass
class AlgoConfig(_Cfg):
    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "gae"
    norm_adv_by_std_in_grpo: bool = True
    use_kl_in_reward: bool = False
    kl_penalty: str = "kl"
    kl_ctrl: KLControlConfig = field(default_factory=KLControlConfig)


@dataclass
class TrainerConfig(_Cfg):
    total_training_steps: int = 200
    critic_warmup: int = 0
    nnodes: int = 1
    n_gpus_per_node: int = 2      # simulated data-parallel ranks (in-process workers)
    balance_batch: bool = True
    test_freq: int = 50
    val_before_train: bool = True
    seed: int = 0


@dataclass
class SFTConfig(_Cfg):
    """The warm-start recipe (not part of verl's PPO config): 550 steps give ~30% accuracy."""
    steps: int = 550
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 0


@dataclass
class PPOConfig(_Cfg):
    data: DataConfig = field(default_factory=DataConfig)
    actor_rollout_ref: ActorRolloutRefConfig = field(default_factory=ActorRolloutRefConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    algorithm: AlgoConfig = field(default_factory=AlgoConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def apply_overrides(self, overrides: dict) -> "PPOConfig":
        """Hydra-style dotted overrides: {"algorithm.adv_estimator": "grpo"}."""
        for dotted, value in overrides.items():
            *path, leaf = dotted.split(".")
            node = self
            for p in path:
                node = getattr(node, p)
            assert hasattr(node, leaf), f"unknown config key {dotted}"
            setattr(node, leaf, value)
        self.actor_rollout_ref.rollout.response_length = self.data.max_response_length
        return self
