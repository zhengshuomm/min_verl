"""Workers: ActorRolloutRefWorker (actor + rollout + reference policy) and CriticWorker.

verl v0.9.1 implements these in ``verl/workers/engine_workers.py`` (``TrainingWorker``,
``ActorRolloutRefWorker``) on top of a model engine
(``verl/workers/engine/fsdp/transformer_impl.py``: ``forward_backward_batch``, ``optimizer_step``)
and the loss functions of ``verl/workers/utils/losses.py`` (``ppo_loss``, ``value_loss``). Older
verl releases (and most blog posts) kept the same logic in one data-parallel actor class
(``DataParallelPPOActor.compute_log_prob`` / ``update_policy``) and one critic class
(``DataParallelPPOCritic``); mini_verl follows that compact layout.

Data parallelism is simulated: every rank holds a full replica (same seed -> same weights) and
gets 1/world_size of each batch through ``Dispatch.DP_COMPUTE_PROTO``. Before each optimizer step
gradients are averaged across ranks with ``all_reduce`` (what FSDP / DDP do with NCCL), and the
loss is normalised by GLOBAL token / sequence counts, so the update equals single-process training.
"""

import copy
from collections import defaultdict

import torch

from .agent_loop import ToolAgentLoop
from .config import ActorRolloutRefConfig, CriticConfig, ModelConfig
from .core_algos import agg_loss, compute_policy_loss, compute_value_loss, kl_penalty, masked_mean
from .data import Tokenizer
from .models import TinyCritic, TinyPolicy, entropy_from_logits, logprobs_from_logits
from .protocol import DataProto
from .rollout import Rollout
from .single_controller import Dispatch, Execute, Worker, register


def forward_micro_batch(module, batch, temperature: float, calculate_entropy: bool = False):
    """Log-probs (and entropy) of the response tokens under ``module``.

    The logit at position t predicts token t+1, so the logits for the R response tokens are
    ``logits[:, -R-1:-1]``; they are divided by the sampling temperature, as the rollout did.
    """
    responses = batch["responses"]
    R = responses.size(1)
    logits = module(batch["input_ids"], batch["attention_mask"], batch["position_ids"])
    logits = logits[:, -R - 1:-1, :] / temperature
    log_probs = logprobs_from_logits(logits, responses)
    return log_probs, (entropy_from_logits(logits) if calculate_entropy else None)


def _optimizer(module, optim_cfg):
    return torch.optim.AdamW(module.parameters(), lr=optim_cfg.lr, betas=tuple(optim_cfg.betas),
                             weight_decay=optim_cfg.weight_decay)


def ppo_train_loop(worker: Worker, module, optimizer, data: DataProto, global_mini_batch_size: int, epochs: int,
                   micro_batch_size: int, clip_grad: float, loss_fn, metric_prefix: str, shuffle=False, seed=42):
    """epochs x mini-batches x micro-batches with gradient accumulation and DP gradient all-reduce.

    ``loss_fn(micro_batch, global_batch_info) -> (loss, {name: (value, "sum" | "mean")})``.
    "sum" metrics are globally normalised partial losses (summed over micro-batches); "mean"
    metrics are per-micro-batch statistics (averaged).
    """
    local_mbs = global_mini_batch_size // worker.world_size
    assert local_mbs > 0 and len(data) % local_mbs == 0, f"{len(data)=} not divisible by mini-batch {local_mbs}"
    module.train()
    params = list(module.parameters())
    metrics = defaultdict(list)
    for epoch in range(epochs):
        epoch_data = data
        if shuffle:
            g = torch.Generator().manual_seed(seed + worker.rank + 1000 * epoch)
            epoch_data = data.select_idxs(torch.randperm(len(data), generator=g))
        for mini in epoch_data.split(local_mbs):
            num_tokens = mini.batch["response_mask"].sum().float()
            worker.all_reduce([num_tokens], op="sum")  # global #tokens of this mini-batch
            gbi = {"dp_size": worker.world_size, "batch_num_tokens": num_tokens,
                   "global_batch_size": global_mini_batch_size}
            optimizer.zero_grad()
            sums, means = defaultdict(float), defaultdict(list)
            for micro in mini.split(micro_batch_size):
                loss, m = loss_fn(micro.batch, gbi)
                loss.backward()  # gradients accumulate across micro-batches
                for k, (v, how) in m.items():
                    if how == "sum":
                        sums[k] += float(v)
                    else:
                        means[k].append(float(v))
            grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
            worker.all_reduce(grads, op="mean")  # DP: every rank now holds the same gradient
            for p, g in zip(params, grads):
                p.grad = g
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=clip_grad)
            optimizer.step()
            for k, v in sums.items():
                metrics[k].append(v)
            for k, v in means.items():
                metrics[k].append(sum(v) / len(v))
            metrics[f"{metric_prefix}/grad_norm"].append(float(grad_norm))
            metrics[f"{metric_prefix}/lr"].append(optimizer.param_groups[0]["lr"])
    return DataProto(meta_info={"metrics": dict(metrics)})


class ActorRolloutRefWorker(Worker):
    """One worker class that can play actor, rollout and reference roles (hybrid engine).

    role: "actor", "rollout", "ref", "actor_rollout" or "actor_rollout_ref".
    """

    def __init__(self, config: ActorRolloutRefConfig, role: str):
        super().__init__()
        self.config, self.role = config, role
        self._is_actor = role in ("actor", "actor_rollout", "actor_rollout_ref")
        self._is_rollout = role in ("rollout", "actor_rollout", "actor_rollout_ref")
        self._is_ref = role in ("ref", "actor_rollout_ref")
        self.tokenizer = Tokenizer()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self, state_dict=None):
        base = TinyPolicy(self.config.model)
        if state_dict is not None:
            base.load_state_dict(state_dict)
        if self._is_actor:
            self.actor_module = base
            self.actor_optimizer = _optimizer(base, self.config.actor.optim)
        if self._is_rollout:  # the inference engine gets its OWN copy of the weights
            self.rollout = Rollout(copy.deepcopy(base), self.config.rollout, self.tokenizer,
                                   seed=self.config.rollout.seed + self.rank)
            self.rollout.weight_version = 0
            self.agent_loop = ToolAgentLoop(self.rollout, self.tokenizer, self.config.rollout)
        if self._is_ref:  # frozen copy of the initial policy
            self.ref_module = copy.deepcopy(base).eval().requires_grad_(False)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        engine = self.agent_loop if self.config.rollout.multi_turn.enable else self.rollout
        return engine.generate_sequences(prompts)

    @torch.no_grad()
    def _compute_log_prob(self, module, data: DataProto, calculate_entropy: bool, micro_batch_size: int):
        temperature = data.meta_info.get("temperature", self.config.rollout.temperature)
        module.eval()
        outs = [forward_micro_batch(module, m.batch, temperature, calculate_entropy) for m in data.split(micro_batch_size)]
        log_probs = torch.cat([o[0] for o in outs])
        return log_probs, (torch.cat([o[1] for o in outs]) if calculate_entropy else None)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto) -> DataProto:
        """pi_old: log-probs of the rollout tokens under the CURRENT actor (before the update)."""
        log_probs, entropys = self._compute_log_prob(self.actor_module, data, True,
                                                     self.config.rollout.log_prob_micro_batch_size_per_gpu)
        return DataProto.from_dict({"old_log_probs": log_probs, "entropys": entropys})

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto) -> DataProto:
        log_probs, _ = self._compute_log_prob(self.ref_module, data, False,
                                              self.config.ref.log_prob_micro_batch_size_per_gpu)
        return DataProto.from_dict({"ref_log_prob": log_probs})

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto) -> DataProto:
        cfg = self.config.actor
        temperature = data.meta_info["temperature"]
        calculate_entropy = cfg.entropy_coeff != 0

        def loss_fn(b, gbi):  # verl/workers/utils/losses.py ppo_loss
            mask = b["response_mask"]
            log_prob, entropy = forward_micro_batch(self.actor_module, b, temperature, calculate_entropy)
            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                old_log_prob=b["old_log_probs"], log_prob=log_prob, advantages=b["advantages"], response_mask=mask,
                cliprange=cfg.clip_ratio, cliprange_low=cfg.clip_ratio_low, cliprange_high=cfg.clip_ratio_high,
                clip_ratio_c=cfg.clip_ratio_c, loss_agg_mode=cfg.loss_agg_mode,
                loss_scale_factor=cfg.loss_scale_factor, **gbi)
            m = {"actor/pg_loss": (pg_loss.item(), "sum"), "actor/pg_clipfrac": (pg_clipfrac.item(), "mean"),
                 "actor/ppo_kl": (ppo_kl.item(), "mean"), "actor/pg_clipfrac_lower": (pg_clipfrac_lower.item(), "mean")}
            policy_loss = pg_loss
            if calculate_entropy:
                entropy_loss = agg_loss(entropy, mask, cfg.loss_agg_mode, loss_scale_factor=cfg.loss_scale_factor, **gbi)
                policy_loss = policy_loss - cfg.entropy_coeff * entropy_loss
                m["actor/entropy_loss"] = (entropy_loss.item(), "sum")
            if cfg.use_kl_loss:  # GRPO: KL(pi || pi_ref) as a loss term instead of a reward penalty
                kld = kl_penalty(log_prob, b["ref_log_prob"], cfg.kl_loss_type)
                kl_loss = agg_loss(kld, mask, cfg.loss_agg_mode, loss_scale_factor=cfg.loss_scale_factor, **gbi)
                policy_loss = policy_loss + kl_loss * cfg.kl_loss_coef
                m["actor/kl_loss"] = (kl_loss.item(), "sum")
                m["actor/kl_coef"] = (cfg.kl_loss_coef, "mean")
            return policy_loss, m

        return ppo_train_loop(self, self.actor_module, self.actor_optimizer, data, data.meta_info["mini_batch_size"],
                              cfg.ppo_epochs, cfg.ppo_micro_batch_size_per_gpu, cfg.optim.clip_grad, loss_fn,
                              "actor", cfg.shuffle, cfg.data_loader_seed)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def update_weights(self, global_steps=None):
        """Trainer -> rollout weight sync (engine_workers.py ``update_weights``, mode="naive")."""
        self.rollout.sync_weights_from(self.actor_module, global_steps)

    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
    def get_actor_state_dict(self):
        """Only rank 0 answers (all ranks hold identical weights) -- Execute.RANK_ZERO."""
        return {k: v.detach().clone() for k, v in self.actor_module.state_dict().items()}


class CriticWorker(Worker):
    """Value model V(s_t) for every response position (engine critic / ``losses.value_loss``)."""

    def __init__(self, config: CriticConfig, model_config: ModelConfig):
        super().__init__()
        self.config, self.model_config = config, model_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self, state_dict=None):
        self.critic_module = TinyCritic(self.model_config)
        if state_dict is not None:
            self.critic_module.load_trunk_from_policy(state_dict)
        self.critic_optimizer = _optimizer(self.critic_module, self.config.optim)

    def _forward_values(self, b):
        R = b["responses"].size(1)
        values = self.critic_module(b["input_ids"], b["attention_mask"], b["position_ids"])
        return values[:, -R - 1:-1]  # V of the state *before* each response token

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @torch.no_grad()
    def compute_values(self, data: DataProto) -> DataProto:
        self.critic_module.eval()
        values = torch.cat([self._forward_values(m.batch) * m.batch["response_mask"]
                            for m in data.split(self.config.forward_micro_batch_size_per_gpu)])
        return DataProto.from_dict({"values": values})

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto) -> DataProto:
        cfg = self.config

        def loss_fn(b, gbi):  # verl/workers/utils/losses.py value_loss
            vpreds = self._forward_values(b)
            vf_loss, vf_clipfrac = compute_value_loss(vpreds, b["returns"], b["values"], b["response_mask"],
                                                      cfg.cliprange_value, cfg.loss_agg_mode, **gbi)
            return vf_loss, {"critic/vf_loss": (vf_loss.item(), "sum"),
                             "critic/vf_clipfrac": (vf_clipfrac.item(), "mean"),
                             "critic/vpred_mean": (masked_mean(vpreds, b["response_mask"]).item(), "mean")}

        return ppo_train_loop(self, self.critic_module, self.critic_optimizer, data, data.meta_info["mini_batch_size"],
                              cfg.ppo_epochs, cfg.ppo_micro_batch_size_per_gpu, cfg.optim.clip_grad, loss_fn,
                              "critic", cfg.shuffle, cfg.data_loader_seed)
