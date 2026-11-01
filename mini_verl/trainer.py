"""The PPO / GRPO driver: ``RayPPOTrainer.fit`` and the driver-side helpers.

Mirrors ``verl/trainer/ppo/ray_trainer.py`` (``RayPPOTrainer.init_workers`` / ``fit`` /
``_validate`` / ``_balance_batch`` / ``_update_actor`` / ``_update_critic``, ``apply_kl_penalty``,
``compute_response_mask``, ``compute_advantage``) and ``verl/trainer/ppo/metric_utils.py``
(``compute_data_metrics``, ``compute_timing_metrics``, ``reduce_metrics``). The step order of
``fit`` is verl's:

    generate -> repeat(n) + union -> balance_batch -> reward -> old_log_prob -> ref_log_prob
    -> values -> (KL in reward) -> compute_advantage -> update_critic -> update_actor
    (after critic_warmup) -> update_weights (actor -> rollout) -> validate -> metrics
"""

import time
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch

from . import core_algos
from .config import PPOConfig
from .core_algos import AdvantageEstimator, agg_loss, masked_mean
from .data import RLHFDataset, Tokenizer, collate_fn, dataloader, make_addition_pairs
from .protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from .reward import compute_reward, default_compute_score
from .single_controller import ClassWithInitArgs, ResourcePool, WorkerGroup, create_colocated_worker_cls
from .workers import ActorRolloutRefWorker, CriticWorker


@contextmanager
def marked_timer(name: str, timing_raw: Dict[str, float]):
    start = time.perf_counter()
    yield
    timing_raw[name] = timing_raw.get(name, 0.0) + time.perf_counter() - start


def reduce_metrics(metrics: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: sum(v) / len(v) for k, v in metrics.items()}


def compute_response_mask(data: DataProto) -> torch.Tensor:
    response_length = data.batch["responses"].size(1)
    return data.batch["attention_mask"][:, -response_length:]


def apply_kl_penalty(data: DataProto, kl_ctrl, kl_penalty="kl"):
    """token_level_rewards = token_level_scores - beta * KL(old || ref) per token."""
    response_mask = data.batch["response_mask"]
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)
    kld = kld * response_mask
    beta = kl_ctrl.value
    data.batch["token_level_rewards"] = data.batch["token_level_scores"] - beta * kld
    current_kl = torch.mean(masked_mean(kld, mask=response_mask, axis=-1), dim=0).item()
    kl_ctrl.update(current_kl=current_kl, n_steps=len(data))
    return data, {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1,
                      norm_adv_by_std_in_grpo=True, config=None) -> DataProto:
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"], values=data.batch["values"],
            response_mask=data.batch["response_mask"], gamma=gamma, lam=lam)
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"], response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"], norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo)
    else:
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {"token_level_rewards": data.batch["token_level_rewards"],
                      "response_mask": data.batch["response_mask"], "config": config}
        if "uid" in data.non_tensor_batch:
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        advantages, returns = adv_estimator_fn(**adv_kwargs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


def get_seqlen_balanced_partitions(seqlen_list: List[int], k_partitions: int, equal_size: bool = True):
    """Greedy longest-first partitioning into k equal-size parts with similar token sums
    (verl/utils/seqlen_balancing.py uses Karmarkar-Karp; same goal, simpler algorithm)."""
    assert equal_size and len(seqlen_list) % k_partitions == 0
    cap = len(seqlen_list) // k_partitions
    parts, sums = [[] for _ in range(k_partitions)], [0] * k_partitions
    for i in sorted(range(len(seqlen_list)), key=lambda i: (-seqlen_list[i], i)):
        j = min((p for p in range(k_partitions) if len(parts[p]) < cap), key=lambda p: (sums[p], p))
        parts[j].append(i)
        sums[j] += seqlen_list[i]
    return parts


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, float]:
    b = batch.batch
    response_length_max = b["responses"].size(1)
    response_mask = b["response_mask"].bool()
    sequence_score, sequence_reward = b["token_level_scores"].sum(-1), b["token_level_rewards"].sum(-1)
    valid_adv, valid_returns = b["advantages"][response_mask], b["returns"][response_mask]
    response_length = b["attention_mask"][:, -response_length_max:].sum(-1).float()
    prompt_length = b["attention_mask"][:, :-response_length_max].sum(-1).float()
    m = {}
    for name, t in (("score", sequence_score), ("rewards", sequence_reward), ("advantages", valid_adv),
                    ("returns", valid_returns)):
        m.update({f"critic/{name}/mean": t.mean().item(), f"critic/{name}/max": t.max().item(),
                  f"critic/{name}/min": t.min().item()})
    if use_critic:
        valid_values = b["values"][response_mask]
        m.update({"critic/values/mean": valid_values.mean().item(), "critic/values/max": valid_values.max().item(),
                  "critic/values/min": valid_values.min().item(),
                  "critic/vf_explained_var": (1.0 - torch.var(valid_returns - valid_values)
                                              / (torch.var(valid_returns) + 1e-5)).item()})
    m.update({"response_length/mean": response_length.mean().item(), "response_length/max": response_length.max().item(),
              "response_length/min": response_length.min().item(),
              "response_length/clip_ratio": (response_length == response_length_max).float().mean().item(),
              "prompt_length/mean": prompt_length.mean().item(), "prompt_length/max": prompt_length.max().item(),
              "prompt_length/min": prompt_length.min().item()})
    return m


class RayPPOTrainer:
    def __init__(self, config: PPOConfig, policy_state_dict=None, tokenizer: Optional[Tokenizer] = None,
                 reward_fn=default_compute_score):
        self.config = config
        config.actor_rollout_ref.rollout.response_length = config.data.max_response_length
        self.tokenizer = tokenizer or Tokenizer()
        self.reward_fn = reward_fn
        self.policy_state_dict = policy_state_dict
        self.use_critic = config.algorithm.adv_estimator == AdvantageEstimator.GAE  # need_critic()
        self.use_reference_policy = config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)
        d = config.data
        train_pairs, val_pairs = make_addition_pairs(d.operand_max, d.val_size, d.seed)
        self.train_dataset = RLHFDataset(train_pairs, self.tokenizer, d.max_prompt_length, d.data_source)
        self.val_dataset = RLHFDataset(val_pairs, self.tokenizer, d.max_prompt_length, d.data_source)
        self.total_training_steps = config.trainer.total_training_steps
        self.history: List[Dict[str, float]] = []

    def init_workers(self):
        """One resource pool; actor+rollout(+ref) and critic are COLOCATED in one fused worker."""
        t = self.config.trainer
        resource_pool = ResourcePool([t.n_gpus_per_node] * t.nnodes, n_gpus_per_node=t.n_gpus_per_node)
        role = "actor_rollout_ref" if self.use_reference_policy else "actor_rollout"
        class_dict = {"actor_rollout": ClassWithInitArgs(ActorRolloutRefWorker, config=self.config.actor_rollout_ref,
                                                         role=role)}
        if self.use_critic:
            class_dict["critic"] = ClassWithInitArgs(CriticWorker, config=self.config.critic,
                                                     model_config=self.config.actor_rollout_ref.model)
        wg_dict = WorkerGroup(resource_pool=resource_pool, ray_cls_with_init=create_colocated_worker_cls(class_dict))
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
        if self.use_critic:
            self.critic_wg = spawn_wg["critic"]
            self.critic_wg.init_model(self.policy_state_dict)
        self.actor_rollout_wg = spawn_wg["actor_rollout"]
        self.actor_rollout_wg.init_model(self.policy_state_dict)

    # ------------------------------------------------------------------ driver-side helpers
    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        return batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])

    def _validate(self) -> Dict[str, float]:
        """Greedy decoding on the held-out set; reward mean = accuracy."""
        batch = DataProto.from_single_dict(collate_fn([self.val_dataset[i] for i in range(len(self.val_dataset))]))
        gen_batch = self._get_gen_batch(batch)
        gen_batch.meta_info = {"do_sample": False, "validate": True}
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
        output = unpad_dataproto(self.actor_rollout_wg.generate_sequences(gen_batch_padded), pad_size)
        batch = batch.union(output)
        _, extra = compute_reward(batch, self.tokenizer, self.reward_fn)
        acc = sum(extra["acc"]) / len(extra["acc"])
        return {f"val-core/{self.config.data.data_source}/reward/mean@1": acc}

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder rows so every DP rank gets a similar number of tokens (rows are chunked in order)."""
        global_seqlen_lst = batch.batch["attention_mask"].sum(-1).tolist()
        dp_size = self.actor_rollout_wg.world_size
        partitions = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=dp_size)
        for idx, partition in enumerate(partitions):  # small at both ends (pipeline bubbles)
            partition.sort(key=lambda x: (global_seqlen_lst[x], x))
            partitions[idx] = partition[::2] + partition[1::2][::-1]
        batch.reorder(torch.tensor([j for p in partitions for j in p]))
        sums = [sum(global_seqlen_lst[j] for j in p) for p in partitions]
        metrics.update({f"{logging_prefix}/min": min(global_seqlen_lst), f"{logging_prefix}/max": max(global_seqlen_lst),
                        f"{logging_prefix}/balanced_min": min(sums), f"{logging_prefix}/balanced_max": max(sums)})

    def _update_actor(self, batch: DataProto) -> DataProto:
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        n = self.config.actor_rollout_ref.rollout.n
        batch.meta_info["mini_batch_size"] = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * n
        return self.actor_rollout_wg.update_actor(batch)

    def _update_critic(self, batch: DataProto) -> DataProto:
        batch.meta_info["mini_batch_size"] = self.config.critic.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        return self.critic_wg.update_critic(batch)

    # ------------------------------------------------------------------ the training loop
    def fit(self, log_every: int = 1):
        """Run ``trainer.total_training_steps`` PPO steps; print every ``log_every`` steps (0 = silent)."""
        cfg = self.config
        rollout_n = cfg.actor_rollout_ref.rollout.n
        self.global_steps = 0
        self.actor_rollout_wg.update_weights(self.global_steps)
        if cfg.trainer.val_before_train:
            val_metrics = self._validate()
            self.history.append({"training/global_step": 0, **val_metrics})
            if log_every:
                print(f"step   0 | val acc {next(iter(val_metrics.values())):.3f}")
        self.global_steps += 1
        train_loader = dataloader(self.train_dataset, cfg.data.train_batch_size, seed=cfg.trainer.seed)
        for batch_dict in train_loader:
            metrics, timing_raw = {}, {}
            batch = DataProto.from_single_dict(batch_dict)
            batch.meta_info["temperature"] = cfg.actor_rollout_ref.rollout.temperature
            batch.non_tensor_batch["uid"] = [f"{self.global_steps}-{i}" for i in range(len(batch))]  # verl: uuid4
            gen_batch = self._get_gen_batch(batch)
            gen_batch.meta_info["global_steps"] = self.global_steps
            gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)
            is_last_step = self.global_steps >= self.total_training_steps
            with marked_timer("step", timing_raw):
                with marked_timer("gen", timing_raw):
                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                batch = batch.repeat(repeat_times=rollout_n, interleave=True)
                batch = batch.union(gen_batch_output)
                if "response_mask" not in batch.batch:
                    batch.batch["response_mask"] = compute_response_mask(batch)
                if cfg.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)
                batch.meta_info["global_token_num"] = batch.batch["attention_mask"].sum(-1).tolist()
                with marked_timer("reward", timing_raw):
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.tokenizer, self.reward_fn)
                with marked_timer("old_log_prob", timing_raw):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    entropys = old_log_prob.batch.pop("entropys")
                    metrics["actor/entropy"] = agg_loss(entropys, batch.batch["response_mask"],
                                                        cfg.actor_rollout_ref.actor.loss_agg_mode).item()
                    batch = batch.union(old_log_prob)
                if self.use_reference_policy:
                    with marked_timer("ref", timing_raw):
                        batch = batch.union(self.actor_rollout_wg.compute_ref_log_prob(batch))
                if self.use_critic:
                    with marked_timer("values", timing_raw):
                        batch = batch.union(self.critic_wg.compute_values(batch))
                with marked_timer("adv", timing_raw):
                    batch.batch["token_level_scores"] = reward_tensor
                    batch.non_tensor_batch.update(reward_extra_infos_dict)
                    if cfg.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward,
                                                             kl_penalty=cfg.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                    batch = compute_advantage(batch, adv_estimator=cfg.algorithm.adv_estimator,
                                              gamma=cfg.algorithm.gamma, lam=cfg.algorithm.lam, num_repeat=rollout_n,
                                              norm_adv_by_std_in_grpo=cfg.algorithm.norm_adv_by_std_in_grpo,
                                              config=cfg.algorithm)
                if self.use_critic:
                    with marked_timer("update_critic", timing_raw):
                        critic_output = self._update_critic(batch)
                    metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))
                if cfg.trainer.critic_warmup <= self.global_steps:
                    with marked_timer("update_actor", timing_raw):
                        actor_output = self._update_actor(batch)
                    with marked_timer("update_weights", timing_raw):  # actor -> rollout weight sync
                        self.actor_rollout_wg.update_weights(self.global_steps)
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
                if cfg.trainer.test_freq > 0 and (is_last_step or self.global_steps % cfg.trainer.test_freq == 0):
                    with marked_timer("testing", timing_raw):
                        metrics.update(self._validate())
            metrics["training/global_step"] = self.global_steps
            metrics.update(compute_data_metrics(batch, use_critic=self.use_critic))
            metrics.update({f"timing_s/{k}": v for k, v in timing_raw.items()})
            self.history.append(metrics)
            val_step = any(k.startswith("val-core/") for k in metrics)
            if log_every and (self.global_steps % log_every == 0 or val_step):
                self._log(metrics)
            if is_last_step:
                return self.history
            self.global_steps += 1

    def _log(self, m):
        val = [v for k, v in m.items() if k.startswith("val-core/")]
        line = (f"step {m['training/global_step']:>3} | score {m['critic/score/mean']:.3f} | "
                f"pg_loss {m.get('actor/pg_loss', float('nan')):+.4f} | entropy {m['actor/entropy']:.3f} | "
                f"resp_len {m['response_length/mean']:.2f} | {m['timing_s/step']:.2f}s")
        print(line + (f" | val acc {val[0]:.3f}" if val else ""))
