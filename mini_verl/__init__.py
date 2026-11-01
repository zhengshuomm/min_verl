"""mini_verl -- a ~2,000-line, CPU-only, pure-PyTorch re-implementation of the core of verl
(the RL post-training framework), written as companion code for the book.

    from mini_verl import PPOConfig, RayPPOTrainer
    from mini_verl.sft import run_sft            # warm-start a tiny policy first (see lab 7)
"""

from .config import PPOConfig
from .protocol import DataProto
from .single_controller import Dispatch, Execute, Worker, WorkerGroup, register
from .trainer import RayPPOTrainer

__all__ = ["PPOConfig", "DataProto", "Dispatch", "Execute", "Worker", "WorkerGroup", "register", "RayPPOTrainer"]
