
from collections import defaultdict, namedtuple
import logging
import math
import os

import torch
from torch import Tensor
import torch.distributed as dist
from nanochat.muon import Muon, zeropower_via_newtonschulz5


logger = logging.getLogger(__name__)


def batch_muon_update(grad, momentum, beta=0.95, nesterov=True):
    """
    Perform Muon update on gradients.
    Args:
        grad: Gradient tensor. Can be a single parameter gradient (2D or 4D) or a batch
              of parameter gradients (3D or 5D where first dim is batch size).
        momentum: Momentum buffer matching the shape of grad.
        beta: Momentum coefficient.
        nesterov: Whether to use Nesterov momentum.
    Returns:
        Update tensor with the same shape as grad.
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum

    # Handle different input shapes:
    # - 2D: single param [rows, cols] -> already 2D
    # - 3D: batched params [batch, rows, cols] -> already 3D
    # - 4D: single conv filters [out_ch, in_ch, h, w] -> [out_ch, in_ch*h*w]
    # - 5D: batched conv filters [batch, out_ch, in_ch, h, w] -> [batch, out_ch, in_ch*h*w]
    if update.ndim == 4:  # single conv filter
        update = update.view(len(update), -1)
    elif update.ndim == 5:  # batched conv filters
        batch_size = update.size(0)
        update = update.view(batch_size, update.size(1), -1)

    update = zeropower_via_newtonschulz5(update, 5)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    update = update.view(grad.shape)
    return update

ParamInBatch = namedtuple("ParamInBatch", ["param", "lr", "weight_decay", "momentum", "rms_norm_scale"])
TensorSpec = namedtuple("TensorSpec", ["shape", "dtype"])

class ShapeBatchedMuon(torch.optim.Optimizer):
    """
    Muon optimizer that groups parameters by shape for batched processing.
    """

    def __init__(
        self, params, lr=0.02, weight_decay=0, momentum=0.95, rms_norm_scale=False
    ):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, rms_norm_scale=rms_norm_scale)
        super().__init__(params, defaults)
        self.muon_sharding_dim = "intra_node"
        self.initialize()

    def initialize(self):
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            if self.muon_sharding_dim in ("intra_node", "hsdp") and world_size % GPUS_PER_NODE == 0:
                start_rank = self.local_rank // GPUS_PER_NODE * GPUS_PER_NODE
                intra_node_group_ranks = list(range(start_rank, start_rank + GPUS_PER_NODE))
                self.muon_pg = dist.new_group(intra_node_group_ranks)
                self.local_rank = self.local_rank % GPUS_PER_NODE
            else:
                self.local_rank = dist.get_rank()
                self.muon_pg = torch.distributed.group.WORLD
            logger.info(f"ShapeBatchedMuon using {self.muon_sharding_dim=} with {self.muon_pg.size() if self.muon_pg else 1=}")
        else:
            self.muon_pg = None
            self.local_rank = 0

        # Group params by shape
        all_batches = []
        for group in self.param_groups:
            params_by_spec = defaultdict(list)
            for p in group["params"]:
                params_by_spec[TensorSpec(shape=p.shape, dtype=p.dtype)].append(
                    ParamInBatch(
                        param=p,
                        lr=group["lr"],
                        weight_decay=group["weight_decay"],
                        momentum=group["momentum"],
                        rms_norm_scale=group["rms_norm_scale"],
                    )
                )
            # Each value in params_by_spec is a batch of same-shaped parameters
            for batch in params_by_spec.values():
                all_batches.append(batch)

        # Try to balance the number of batches to fit on the specified process group.
        group_size = self.muon_pg.size() if self.muon_pg is not None else 1
        while len(all_batches) % group_size != 0:
            # Find the largest batch to split
            largest_idx = max(range(len(all_batches)), key=lambda i: len(all_batches[i]))
            batch = all_batches[largest_idx]
            if len(batch) < 2:
                break  # Can't split further
            mid = len(batch) // 2
            all_batches[largest_idx : largest_idx + 1] = [batch[:mid], batch[mid:]]

        if len(all_batches) % group_size != 0:
            raise ValueError(
                f"Number of shape-batched groups {len(all_batches)} is not divisible along dim {self.muon_sharding_dim} with group size {group_size}."
            )
        self.param_groups_schedule = all_batches
        logger.info(
            f"ShapeBatchedMuon initialized with {len(self.param_groups_schedule)} shape-batched groups, sizes {[len(g) for g in self.param_groups_schedule]}"
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Iterate through param_groups_schedule in steps of self.group_size
        group_size = self.muon_pg.size() if self.muon_pg is not None else 1
        num_groups = len(self.param_groups_schedule)
        for base_i in range(0, num_groups, group_size):
            hsdp_group_batches = self.param_groups_schedule[base_i : base_i + group_size]
            hsdp_group_updates = []
            for j, batch in enumerate(hsdp_group_batches):
                grads = []
                momentum_buffers = []
                for param_in_batch in batch:
                    p = param_in_batch.param
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    grads.append(p.grad)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    momentum_buffers.append(state["momentum_buffer"])

                if j == self.local_rank % group_size:
                    # Stack gradients and get/create momentum buffers
                    grad_batch = torch.stack(grads, dim=0)
                    momentum_batch = torch.stack(momentum_buffers, dim=0)
                    update_batch = batch_muon_update(grad_batch, momentum_batch, beta=batch[0].momentum)
                    for k, param_in_batch in enumerate(batch):
                        p = param_in_batch.param
                        self.state[p]["momentum_buffer"].copy_(momentum_batch[k])
                        p.grad.copy_(grad_batch[k])
                else:
                    shape = (len(grads),) + grads[0].shape
                    update_batch = torch.zeros(shape, dtype=torch.bfloat16, device=grads[0].device)
                hsdp_group_updates.append(update_batch)

            if self.muon_pg is not None:
                torch.distributed.all_gather(
                    hsdp_group_updates, hsdp_group_updates[self.local_rank % group_size], group=self.muon_pg
                )

            for j, batch in enumerate(hsdp_group_batches):
                update_batch = hsdp_group_updates[j]
                for k, param_in_batch in enumerate(batch):
                    p = param_in_batch.param
                    p.mul_(1 - param_in_batch.lr * param_in_batch.weight_decay)
                    if param_in_batch.rms_norm_scale:
                        p.add_(
                            0.2 * math.sqrt(max(p.shape)) * update_batch[k], alpha=-param_in_batch.lr
                        )  # RMSNorm scaling to keep updates Muon similar scale to Adam (https://arxiv.org/pdf/2502.16982)
                    else:
                        p.add_(update_batch[k], alpha=-param_in_batch.lr)

        return loss

GPUS_PER_NODE = 8
class IntraNodeMuon(torch.optim.Optimizer):
    def __init__(self, param_groups, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, momentum=0.95, rms_norm_scale=False):
        for group in param_groups:
            group["lr"] = group.get("lr", lr)
            group["momentum"] = group.get("momentum", momentum)
            group["weight_decay"] = group.get("weight_decay", weight_decay)
            group["rms_norm_scale"] = group.get("rms_norm_scale", rms_norm_scale)

        super().__init__(param_groups, dict())
        if dist.is_available() and dist.is_initialized():
            self.global_rank = dist.get_rank()
            world_size = dist.get_world_size()
            if world_size % GPUS_PER_NODE == 0:
                self.group_size = GPUS_PER_NODE
                start_rank = self.global_rank // GPUS_PER_NODE * GPUS_PER_NODE
                intra_node_group_ranks = list(range(start_rank, start_rank + self.group_size))
                self.used_pg = dist.new_group(intra_node_group_ranks)
            else:
                self.group_size = world_size
                self.used_pg = dist.default_pg
        else:
            # Default values for single-rank or non-distributed setup
            self.global_rank = 0
            self.group_size = 1
            self.used_pg = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        local_rank = self.global_rank % self.group_size
        pending_work = None
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (self.group_size - len(params) % self.group_size)

            for base_i in range(len(params))[:: self.group_size]:
                if base_i + local_rank < len(params):
                    p = params[base_i + local_rank]
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    if group["rms_norm_scale"]:
                        p.add_(
                            0.2 * math.sqrt(max(p.shape)) * update.reshape(p.shape), alpha=-group["lr"]
                        )  # RMSNorm scaling to keep updates Muon similar scale to Adam (https://arxiv.org/pdf/2502.16982)
                    else:
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])

                if self.used_pg is not None:
                    pending_work = dist.all_gather(
                        params_pad[base_i : base_i + self.group_size],
                        params_pad[base_i + local_rank],
                        async_op=True,
                        group=self.used_pg,
                    )
        if pending_work is not None:
            pending_work.wait()

        return loss
    