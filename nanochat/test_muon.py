"""
Unit tests for muon_update and batch_muon_update to verify correctness
and torch.compile compatibility.
"""

import time

import pytest
import torch

from nanochat.muon import Muon
from nanochat.batch_muon import (
    ShapeBatchedMuon,
)

def test_shape_batched_muon_equivalence():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create params
    shape = (64, 64) # Smaller shape for speed
    count = 4
    params_ref = [torch.randn(shape, device=device, requires_grad=True) for _ in range(count)]
    params_test = [p.clone().detach().requires_grad_(True) for p in params_ref]
    
    # Optimizers
    # Muon uses nesterov=True by default. ShapeBatchedMuon uses momentum=0.95.
    # batch_muon_update has nesterov=True default.
    opt_ref = Muon(params_ref, lr=0.01, momentum=0.95, nesterov=True, ns_steps=5)
    opt_test = ShapeBatchedMuon(params_test, lr=0.01, momentum=0.95)
    
    # Gradients
    grads = [torch.randn(shape, device=device) for _ in range(count)]
    for p_ref, p_test, g in zip(params_ref, params_test, grads):
        p_ref.grad = g.clone()
        p_test.grad = g.clone()
        
    # Step
    opt_ref.step()
    opt_test.step()
    
    # Compare
    for i, (p_ref, p_test) in enumerate(zip(params_ref, params_test)):
        # Tolerances for bfloat16 matrix multiplication differences
        # bfloat16 has ~2-3 decimal digits of precision.
        # Accumulation differences can be significant.
        assert torch.allclose(p_ref, p_test, atol=1e-3, rtol=1e-2), f"Param {i} mismatch. Max diff: {(p_ref - p_test).abs().max()}"

