"""Test programmatic API for base_train.py"""
import pytest
import torch
from dataclasses import asdict

from scripts.base_train import BaseTrainConfig, BaseTrainResults


def get_available_devices():
    """Return list of available devices for testing."""
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    return devices


@pytest.fixture(params=get_available_devices())
def device(request):
    """Fixture that provides available devices (cpu and cuda if available)."""
    return request.param


def test_base_train_config_creation():
    """Test that config can be created with defaults."""
    config = BaseTrainConfig()

    assert config.depth == 20
    assert config.device_batch_size == 32
    assert config.run == "dummy"
    assert config.device_type == ""


def test_base_train_config_from_dict():
    """Test creating config from dict."""
    config = BaseTrainConfig.from_args({
        'depth': 12,
        'num_iterations': 100,
        'run': 'test-run',
    })

    assert config.depth == 12
    assert config.num_iterations == 100
    assert config.run == 'test-run'
    assert config.device_batch_size == 32  # default preserved


def test_base_train_config_from_none():
    """Test that from_args(None) returns default config."""
    config = BaseTrainConfig.from_args(None)

    assert config.depth == 20
    assert config.device_batch_size == 32


def test_base_train_config_from_config():
    """Test that from_args(config) returns the same config."""
    original = BaseTrainConfig(depth=15)
    result = BaseTrainConfig.from_args(original)

    assert result is original


def test_base_train_config_to_dict():
    """Test that config can be converted to dict."""
    config = BaseTrainConfig(depth=12, num_iterations=100)
    config_dict = asdict(config)

    assert isinstance(config_dict, dict)
    assert config_dict['depth'] == 12
    assert config_dict['num_iterations'] == 100


def test_base_train_config_with_device(device):
    """Test that config can be created with specific device type."""
    config = BaseTrainConfig(device_type=device)

    assert config.device_type == device


def test_base_train_results_creation():
    """Test that results dataclass can be created."""
    results = BaseTrainResults(
        status='success',
        final_val_bpb=1.234,
        mfu_percent=45.6,
        num_steps=100,
    )

    assert results.status == 'success'
    assert results.final_val_bpb == 1.234
    assert results.mfu_percent == 45.6
    assert results.num_steps == 100


def test_base_train_results_optional_fields():
    """Test that results can be created with minimal fields."""
    results = BaseTrainResults(status='error')

    assert results.status == 'error'
    assert results.final_val_bpb is None
    assert results.mfu_percent is None
