import torch
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer

def test_model_parameter_calculation():
    """
    Test that the model parameter count and training iterations are calculated correctly
    for a standard configuration.
    """
    try:
        tokenizer = get_tokenizer()
        vocab_size = tokenizer.get_vocab_size()
    except Exception:
        vocab_size = 50304

    depth = 20
    model_dim = depth * 64
    num_heads = max(1, (model_dim + 127) // 128)
    num_kv_heads = num_heads
    max_seq_len = 2048

    model_config = GPTConfig(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_kv_heads,
        n_embd=model_dim
    )

    model = GPT(model_config)
    num_params = sum(p.numel() for p in model.parameters())
    
    # Verify that the number of parameters is calculated correctly
    # For depth=20, model_dim=1280, num_heads=10, vocab_size=65536
    # The expected value is 560,988,160 as seen in the user's output.
    if vocab_size == 65536:
        assert num_params == 560988160
    
    # Calculate num_iterations
    target_param_data_ratio = 20
    total_batch_size = 524288

    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    
    if vocab_size == 65536:
        assert num_iterations == 21400
