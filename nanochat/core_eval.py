"""
Functions for evaluating the CORE metric, as described in the DCLM paper.
https://arxiv.org/abs/2406.11794

TODOs:
- All tasks ~match except for squad. We get 31% reference is 37%. Figure out why.
"""
import logging
import random
import time

from jinja2 import Template
import torch
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Prompt rendering utilities

# Pre-compile templates to avoid recompilation overhead
_TEMPLATE_MC = Template("""
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip())

_TEMPLATE_SCHEMA = Template("""
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip())

_TEMPLATE_LM = Template("""
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip())

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a multiple choice question"""
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [_TEMPLATE_MC.render(choice=choice, **context) for choice in item['choices']]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a schema question"""
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [_TEMPLATE_SCHEMA.render(context=context_option, **context)
               for context_option in item['context_options']]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """
    Render complete prompt for a language modeling task.
    Notice that we manually trim the context in the template,
    which in some datasets seems to have trailing whitespace (which we don't want).
    """
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    # Return two prompts: without and with the continuation
    prompt_without = _TEMPLATE_LM.render(include_continuation=False, **context)
    prompt_with = _TEMPLATE_LM.render(include_continuation=True, **context)
    # Due to the way the data seems to be stored, I think I need to strip in the case of LM here.
    # Otherwise we may get trailing whitespaces in prompt_without (which get absorbed into the next
    # token in prompt_with), meaning we don't get a nice and clean prefix in the token space
    # to detect the final continuation. Tokenizers...
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction='left'):
    """
    Find the length of the common prefix or suffix across token sequences
    - direction: 'left' for prefix, 'right' for suffix
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len-1, -1)
    }[direction]
    # Find the first position where the token sequences differ
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    """Stack up a list of token sequences, pad to longest on the right"""
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids




@torch.no_grad()
def forward_model(model, input_ids):
    """
    Take BxT tensor of token ids, return BxT tensor of losses and argmax predictions.
    The last column of losses is set to nan because we don't have autoregressive targets there.
    """
    batch_size, seq_len = input_ids.size()
    outputs = model(input_ids)
    # Roll the tensor to the left by one position to get the (autoregressive) target ids
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    # Calculate cross entropy at all positions
    losses = torch.nn.functional.cross_entropy(
        outputs.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction='none'
    ).view(batch_size, seq_len)
    # Set the last column to be nan because there is no autoregressive loss there
    losses[:, -1] = float('nan')
    # Get the argmax predictions at each position
    predictions = outputs.argmax(dim=-1)
    return losses, predictions


@torch.no_grad()
def evaluate_batch(indices, model, tokenizer, data, device, task_meta):
    """
    Evaluate a batch of examples efficiently.
    Returns a list of boolean results corresponding to each index.
    """
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']
    
    # 1. Render all prompts
    all_prompts = []
    item_metadata = [] # Stores (idx, item, num_prompts_raw)
    
    data_len = len(data)
    all_indices_set = set(range(data_len))

    for idx in indices:
        item = data[idx]
        # Sample few-shot
        fewshot_examples = []
        if num_fewshot > 0:
            rng = random.Random(1234 + idx)
            available_indices = list(all_indices_set - {idx})
            fewshot_indices = rng.sample(available_indices, num_fewshot)
            fewshot_examples = [data[i] for i in fewshot_indices]
        
        if task_type == 'multiple_choice':
            prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        elif task_type == 'schema':
            prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        elif task_type == 'language_modeling':
            prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
            
        all_prompts.extend(prompts)
        item_metadata.append({'idx': idx, 'item': item, 'num_prompts_raw': len(prompts)})

    # 2. Tokenize all
    all_tokens = tokenizer(all_prompts, prepend=tokenizer.get_bos_token_id())
    
    # 3. Process tokens (find indices, truncate, filter for LM)
    flat_tokens = []
    flat_start_idxs = []
    flat_end_idxs = []
    
    token_cursor = 0
    for meta in item_metadata:
        num_raw = meta['num_prompts_raw']
        raw_tokens = all_tokens[token_cursor : token_cursor + num_raw]
        token_cursor += num_raw
        
        # Logic from batch_sequences_* and truncation
        if task_type == 'multiple_choice':
            answer_start_idx = find_common_length(raw_tokens, direction='left')
            start_idxs = [answer_start_idx] * len(raw_tokens)
            end_idxs = [len(x) for x in raw_tokens]
            tokens = raw_tokens
        elif task_type == 'schema':
            suffix_length = find_common_length(raw_tokens, direction='right')
            end_idxs = [len(x) for x in raw_tokens]
            start_idxs = [ei - suffix_length for ei in end_idxs]
            tokens = raw_tokens
        elif task_type == 'language_modeling':
            tokens_without, tokens_with = raw_tokens
            start_idx, end_idx = len(tokens_without), len(tokens_with)
            tokens = [tokens_with]
            start_idxs = [start_idx]
            end_idxs = [end_idx]
            
        # Truncation
        if hasattr(model, 'max_seq_len') and model.max_seq_len is not None:
            max_tokens = model.max_seq_len
            new_tokens, new_start_idxs, new_end_idxs = [], [], []
            for t, s, e in zip(tokens, start_idxs, end_idxs):
                if len(t) > max_tokens:
                    num_to_crop = len(t) - max_tokens
                    new_tokens.append(t[-max_tokens:])
                    new_start_idxs.append(s - num_to_crop)
                    new_end_idxs.append(e - num_to_crop)
                else:
                    new_tokens.append(t)
                    new_start_idxs.append(s)
                    new_end_idxs.append(e)
            tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs
            
        flat_tokens.extend(tokens)
        flat_start_idxs.extend(start_idxs)
        flat_end_idxs.extend(end_idxs)
        meta['num_sequences'] = len(tokens) # Store how many sequences this item resulted in

    # 4. Stack and Forward
    pad_token_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(flat_tokens, pad_token_id)
    input_ids = input_ids.to(device)
    
    losses, predictions = forward_model(model, input_ids)
    
    # 5. Calculate results
    batch_results = torch.zeros(len(item_metadata), device=device)
    result_cursor = 0
    
    for i, meta in enumerate(item_metadata):
        num_seq = meta['num_sequences']
        item_losses = losses[result_cursor : result_cursor + num_seq]
        item_predictions = predictions[result_cursor : result_cursor + num_seq]
        item_input_ids = input_ids[result_cursor : result_cursor + num_seq]
        
        # Need start/end idxs for this item
        item_start_idxs = flat_start_idxs[result_cursor : result_cursor + num_seq]
        item_end_idxs = flat_end_idxs[result_cursor : result_cursor + num_seq]
        
        result_cursor += num_seq
        
        if task_type == 'language_modeling':
            si = item_start_idxs[0]
            ei = item_end_idxs[0]
            predicted_tokens = item_predictions[0, si-1:ei-1]
            actual_tokens = item_input_ids[0, si:ei]
            is_correct = torch.all(predicted_tokens == actual_tokens)
        elif task_type in ['multiple_choice', 'schema']:
            item_mean_losses = torch.stack([item_losses[j, si-1:ei-1].mean()
                            for j, (si, ei) in enumerate(zip(item_start_idxs, item_end_idxs))])
            pred_idx = torch.argmin(item_mean_losses)
            is_correct = (pred_idx == meta['item']['gold'])
            
        batch_results[i] = is_correct.float()
    
    return batch_results




def evaluate_task(model, tokenizer, data, device, task_meta, batch_size=8):
    """
    This function is responsible for evaluating one task across many examples.
    It also handles dispatch to all processes if the script is run with torchrun.
    Optimized to batch multiple examples together for faster inference.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
        
    # stride the examples to each rank
    local_indices = list(range(rank, len(data), world_size))
    
    # Process examples in batches for efficiency
    batch_num = 0
    for batch_start in range(0, len(local_indices), batch_size):
        batch_indices = local_indices[batch_start:batch_start + batch_size]
        batch_results = evaluate_batch(batch_indices, model, tokenizer, data, device, task_meta)
        
        correct[batch_indices] = batch_results
        
        batch_num += len(batch_indices)
    
    # sync results across all the processes if running distributed
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    # compute the mean
    mean_correct = correct.mean().item()
    return mean_correct
