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

from nanochat.common import print0

# -----------------------------------------------------------------------------
# Prompt rendering utilities

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a multiple choice question"""
    fewshot_examples = fewshot_examples or []
    parts = []
    for ex in fewshot_examples:
        parts.append(f"{ex['query']}{continuation_delimiter}{ex['choices'][ex['gold']]}")
    prefix = "\n\n".join(parts)
    if prefix:
        prefix += "\n\n"
    
    query = item['query']
    prompts = [f"{prefix}{query}{continuation_delimiter}{choice}" for choice in item['choices']]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a schema question"""
    fewshot_examples = fewshot_examples or []
    parts = []
    for ex in fewshot_examples:
        parts.append(f"{ex['context_options'][ex['gold']]}{continuation_delimiter}{ex['continuation']}")
    prefix = "\n\n".join(parts)
    if prefix:
        prefix += "\n\n"
    
    continuation = item['continuation']
    prompts = [f"{prefix}{context_option}{continuation_delimiter}{continuation}"
               for context_option in item['context_options']]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """
    Render complete prompt for a language modeling task.
    """
    fewshot_examples = fewshot_examples or []
    parts = []
    for ex in fewshot_examples:
        parts.append(f"{ex['context'].strip()}{continuation_delimiter}{ex['continuation']}")
    prefix = "\n\n".join(parts)
    if prefix:
        prefix += "\n\n"
    
    context_str = item['context'].strip()
    prompt_without_raw = f"{prefix}{context_str}{continuation_delimiter}"
    prompt_with = f"{prompt_without_raw}{item['continuation']}"
    return [prompt_without_raw.strip(), prompt_with]


def find_common_length(token_sequences, direction='left'):
    """
    Find the length of the common prefix or suffix across token sequences
    - direction: 'left' for prefix, 'right' for suffix
    """
    if not token_sequences: return 0
    n = len(token_sequences)
    if n == 1: return len(token_sequences[0])
    
    min_len = min(len(s) for s in token_sequences)
    if direction == 'left':
        for i in range(min_len):
            tok = token_sequences[0][i]
            for j in range(1, n):
                if token_sequences[j][i] != tok:
                    return i
        return min_len
    else:
        for i in range(1, min_len + 1):
            tok = token_sequences[0][-i]
            for j in range(1, n):
                if token_sequences[j][-i] != tok:
                    return i - 1
        return min_len


def stack_sequences(tokens, pad_token_id):
    """Stack up a list of token sequences, pad to longest on the right"""
    bsz = len(tokens)
    seq_len = max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.as_tensor(x, dtype=torch.long)
    return input_ids




@torch.no_grad()
def forward_model(model, input_ids, task_type):
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
        outputs.view(-1, outputs.size(-1)),
        target_ids.view(-1),
        reduction='none'
    ).view(batch_size, seq_len)
    # Set the last column to be nan because there is no autoregressive loss there
    losses[:, -1] = float('nan')
    
    # Get the argmax predictions only if needed (for language_modeling)
    # Use max instead of argmax for better performance when we only need indices
    predictions = None
    if task_type == 'language_modeling':
        predictions = outputs.argmax(dim=-1)
    return losses, predictions


# -----------------------------------------------------------------------------
# Evaluation helper functions

def _prepare_item_metadata(tasks_info, rank, world_size):
    """Render prompts and create metadata for all items across all tasks"""
    all_item_metadata = []
    all_prompts = []
    for task_info in tasks_info:
        data = task_info['data']
        task_meta = task_info['task_meta']
        label = task_info['label']
        if len(data) == 0: continue
        
        local_indices = list(range(rank, len(data), world_size))
        task_type = task_meta['task_type']
        num_fewshot = task_meta['num_fewshot']
        continuation_delimiter = task_meta['continuation_delimiter']
        all_indices_set = set(range(len(data)))
        
        for idx in local_indices:
            item = data[idx]
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
            meta = {
                'task_label': label,
                'task_type': task_type,
                'idx': idx,
                'item': item,
                'num_prompts_raw': len(prompts)
            }
            all_item_metadata.append(meta)
    return all_item_metadata, all_prompts


def _process_item_tokens(all_item_metadata, all_tokens, model_max_seq_len=None):
    """Process tokenized prompts: find indices, handle truncation, and filter for LM"""
    token_cursor = 0
    for meta in all_item_metadata:
        num_raw = meta['num_prompts_raw']
        raw_tokens = all_tokens[token_cursor : token_cursor + num_raw]
        token_cursor += num_raw
        task_type = meta['task_type']
        
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
            
        if model_max_seq_len is not None:
            new_tokens, new_start_idxs, new_end_idxs = [], [], []
            for t, s, e in zip(tokens, start_idxs, end_idxs):
                if len(t) > model_max_seq_len:
                    num_to_crop = len(t) - model_max_seq_len
                    new_tokens.append(t[-model_max_seq_len:])
                    new_start_idxs.append(s - num_to_crop)
                    new_end_idxs.append(e - num_to_crop)
                else:
                    new_tokens.append(t)
                    new_start_idxs.append(s)
                    new_end_idxs.append(e)
            tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs
            
        meta['tokens'] = tokens
        meta['start_idxs'] = start_idxs
        meta['end_idxs'] = end_idxs
        meta['num_sequences'] = len(tokens)


def _run_inference_for_task_type(model, device, task_type, indices, all_item_metadata, pad_token_id, max_tokens, results_per_task):
    """Run inference for all items of a specific task type using dynamic batching"""
    current_items = []
    current_sequences = []
    current_max_len = 0
    
    def run_inference_batch(items_to_process, sequences_to_process):
        if not sequences_to_process: return
        input_ids = stack_sequences(sequences_to_process, pad_token_id).to(device)
        
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        losses, predictions = forward_model(model, input_ids, task_type)
        if device.type == 'cuda': torch.cuda.synchronize()
        duration = time.time() - t0
        
        seq_cursor = 0
        for meta_idx in items_to_process:
            meta = all_item_metadata[meta_idx]
            label = meta['task_label']
            num_seq = meta['num_sequences']
            
            item_losses = losses[seq_cursor : seq_cursor + num_seq]
            item_input_ids = input_ids[seq_cursor : seq_cursor + num_seq]
            item_start_idxs = meta['start_idxs']
            item_end_idxs = meta['end_idxs']
            
            if task_type == 'language_modeling':
                item_predictions = predictions[seq_cursor : seq_cursor + num_seq]
                si, ei = item_start_idxs[0], item_end_idxs[0]
                predicted_tokens = item_predictions[0, si-1:ei-1]
                actual_tokens_item = item_input_ids[0, si:ei]
                is_correct = torch.all(predicted_tokens == actual_tokens_item)
            elif task_type in ['multiple_choice', 'schema']:
                item_mean_losses = torch.stack([item_losses[j, si-1:ei-1].mean()
                                for j, (si, ei) in enumerate(zip(item_start_idxs, item_end_idxs))])
                pred_idx = torch.argmin(item_mean_losses)
                is_correct = (pred_idx == meta['item']['gold'])
            
            results_per_task[label]['correct'].append((meta['idx'], is_correct.float()))
            results_per_task[label]['total_tokens'] += sum(len(s) for s in meta['tokens'])
            results_per_task[label]['inference_duration'] += duration * (num_seq / len(sequences_to_process))
            seq_cursor += num_seq

    for i in indices:
        meta = all_item_metadata[i]
        item_sequences = meta['tokens']
        num_seq = len(item_sequences)
        item_max_len = max(len(s) for s in item_sequences)
        
        new_max_len = max(current_max_len, item_max_len)
        new_num_seq = len(current_sequences) + num_seq
        
        if new_num_seq * new_max_len > max_tokens and current_sequences:
            run_inference_batch(current_items, current_sequences)
            current_items, current_sequences, current_max_len = [], [], 0
            new_max_len, new_num_seq = item_max_len, num_seq
            
        current_items.append(i)
        current_sequences.extend(item_sequences)
        current_max_len = new_max_len
    run_inference_batch(current_items, current_sequences)


@torch.no_grad()
def evaluate_task(model, tokenizer, data, device, task_meta, max_tokens=131072):
    """
    This function is responsible for evaluating one task across many examples.
    It also handles dispatch to all processes if the script is run with torchrun.
    Optimized to batch multiple examples together for faster inference.
    Returns a tuple of (accuracy, total_tokens, inference_duration).
    """
    if len(data) == 0:
        return 0.0, 0, 0.0
    
    tasks_info = [{'data': data, 'task_meta': task_meta, 'label': 'task'}]
    results = evaluate_tasks(model, tokenizer, tasks_info, device, max_tokens)
    return results['task']


@torch.no_grad()
def evaluate_tasks(model, tokenizer, tasks_info, device, max_tokens=131072):
    """
    Evaluate multiple tasks at once to maximize GPU utilization.
    tasks_info: list of dicts with 'data', 'task_meta', 'label'
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    # 1. Prepare all items across all tasks
    all_item_metadata, all_prompts = _prepare_item_metadata(tasks_info, rank, world_size)

    # 2. Tokenize all prompts at once
    pad_token_id = tokenizer.get_bos_token_id()
    all_tokens = tokenizer(all_prompts, prepend=pad_token_id) if all_prompts else []
    
    # 3. Process all tokens
    model_max_seq_len = getattr(model, 'max_seq_len', None)
    _process_item_tokens(all_item_metadata, all_tokens, model_max_seq_len)

    # 4. Group by task_type for inference
    all_task_types = sorted(list(set(info['task_meta']['task_type'] for info in tasks_info)))
    task_type_to_indices = {tt: [] for tt in all_task_types}
    for i, meta in enumerate(all_item_metadata):
        task_type_to_indices[meta['task_type']].append(i)
        
    results_per_task = {info['label']: {'correct': [], 'total_tokens': 0, 'inference_duration': 0} for info in tasks_info}

    for task_type in all_task_types:
        indices = task_type_to_indices[task_type]
        # Sort by length within task_type to minimize padding
        indices = sorted(indices, key=lambda i: max(len(s) for s in all_item_metadata[i]['tokens']), reverse=True)
        _run_inference_for_task_type(model, device, task_type, indices, all_item_metadata, pad_token_id, max_tokens, results_per_task)

    # 5. Aggregate results per task
    final_results = {}
    for task_info in tasks_info:
        label = task_info['label']
        data_len = len(task_info['data'])
        if data_len == 0:
            final_results[label] = (0.0, 0, 0.0)
            continue
            
        res = results_per_task[label]
        correct_tensor = torch.zeros(data_len, device=device)
        for idx, val in res['correct']:
            correct_tensor[idx] = val
            
        total_tokens = res['total_tokens']
        inference_duration = res['inference_duration']
        
        if world_size > 1:
            dist.barrier()
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            tt_tensor = torch.tensor([total_tokens], dtype=torch.long, device=device)
            dist.all_reduce(tt_tensor, op=dist.ReduceOp.SUM)
            total_tokens = tt_tensor.item()
            id_tensor = torch.tensor([inference_duration], dtype=torch.float32, device=device)
            dist.all_reduce(id_tensor, op=dist.ReduceOp.MAX)
            inference_duration = id_tensor.item()
            
        accuracy = correct_tensor.mean().item()
        final_results[label] = (accuracy, total_tokens, inference_duration)
        
    return final_results
