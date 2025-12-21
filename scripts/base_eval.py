"""
Evaluate the CORE metric for a given model.

Run on a single GPU:
python -m scripts.base_eval

Run with torchrun on e.g. 8 GPUs:
torchrun --nproc_per_node=8 -m scripts.base_eval

The script will print the CORE metric to the console.
"""
import os
import csv
import time
import json
import yaml
import shutil
import random
import zipfile
import tempfile
from contextlib import nullcontext

import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, download_file_with_lock
from nanochat.tokenizer import HuggingFaceTokenizer
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import evaluate_task, evaluate_tasks

# -----------------------------------------------------------------------------
# nanochat specific function dealing with I/O etc.

# ~162MB of data needed to evaluate the CORE metric
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def place_eval_bundle(file_path):
    # here file_path is the path to the eval_bundle.zip file
    # we need to unzip it and place it in the base directory
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")

def evaluate_model(model, tokenizer, device, max_per_task=-1, max_tokens=131072):
    """
    Evaluate a base model on the CORE benchmark.
    - max_per_task: crop the data to this many examples per task for testing (-1 = disable)
    - max_tokens: max tokens per batch for evaluation
    """
    # Load config and task metadata
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    # Download the eval bundle to disk (and unzip if needed)
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # Load random baseline values from eval metadata
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

    # Estimate FLOPs per token for MFU calculation
    num_flops_per_token = None
    # Get the original model if it's compiled
    raw_model = getattr(model, '_orig_mod', model)
    if hasattr(raw_model, 'estimate_flops'):
        num_flops_per_token = raw_model.estimate_flops() / 3 # forward only
    elif hasattr(raw_model, 'model') and hasattr(raw_model.model, 'estimate_flops'):
        num_flops_per_token = raw_model.model.estimate_flops() / 3
    
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    promised_flops_per_sec_h100 = 989e12 * world_size

    # Evaluate each task
    results = {}
    centered_results = {}
    total_eval_tokens = 0
    total_inference_duration = 0
    
    tasks_info = []
    for task in tasks:
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }

        # Load data for this task
        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # shuffle the data because in many cases it appears ordered but we want
        # the ability to only run a subset of the data for debugging purposes etc.
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]
        
        tasks_info.append({'data': data, 'task_meta': task_meta, 'label': label})

    start_eval_time = time.time()
    all_results = evaluate_tasks(model, tokenizer, tasks_info, device, max_tokens=max_tokens)
    total_eval_duration = time.time() - start_eval_time

    for task_info in tasks_info:
        label = task_info['label']
        data = task_info['data']
        accuracy, total_tokens, inference_duration = all_results[label]

        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        
        total_eval_tokens += total_tokens
        total_inference_duration += inference_duration

    core_metric = sum(centered_results.values()) / len(centered_results)
    
    # Print summary
    avg_tok_per_sec = total_eval_tokens / total_inference_duration if total_inference_duration > 0 else 0
    avg_mfu = "n/a"
    if num_flops_per_token is not None and total_inference_duration > 0:
        avg_flops_per_sec = num_flops_per_token * total_eval_tokens / total_inference_duration
        avg_mfu = f"{100 * avg_flops_per_sec / promised_flops_per_sec_h100:.2f}%"
    
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
        "total_tokens": total_eval_tokens,
        "total_duration": total_eval_duration,
        "avg_tok_per_sec": avg_tok_per_sec,
        "avg_mfu": avg_mfu,
    }
    return out

# -----------------------------------------------------------------------------
# HuggingFace loading utilities and light wrappers for a model

class ModelWrapper:
    """Lightweight wrapper for a HuggingFace model"""
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids):
        outputs = self.model(input_ids)
        logits = outputs.logits
        return logits

def load_hf_model(hf_path: str, device):
    print0(f"Loading model from: {hf_path}")
    # Load the model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "openai-community/gpt2" in hf_path else None
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    # Load the tokenizer
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer

# -----------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf-path', type=str, default=None, help='HuggingFace model path to evaluate')
    parser.add_argument('--max-per-task', type=int, default=-1, help='Max examples per task to evaluate (-1 = disable)')
    parser.add_argument('--max-tokens', type=int, default=131072, help='Max tokens per batch for evaluation')
    args = parser.parse_args()

    # distributed / precision setup
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    start_time = time.time()
    # Load model and tokenizer from command line or from file system
    if args.hf_path is not None:
        # atm assume that if a path is given, it's a huggingface model path
        hf_path = args.hf_path
        print0(f"Loading huggingface model from: {hf_path}")
        model, tokenizer = load_hf_model(hf_path, device)
        model_name = hf_path # just for logging
        model_slug = hf_path.replace("/", "-") # for the output csv file
    else:
        # load a local model from the file system
        model, tokenizer, meta = load_model("base", device, phase="eval")
        model_name = f"base_model (step {meta['step']})" # just for logging
        model_slug = f"base_model_{meta['step']:06d}" # for the output csv file

    # Compile the model for faster evaluation
    model = torch.compile(model)
    init_duration = time.time() - start_time
    print0(f"Data loading and model compilation took {init_duration:.4f}s")

    # Evaluate the model
    with autocast_ctx:
        out = evaluate_model(model, tokenizer, device, max_per_task=args.max_per_task, max_tokens=args.max_tokens)

    # Write out the results to a csv file
    core_metric = None
    centered_results = {}
    if ddp_rank == 0:
        base_dir = get_base_dir()
        output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        results = out["results"]
        centered_results = out["centered_results"]
        core_metric = out["core_metric"]
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
            f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
            for label in results:
                f.write(f"{label:<35}, {results[label]:<10.6f}, {centered_results[label]:<10.6f}\n")
            f.write(f"{'CORE':<35}, {'':<10}, {core_metric:<10.6f}\n")
        # Print the content of the csv file to console too
        print0("="*80)
        print0(f"Model: {model_name}")
        print0("="*80)
        with open(output_csv_path, 'r', encoding='utf-8') as f:
            print0(f.read())
        print0(f"Total time: {out['total_duration']:.2f}s")
        print0(f"Average tokens/sec: {out['avg_tok_per_sec']:.1f}")
        print0(f"Average MFU: {out['avg_mfu']}")

    # Log to report
    from nanochat.report import get_report
    get_report().log(section="Base model evaluation", data=[
        {
            "Model": model_name,
            "CORE metric": core_metric,
            "Total tokens": out.get("total_tokens"),
            "Total time": f"{out.get('total_duration'):.2f}s",
            "Avg tok/s": f"{out.get('avg_tok_per_sec'):.1f}",
            "Avg MFU": out.get("avg_mfu"),
        },
        centered_results, # the full table
    ])

    compute_cleanup()

if __name__ == "__main__":
    main()
