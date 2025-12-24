import os

# Ensure we are NOT in offline mode for the download step
# These MUST be set before importing datasets or transformers
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"

from datasets import load_dataset
from nanochat.common import download_file_with_lock, get_base_dir

def download():
    # 1. Download Hugging Face datasets
    # List of (path, name) tuples
    datasets_to_download = [
        ("HuggingFaceTB/smol-smoltalk", None),
        ("openai/gsm8k", "main"),
        ("openai/gsm8k", "socratic"),
        ("openai/openai_humaneval", None),
        ("allenai/ai2_arc", "ARC-Easy"),
        ("allenai/ai2_arc", "ARC-Challenge"),
        ("cais/mmlu", "auxiliary_train"),
        ("cais/mmlu", "all"),
    ]

    print("Pre-downloading datasets to local cache...")
    for path, name in datasets_to_download:
        try:
            print(f"Processing {path} (name={name})...")
            load_dataset(path, name)
        except Exception as e:
            print(f"Failed to process {path} {name}: {e}")

    # 2. Download identity conversations for SFT
    print("Downloading identity conversations...")
    identity_url = "https://raw.githubusercontent.com/TrelisResearch/nanochat/master/identity_conversations.jsonl"
    try:
        download_file_with_lock(identity_url, "identity_conversations.jsonl")
    except Exception as e:
        print(f"Failed to download identity conversations: {e}")

    # 3. Download word list for SpellingBee
    print("Downloading word list for SpellingBee...")
    word_list_url = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
    try:
        download_file_with_lock(word_list_url, "words_alpha.txt")
    except Exception as e:
        print(f"Failed to download word list: {e}")

    print("All datasets processed.")

if __name__ == "__main__":
    download()
