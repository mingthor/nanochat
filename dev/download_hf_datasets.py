import os

# Ensure we are NOT in offline mode for the download step
# These MUST be set before importing datasets or transformers
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"

from datasets import load_dataset

def download():
    datasets_to_download = [
        ("HuggingFaceTB/smol-smoltalk", None, ["train", "test"]),
        ("cais/mmlu", "all", ["test"]),
        ("cais/mmlu", "auxiliary_train", ["train"]),
        ("openai/gsm8k", "main", ["train", "test"]),
        ("openai/openai_humaneval", None, ["test"]),
        ("allenai/ai2_arc", "ARC-Easy", ["train", "validation", "test"]),
        ("allenai/ai2_arc", "ARC-Challenge", ["train", "validation", "test"]),
    ]

    print("Pre-downloading datasets to local cache...")
    for path, name, splits in datasets_to_download:
        for split in splits:
            print(f"Downloading {path} (name={name}, split={split})...")
            try:
                load_dataset(path, name, split=split)
            except Exception as e:
                print(f"Failed to download {path} {name} {split}: {e}")
    print("All datasets downloaded successfully.")

if __name__ == "__main__":
    download()
