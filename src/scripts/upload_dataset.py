"""Upload the sentiment dataset to the HuggingFace Hub.

Usage:
    >>> python src/scripts/upload_dataset.py
"""


from datasets import load_dataset

DATASET_FOLDER_PATH = "data/processed"
DATASET_HF_PATH = "oliverkinch/sentiment"


dataset = load_dataset(DATASET_FOLDER_PATH)
dataset.push_to_hub(DATASET_HF_PATH, private=True)
