# src/dataset.py
from datasets import load_dataset

DATASET_NAME = "gretelai/synthetic_text_to_sql"


def load_text2sql_dataset():
    dataset = load_dataset(DATASET_NAME)

    split_dataset = dataset["train"].train_test_split(
        test_size=0.05,
        seed=42
    )

    return split_dataset["train"], split_dataset["test"]
