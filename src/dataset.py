# src/dataset.py

from datasets import load_dataset
from .config import DATASET_NAME, TEXT_COLUMN, SQL_COLUMN


def load_text2sql_dataset():
    """
    Loads the gretelai synthetic Text-to-SQL dataset
    Uses only sql_prompt and sql columns (as per paper)
    """

    dataset = load_dataset(DATASET_NAME)

    dataset = dataset["train"].train_test_split(
        train_size=100_000,
        test_size=5_000,
        seed=42
    )

    return dataset["train"], dataset["test"]
