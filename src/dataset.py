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
    test_size=0.05,
    seed=42
)

    return dataset["train"], dataset["test"]
