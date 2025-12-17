# src/train.py

from trl import SFTTrainer
from transformers import TrainingArguments
from .config import (
    OUTPUT_DIR,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    WARMUP_STEPS,
    MAX_STEPS,
)


def train_model(model, tokenizer, train_dataset):
    """
    Fine-tunes Gemma-2B using SFTTrainer (PEFT-based)
    """

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        report_to="none",
    )

    trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    dataset_text_field="sql_prompt",
)

    trainer.train()
