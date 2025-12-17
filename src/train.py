from trl import SFTTrainer
from transformers import TrainingArguments
from .config import *

def formatting_func(example):
    return example["sql_prompt"]

def train_model(model, train_dataset):
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        fp16=True,
        logging_steps=5,
        save_steps=50,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=args,
        formatting_func=formatting_func,
    )

    trainer.train()
