# src/config.py

MODEL_NAME = "google/gemma-2b"

DATASET_NAME = "gretelai/synthetic_text_to_sql"
TEXT_COLUMN = "sql_prompt"
SQL_COLUMN = "sql"

MAX_LENGTH = 512

# Training configuration 
PER_DEVICE_TRAIN_BATCH_SIZE = 3
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
WARMUP_STEPS = 10
MAX_STEPS = 250

# LoRA configuration
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

DEVICE = "cuda"
RANDOM_SEED = 42

OUTPUT_DIR = "./outputs"
