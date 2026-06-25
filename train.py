import json
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# =====================================
# Configuration
# =====================================

MODEL_NAME = "distilgpt2"
DATASET_PATH = "dataset/high_quality_story_dataset.json"
OUTPUT_DIR = "model/fine_tuned_model"

MAX_LENGTH = 512

# =====================================
# Load Dataset
# =====================================

print("Loading dataset...")

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    stories = json.load(f)

formatted_data = []

for item in stories:

    text = f"""### Instruction:
{item['instruction']}

### Story:
{item['story']}
"""

    formatted_data.append({"text": text})

dataset = Dataset.from_list(formatted_data)

print(f"Loaded {len(dataset)} examples")

# =====================================
# Tokenizer
# =====================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# GPT2 has no PAD token
tokenizer.pad_token = tokenizer.eos_token

# =====================================
# Tokenization
# =====================================

def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# =====================================
# Model
# =====================================

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.resize_token_embeddings(len(tokenizer))

# =====================================
# Data Collator
# =====================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# =====================================
# Training Arguments
# =====================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    overwrite_output_dir=True,

    num_train_epochs=5,

    per_device_train_batch_size=2,

    save_steps=100,

    save_total_limit=2,

    logging_steps=10,

    learning_rate=5e-5,

    weight_decay=0.01,

    warmup_steps=20,

    fp16=torch.cuda.is_available(),

    report_to="none"
)

# =====================================
# Trainer
# =====================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# =====================================
# Train
# =====================================

print("\nStarting Training...\n")

trainer.train()

# =====================================
# Save Model
# =====================================

print("\nSaving model...")

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print("\nTraining Complete!")
print(f"Model saved at: {OUTPUT_DIR}")