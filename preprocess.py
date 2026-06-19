import json
from datasets import Dataset
from transformers import AutoTokenizer

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "distilgpt2"
DATASET_PATH = "dataset/high_quality_story_dataset.json"
MAX_LENGTH = 512

# -----------------------------
# Load Dataset
# -----------------------------
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    stories = json.load(f)

print(f"Loaded {len(stories)} stories")

# -----------------------------
# Convert to Instruction Format
# -----------------------------
formatted_data = []

for item in stories:
    text = f"""### Instruction:
{item['instruction']}

### Story:
{item['story']}
"""

    formatted_data.append({"text": text})

print("\nSample Training Example:\n")
print(formatted_data[0]["text"][:500])

# -----------------------------
# Create HF Dataset
# -----------------------------
dataset = Dataset.from_list(formatted_data)

print("\nDataset Created")
print(dataset)

# -----------------------------
# Load Tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# GPT-2 has no PAD token
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Tokenization Function
# -----------------------------
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

# -----------------------------
# Tokenize Dataset
# -----------------------------
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)

# -----------------------------
# Save Tokenized Dataset
# -----------------------------
tokenized_dataset.save_to_disk(
    "dataset/tokenized_dataset"
)

print("\nTokenized dataset saved successfully!")
print("Path: dataset/tokenized_dataset")