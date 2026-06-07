# # ── Install dependencies (run this cell first on Kaggle) ──────
# # !pip install -q transformers trl peft bitsandbytes accelerate datasets huggingface_hub

# import os
# import json
# import torch
# from pathlib import Path
# from datasets import Dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     BitsAndBytesConfig,
#     TrainingArguments,
# )
# from peft import LoraConfig, get_peft_model
# from trl import SFTTrainer
# from huggingface_hub import login

# # ── Config ────────────────────────────────────────────────────
# HF_TOKEN    = "hf_YOUR_TOKEN_HERE"   # ← paste your HF token
# BASE_MODEL  = "mistralai/Mistral-7B-Instruct-v0.3"
# OUTPUT_DIR  = "/kaggle/working/sec-mistral-lora"
# HF_REPO     = "your-hf-username/sec-analyst-mistral"  # ← change this
# DATASET_PATH = "/kaggle/input/sec-qa-dataset/sec_qa_dataset.jsonl"

# LORA_R          = 16
# LORA_ALPHA      = 32
# LORA_DROPOUT    = 0.05
# MAX_SEQ_LENGTH  = 1024
# NUM_EPOCHS      = 3
# BATCH_SIZE      = 4
# GRAD_ACCUM      = 4
# LEARNING_RATE   = 2e-4

# # ── Login to HF ───────────────────────────────────────────────
# login(token=HF_TOKEN)

# # ── Load dataset ──────────────────────────────────────────────
# def load_jsonl(path: str) -> Dataset:
#     records = []
#     with open(path) as f:
#         for line in f:
#             records.append(json.loads(line))
#     return Dataset.from_list(records)

# def format_prompt(example: dict) -> dict:
#     """Format as Mistral instruction format."""
#     text = f"""<s>[INST] You are a financial analyst assistant. Use the provided SEC filing context to answer the question accurately.

# {example['input']}

# {example['instruction']} [/INST] {example['output']}</s>"""
#     return {"text": text}

# dataset = load_jsonl(DATASET_PATH)
# dataset = dataset.map(format_prompt)
# split   = dataset.train_test_split(test_size=0.1, seed=42)
# train_dataset = split["train"]
# eval_dataset  = split["test"]
# print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

# # ── Load model in 4-bit ───────────────────────────────────────
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
# )

# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"

# model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL,
#     quantization_config=bnb_config,
#     device_map="auto",
# )
# model.config.use_cache = False

# # ── Apply LoRA ────────────────────────────────────────────────
# lora_config = LoraConfig(
#     r=LORA_R,
#     lora_alpha=LORA_ALPHA,
#     lora_dropout=LORA_DROPOUT,
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
#                     "gate_proj", "up_proj", "down_proj"],
# )
# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()

# # ── Training args ─────────────────────────────────────────────
# training_args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     num_train_epochs=NUM_EPOCHS,
#     per_device_train_batch_size=BATCH_SIZE,
#     gradient_accumulation_steps=GRAD_ACCUM,
#     learning_rate=LEARNING_RATE,
#     fp16=True,
#     logging_steps=10,
#     evaluation_strategy="steps",
#     eval_steps=50,
#     save_steps=100,
#     warmup_ratio=0.05,
#     lr_scheduler_type="cosine",
#     report_to="none",
# )

# # ── Train ─────────────────────────────────────────────────────
# trainer = SFTTrainer(
#     model=model,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     args=training_args,
#     dataset_text_field="text",
#     max_seq_length=MAX_SEQ_LENGTH,
# )

# print("\nStarting QLoRA fine-tuning...")
# trainer.train()
# print("Training complete!")

# # ── Save and push to HF Hub ───────────────────────────────────
# trainer.model.save_pretrained(OUTPUT_DIR)
# tokenizer.save_pretrained(OUTPUT_DIR)
# print(f"Saved to {OUTPUT_DIR}")

# trainer.model.push_to_hub(HF_REPO)
# tokenizer.push_to_hub(HF_REPO)
# print(f"Pushed to HuggingFace: https://huggingface.co/{HF_REPO}")


# ── Install dependencies (run this cell first on Kaggle) ──────
# !pip install -q transformers trl peft bitsandbytes accelerate datasets huggingface_hub

import os
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from huggingface_hub import login

# ── Config ────────────────────────────────────────────────────
HF_TOKEN    = "hf_YOUR_TOKEN_HERE"   # ← paste your HF token
BASE_MODEL  = "microsoft/Phi-3-mini-4k-instruct"  # ← Changed to Phi-3
OUTPUT_DIR  = "/kaggle/working/sec-phi3-lora"  # ← Updated name
HF_REPO     = "narendrarane50/sec-analyst-phi3"  # ← Your actual repo
DATASET_PATH = "/kaggle/input/datasets/narendrarane1/sec-qa-dataset/sec_qa_dataset.jsonl"  # ← Your actual path

LORA_R          = 8  # ← Changed from 16 to 8 (what you used)
LORA_ALPHA      = 16  # ← Changed from 32 to 16 (what you used)
LORA_DROPOUT    = 0.05
MAX_SEQ_LENGTH  = 256  # ← Reduced from 1024 (Phi-3 is smaller)
NUM_EPOCHS      = 3
BATCH_SIZE      = 1  # ← Changed from 4 to 1 (what you used)
GRAD_ACCUM      = 16  # ← Changed from 4 to 16 (what you used)
LEARNING_RATE   = 2e-4

# ── Login to HF ───────────────────────────────────────────────
login(token=HF_TOKEN)

# ── Load dataset ──────────────────────────────────────────────
def load_jsonl(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)

def format_prompt(example: dict) -> dict:
    """Format as Phi-3 instruction format."""
    text = f"""<s>[INST] You are a financial analyst assistant. Use the provided SEC filing context to answer the question accurately.
{example['input']}
{example['instruction']} [/INST] {example['output']}</s>"""
    return {"text": text}

dataset = load_jsonl(DATASET_PATH)
dataset = dataset.map(format_prompt)
split   = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]
print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

# ── Load model in 4-bit ───────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # ← Changed to bfloat16 for better stability
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False

# ── Apply LoRA ────────────────────────────────────────────────
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Training args ─────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    bf16=True,  # ← Changed to bf16
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=100,
    warmup_steps=20,  # ← Changed to warmup_steps
    lr_scheduler_type="cosine",
    report_to="none",
    optim="paged_adamw_8bit",
)

# ── Train ─────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
)

print("\nStarting QLoRA fine-tuning...")
trainer.train()
print("Training complete!")

# ── Save and push to HF Hub ───────────────────────────────────
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}")

trainer.model.push_to_hub(HF_REPO)
tokenizer.push_to_hub(HF_REPO)
print(f"Pushed to HuggingFace: https://huggingface.co/{HF_REPO}")