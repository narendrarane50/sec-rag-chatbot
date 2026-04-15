import json
import random

# Load your full dataset
with open('finetuning/data/sec_qa_dataset.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

print(f"Total samples: {len(data)}")

# Shuffle and split
random.seed(42)
random.shuffle(data)

# 80% train, 10% eval, 10% test
train_size = int(0.8 * len(data))
eval_size = int(0.1 * len(data))

train_data = data[:train_size]
eval_data = data[train_size:train_size + eval_size]
test_data = data[train_size + eval_size:]

# Save splits
with open('finetuning/data/train.jsonl', 'w') as f:
    for item in train_data:
        f.write(json.dumps(item) + '\n')

with open('finetuning/data/eval.jsonl', 'w') as f:
    for item in eval_data:
        f.write(json.dumps(item) + '\n')

with open('finetuning/data/test.jsonl', 'w') as f:
    for item in test_data:
        f.write(json.dumps(item) + '\n')

print(f"Train: {len(train_data)}")
print(f"Eval: {len(eval_data)}")
print(f"Test: {len(test_data)}")
print("\nFiles created:")
print("- finetuning/data/train.jsonl")
print("- finetuning/data/eval.jsonl")
print("- finetuning/data/test.jsonl")