from datasets import load_dataset
import json

# ds = load_dataset("xavierdurawa/proof-pile-2-streaming", 'default', split="train", streaming=True)
ds = load_dataset("TinyGSM/TinyGSM", split="train", streaming=True)
i = 0
all_examples = []
for sample in ds:
    all_examples.append({"question": sample["question"], "solution": sample["code"]})
    i += 1
    if i >= 10_000:
        break

with open("pretrain_data/tinygsm_sampled.json", "w") as f:
    for example in all_examples:
        f.write(json.dumps(example) + "\n")