import torch
import pandas as pd
from copy import deepcopy
from collections import Counter
import torch.nn as nn
from math import sqrt

@torch.no_grad()
def get_delta_dict(finetuned_model, base_model):
    delta_dict = {}
    for name, param in finetuned_model.named_parameters():
        delta_dict[name] = param - base_model.get_parameter(name)
    return delta_dict

def read_parquet(file_path):
    df = pd.read_parquet(file_path)
    ground_truths = df['reward_model'].tolist()
    ground_truths = [gt['ground_truth'] for gt in ground_truths]
    prompts = df['prompt'].tolist()
    outputs = df['output'].tolist()
    all_prompts = []
    all_outputs = []
    all_ground_truths = []
    for prompt, output_lst, gt in zip(prompts, outputs, ground_truths):
        for output in output_lst:
            all_prompts.append(prompt)
            all_outputs.append(output)
            all_ground_truths.append(gt)
    return all_prompts, all_outputs, all_ground_truths


@torch.no_grad()
def l2_norm(model: nn.Module):
    norm = 0.0
    for name, param in model.named_parameters():
        norm += torch.norm(param).item()**2
    return sqrt(norm)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        input_ids = []
        labels = []
        self.tokenizer.truncation_side = "right"
        input_ids= self.tokenizer(self.texts[idx], truncation=True, add_special_tokens=False, max_length=2048)["input_ids"]
        labels = deepcopy(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids)
        }

class InstructionDataset(torch.utils.data.Dataset):
    def __init__(self, prompts, outputs, tokenizer):
        self.prompts = prompts
        self.outputs = outputs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        input_ids = []
        labels = []
        self.tokenizer.truncation_side = "left"
        prompt_enc = self.tokenizer(self.prompts[idx], truncation=True, add_special_tokens=False, max_length=1024)["input_ids"]
        input_ids.extend(prompt_enc)
        labels.extend([-100] * len(prompt_enc))
        self.tokenizer.truncation_side = "right"
        output_enc = self.tokenizer(self.outputs[idx], truncation=True, add_special_tokens=False, max_length=1024)["input_ids"]
        input_ids.extend(output_enc)
        labels.extend(output_enc)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids)
        }


def collate_fn(batch, tokenizer):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    labels = []
    attention_mask = []
    all_prompt_lengths = []
    all_output_lengths = []

    for item in batch:
        padding_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * padding_len)
        attention_mask.append(item["attention_mask"] + [0] * padding_len)
        # prompt length is the first token that is not -100
        prompt_length = next(i for i, label in enumerate(item["labels"]) if label != -100)
        all_prompt_lengths.append(prompt_length)
        output_length = len(item["labels"]) - prompt_length
        all_output_lengths.append(output_length)
        labels.append(item["labels"] + [-100] * padding_len)

    return {
        "prompt_lengths": torch.tensor(all_prompt_lengths),
        "output_lengths": torch.tensor(all_output_lengths),
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }