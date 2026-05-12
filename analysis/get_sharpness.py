from datasets import load_dataset
import numpy as np
import torch
import json
import os
import argparse
from utils import TextDataset, InstructionDataset, collate_fn
from collections import deque
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
from hf_olmo.configuration_olmo import OLMoConfig
from hf_olmo.modeling_olmo import OLMoForCausalLM
from hf_olmo.tokenization_olmo_fast import OLMoTokenizerFast
torch.backends.cuda.matmul.allow_tf32 = True

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="tinygsm")
    parser.add_argument("--step", type=int, default=3000)
    parser.add_argument("--scheduler_type", type=str, default="cosine")
    parser.add_argument("--model_size", type=str, default="150M")
    parser.add_argument("--epsilon", type=float, default=1e-3)
    args = parser.parse_args()

    model_path = f"OLMo-{args.model_size}/OLMo-{args.model_size}-{args.scheduler_type}/step{args.step}-unsharded"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_config = {
        "torch_dtype": torch.float32,
        "device_map": "cuda",
        "attn_implementation": "flash_attention_2",
    }

    model = OLMoForCausalLM.from_pretrained(model_path, **model_config)
    breakpoint()
    perturbed_model = deepcopy(model)
    for name, param in perturbed_model.named_parameters():
        with torch.no_grad():
            if param.requires_grad:
                param.add_(args.epsilon * torch.randn_like(param))
    
    if args.dataset_name in ["tinygsm", "openmathinstruct-1", "openmathinstruct-2"]:
        all_prompts = []
        all_outputs = []
    else:
        all_texts = []

    with open(f"pretrain_data/{args.dataset_name}_sampled.json", "r") as f:
        if args.dataset_name in ["tinygsm", "openmathinstruct-1", "openmathinstruct-2"]:
            for line in f:
                data = json.loads(line)
                all_prompts.append(data["question"])
                all_outputs.append(data["solution"])
        else:
            for line in f:
                data = json.loads(line)
                all_texts.append(data["text"])

    if args.dataset_name in ["tinygsm", "openmathinstruct-1", "openmathinstruct-2"]:
        dataset = InstructionDataset(all_prompts, all_outputs, tokenizer)
    elif args.dataset_name in ["proofpile", "finemath-3plus"]:
        dataset = TextDataset(all_texts, tokenizer)
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer), num_workers=8, pin_memory=True)
    
    all_kl_divs = []
    from tqdm import tqdm
    rolling_kl_div = deque(maxlen=100)
    progress_bar = tqdm(dataloader)
    for batch in progress_bar:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            labels = batch["labels"][:, 1:]
            attention_mask = batch["attention_mask"]
            loss_mask = labels != -100
            labels[labels == -100] = 0
        
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :].to(torch.float32)
            logprobs = logits.log_softmax(dim=-1)
            perturbed_logits = perturbed_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :].to(torch.float32)
            perturbed_logprobs = perturbed_logits.log_softmax(dim=-1)
            kl_divs = torch.nn.functional.kl_div(perturbed_logprobs, logprobs, log_target=True, reduction="none").sum(dim=-1)
            kl_divs = (kl_divs * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)
            all_kl_divs.append(kl_divs.mean().item())
            rolling_kl_div.append(kl_divs.mean().item())
        progress_bar.set_description(f"Average KL divergence: {np.mean(rolling_kl_div).item():.6f}")

    print(f"Average KL divergence: {np.mean(all_kl_divs)}")
    results = {
        "model_size": args.model_size,
        "step": args.step,
        "scheduler_type": args.scheduler_type,
        "epsilon": args.epsilon,
        "dataset_name": args.dataset_name,
        "kl_div": max(np.mean(all_kl_divs).item(), 0.0),
    }
    import os
    os.makedirs(f"results/{args.dataset_name}", exist_ok=True)
    with open(f"results/{args.dataset_name}/perturb_sharpness_results.json", "a") as f:
        json.dump(results, f)
        f.write("\n")