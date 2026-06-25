#!/usr/bin/env python3
"""Compute base model eval loss / perplexity (no LoRA adapter)."""

import json
import pathlib

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B"
MAX_SEQ_LENGTH = 1024
SEED = 42
OUT_PATH = pathlib.Path(__file__).parent / "lab21_lora_t4" / "base_perplexity.json"


def format_alpaca(example):
    instruction = example["instruction_vi"]
    inp = example.get("input_vi") or ""
    output = example["output_vi"]
    if inp.strip():
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{output}"
        )
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    return {"text": text}


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("⚠️ CPU only — eval sẽ chậm")

    print(f"Device: {device}, dtype: {dtype}")

    raw = load_dataset("5CD-AI/Vietnamese-alpaca-gpt4-gg-translated", split="train")
    raw = raw.shuffle(seed=SEED).select(range(200))
    ds = raw.map(format_alpaca, remove_columns=raw.column_names)
    eval_ds = ds.train_test_split(test_size=0.1, seed=SEED)["test"]
    print(f"Eval samples: {len(eval_ds)}")

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    total_loss, n = 0.0, 0
    with torch.no_grad():
        for i, row in enumerate(eval_ds):
            inputs = tokenizer(
                row["text"],
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
            ).to(device)
            out = model(**inputs, labels=inputs["input_ids"])
            total_loss += out.loss.item()
            n += 1
            if (i + 1) % 5 == 0:
                print(f"  [{i + 1}/{len(eval_ds)}] running avg loss = {total_loss / n:.4f}")

    eval_loss = total_loss / n
    perplexity = float(np.exp(eval_loss))
    result = {
        "model": MODEL_NAME,
        "eval_loss": eval_loss,
        "perplexity": perplexity,
        "n_eval": n,
        "max_seq_length": MAX_SEQ_LENGTH,
        "device": str(device),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\n✓ Base eval_loss = {eval_loss:.4f}")
    print(f"✓ Base perplexity = {perplexity:.2f}")
    print(f"✓ Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
