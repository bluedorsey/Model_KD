"""LoRA / QLoRA supervised fine-tuning on (prompt, response) pairs. GPU; run on the training machine.

One script for both roles:
  teacher canary adapter (step 2):  --model Qwen/Qwen2.5-7B-Instruct --qlora --data canary-adapter
  student (steps 5, 16):           --model Qwen/Qwen2.5-1.5B-Instruct --data <responses.parquet>
  hardware timing (step 1):        add --time-steps 20 (prints s/step and exits)

`--data canary-adapter` reads Data/Processed/canary/items.parquet split=adapter (prompt -> full_response).
A parquet path must have columns prompt, response. Loss is on response tokens only. fp16 + GradScaler
unless the card supports bf16 (Quadro RTX 5000 does not). Needs: uv sync --extra gpu

  .venv/Scripts/python.exe -m student.train_lora --model Qwen/Qwen2.5-0.5B-Instruct --data canary-adapter --out runs/t05
Then gate it: python -m teacher.eval_canary --kind local --model <base> --adapter <out>
"""
import argparse
import json
import random
import time
from pathlib import Path

import duckdb
import torch

from configs import load


def pairs(spec, cfg):
    if spec == "canary-adapter":
        q = f"SELECT prompt, full_response FROM '{(cfg['processed'] / 'canary' / 'items.parquet').as_posix()}' WHERE split = 'adapter'"
    else:
        q = f"SELECT prompt, response FROM '{Path(spec).as_posix()}'"
    return duckdb.sql(q).fetchall()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/lora")
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-steps", type=int, default=0)
    a = ap.parse_args()
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cfg = load()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    tok = AutoTokenizer.from_pretrained(a.model)
    kw = {"torch_dtype": dtype}
    if a.qlora:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, device_map="auto", **kw)
    if a.qlora:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    if not a.qlora and not bf16:
        for p in model.parameters():  # fp16 base, fp32 trainable LoRA weights for GradScaler
            if p.requires_grad:
                p.data = p.data.float()

    def encode(prompt, response):
        p = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True)
        r = tok(response + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p + r)[: a.max_len]
        return ids, ([-100] * len(p) + r)[: a.max_len]

    data = [encode(p, r) for p, r in pairs(a.data, cfg)]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    scaler = torch.amp.GradScaler(enabled=not bf16)
    steps = a.time_steps or int(a.epochs * len(data) / a.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))
    model.train()
    t0, order = time.time(), []
    for step in range(steps):
        if len(order) < a.batch:
            order += random.sample(range(len(data)), len(data))
        batch, order = [data[i] for i in order[: a.batch]], order[a.batch:]
        L = max(len(x) for x, _ in batch)
        ids = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x, _ in batch], device=model.device)
        lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in batch], device=model.device)
        att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in batch], device=model.device)
        with torch.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if step % 50 == 0 or step == steps - 1:
            print(f"step {step}/{steps} loss {loss.item():.4f} {(time.time() - t0) / (step + 1):.2f}s/step", flush=True)
    meta = {"model": a.model, "data": a.data, "examples": len(data), "steps": steps, "qlora": a.qlora,
            "dtype": str(dtype), "seed": a.seed, "s_per_step": round((time.time() - t0) / steps, 3),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    print(json.dumps(meta))
    if not a.time_steps:
        model.save_pretrained(a.out)
        Path(a.out, "train_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
