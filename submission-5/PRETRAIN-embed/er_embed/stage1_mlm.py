
"""Stage 1 of 02c: masked-language-model pretraining of a small BERT (one process per GPU under torchrun)."""
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import BertConfig, BertModel

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
world = int(os.environ.get("WORLD_SIZE", 1))
rank = int(os.environ.get("RANK", 0))
local = int(os.environ.get("LOCAL_RANK", 0))
cuda = torch.cuda.is_available()
if world > 1:
    dist.init_process_group("nccl" if cuda else "gloo", timeout=datetime.timedelta(minutes=10))
device = torch.device(f"cuda:{local}" if cuda else "cpu")
if cuda:
    torch.cuda.set_device(device)
torch.manual_seed(cfg["seed"] + rank)
main = rank == 0
fp16 = cuda and cfg["fp16"]

tokens = np.memmap(cfg["tokens_file"], dtype=np.uint16, mode="r")
offsets = np.load(cfg["offsets_file"])
lengths = np.diff(offsets)
n = len(lengths)
pad, cls_id, sep_id, mask_id = cfg["pad_id"], cfg["cls_id"], cfg["sep_id"], cfg["mask_id"]
per_gpu = max(1, cfg["batch_size"] // world)


def epoch_batches(epoch):
    """This process's batches for one epoch: rows shuffled, sorted by length within groups of 50 batches, the batches
    shuffled again, then split evenly between the processes."""
    r = np.random.default_rng(cfg["seed"] + epoch)
    order = r.permutation(n)
    group, batches = per_gpu * 50, []
    for s in range(0, n, group):
        g = order[s:s + group]
        g = g[np.argsort(lengths[g], kind="stable")]
        batches += [g[i:i + per_gpu] for i in range(0, len(g) - per_gpu + 1, per_gpu)]
    batches = [batches[i] for i in r.permutation(len(batches))]
    steps = len(batches) // world
    return batches[rank * steps:(rank + 1) * steps]


def make_batch(rows):
    """Padded token ids of the rows, with 15% of the tokens hidden (80% [MASK], 10% random, 10% unchanged)."""
    ids = np.full((len(rows), int(lengths[rows].max())), pad, np.int64)
    for k, i in enumerate(rows):
        ids[k, :lengths[i]] = tokens[offsets[i]:offsets[i + 1]]
    ids = torch.from_numpy(ids).to(device)
    attention = ids != pad
    hidden = (torch.rand(ids.shape, device=device) < cfg["mlm_prob"]) & attention & (ids != cls_id) & (ids != sep_id)
    labels = torch.where(hidden, ids, torch.full_like(ids, -100))
    roll = torch.rand(ids.shape, device=device)
    inputs = torch.where(hidden & (roll < 0.8), torch.full_like(ids, mask_id), ids)
    inputs = torch.where(hidden & (roll >= 0.8) & (roll < 0.9),
                         torch.randint_like(ids, cfg["n_special"], cfg["vocab_size"]), inputs)
    return inputs, attention.long(), labels


config = BertConfig(vocab_size=cfg["vocab_size"], hidden_size=cfg["hidden"], num_hidden_layers=cfg["layers"],
                    num_attention_heads=cfg["heads"], intermediate_size=4 * cfg["hidden"],
                    max_position_embeddings=cfg["max_len"], type_vocab_size=2, pad_token_id=pad)


class MaskedLM(torch.nn.Module):
    """BERT encoder + prediction head tied to the input embeddings, scored only at the hidden positions."""

    def __init__(self):
        super().__init__()
        self.bert = BertModel(config, add_pooling_layer=False)
        h = config.hidden_size
        self.transform = torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.GELU(),
                                             torch.nn.LayerNorm(h, eps=config.layer_norm_eps))
        self.bias = torch.nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, input_ids, attention_mask, labels):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        picked = labels != -100
        logits = self.transform(out[picked]) @ self.bert.embeddings.word_embeddings.weight.t() + self.bias
        target = labels[picked]
        return F.cross_entropy(logits.float(), target), (logits.argmax(-1) == target).float().mean()


model = MaskedLM().to(device)
core = model
if world > 1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local] if cuda else None)
params = list(core.named_parameters())
opt = torch.optim.AdamW([{"params": [p for _, p in params if p.ndim > 1], "weight_decay": 0.01},
                         {"params": [p for _, p in params if p.ndim <= 1], "weight_decay": 0.0}],
                        lr=cfg["lr"], betas=(0.9, 0.98), eps=1e-6)
steps_per_epoch = len(epoch_batches(0))
total = max(1, int(steps_per_epoch * cfg["epochs"]))
warmup = max(1, int(total * cfg["warmup_share"]))
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min((s + 1) / warmup, max(0.0, (total - s) / max(1, total - warmup))))
try:
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)
except (AttributeError, TypeError):                  # PyTorch < 2.3
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)
if main:
    print(f"{world} process(es) x {per_gpu} texts per step | {n:,} texts, {total:,} steps "
          f"({cfg['epochs']:g} epoch(s)) | {sum(p.numel() for _, p in params) / 1e6:.1f}M parameters", flush=True)


def save():
    core.bert.save_pretrained(cfg["out_dir"])


log, step, epoch, t0 = [], 0, 0, time.time()
sum_loss = sum_acc = torch.zeros((), device=device)
t_last, last_step = time.time(), 0
while step < total:
    for rows in epoch_batches(epoch):
        if step >= total:
            break
        inputs, attention, labels = make_batch(rows)
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            loss, acc = model(inputs, attention, labels)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        sum_loss, sum_acc = sum_loss + loss.detach(), sum_acc + acc.detach()
        if step % cfg["log_every"] == 0 or step == total:
            k = step - last_step
            entry = {"step": step, "loss": round(sum_loss.item() / k, 4), "masked accuracy": round(sum_acc.item() / k, 4),
                     "lr": sched.get_last_lr()[0]}
            if main:
                minutes = (time.time() - t0) / 60
                speed = k * per_gpu * world / max(time.time() - t_last, 1e-9)
                print(f"step {step:,}/{total:,}  loss {entry['loss']:.4f}  masked accuracy {entry['masked accuracy']:.3f}  "
                      f"lr {entry['lr']:.2e}  {speed:,.0f} texts/s  {minutes:.1f} min, "
                      f"~{minutes / step * (total - step):.0f} min left", flush=True)
                log.append(entry)
            sum_loss = sum_acc = torch.zeros((), device=device)
            t_last, last_step = time.time(), step
        if main and step % cfg["save_every"] == 0:
            save()
    epoch += 1

if main:
    save()
    pd.DataFrame(log).to_csv(cfg["log_file"], index=False)
    Path(cfg["done_file"]).write_text(json.dumps({"minutes": round((time.time() - t0) / 60, 1), "processes": world,
                                                  "steps": total, "texts": n}), encoding="utf-8")
if world > 1:
    dist.barrier()                                   # the other process waits until the model is saved
    dist.destroy_process_group()
